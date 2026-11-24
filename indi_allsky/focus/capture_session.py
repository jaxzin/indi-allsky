"""What the capture worker builds for a focus session: the link, the service, and the camera steps around a session.

`FocusGlue` holds the camera and turns a page's `start` into a `FocusController`:
  1. read the camera state (video only: stills change no stream property); a driver without a required property
     refuses the start here, before the camera is touched;
  2. abort the exposure in progress (it is discarded) and write the state next to the link's socket;
  3. build the picture source for the rig and the controller.
When the session ends, `on_end` stops nothing itself (the controller already stopped the source), restores the saved
state, deletes the file (kept when a required property failed, for the next startup), clears the client's blob sink
and asks capture to reconfigure the camera for its normal schedule.

No import of PyIndi, Flask or capture: tests drive this with fake cameras.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Callable
from typing import Optional

from indi_allsky.focus import rigs
from indi_allsky.focus.controller import CameraInfo
from indi_allsky.focus.controller import FocusController
from indi_allsky.focus.controller import FocusService
from indi_allsky.focus.controller import MSG_MISSING_PROPERTY
from indi_allsky.focus.controller import StartRefused
from indi_allsky.focus.exposure_search import ExposureSearch
from indi_allsky.focus.exposure_search import default_ceiling
from indi_allsky.focus.indi_stream import IndiStreamSource
from indi_allsky.focus.indi_stream import MissingProperty
from indi_allsky.focus.indi_stream import REQUIRED
from indi_allsky.focus.indi_stream import REQUIRED_VIDEO
from indi_allsky.focus.indi_stream import has_video
from indi_allsky.focus.indi_stream import read_state
from indi_allsky.focus.indi_stream import remove_state
from indi_allsky.focus.indi_stream import restore_state
from indi_allsky.focus.indi_stream import save_state
from indi_allsky.focus.indi_stream import write_state
from indi_allsky.focus.link import LinkError
from indi_allsky.focus.link import LinkServer
from indi_allsky.focus.session import SessionClock
from indi_allsky.focus.sources import StillsSource


logger = logging.getLogger('indi_allsky')


STATE_SUFFIX = '.state.json'
VIDEO_START_EXPOSURE_S = 0.1
SHORTEST_FLOOR_S = 1e-6      # a driver that reports a minimum exposure of 0 or less
VIDEO_REGION_PX = 1024       # the first region: a centred square, at most this size
RESTORE_NOTICE_KEY = 'focus_restore'
MSG_RESTORE_FAILED = ('The focus session could not put back the camera setting {0}. indi-allsky tries again when '
                      'capture restarts; until then pictures may come out wrong.')


def state_path_for(address: str) -> str:
    """The saved camera state sits next to the socket: focus-session.sock -> focus-session.state.json."""
    base, ext = os.path.splitext(address)
    return (base if ext == '.sock' else address) + STATE_SUFFIX


def open_link(address: str) -> Optional[LinkServer]:
    """The link, listening; None when it cannot be (another capture holds it, or the path is unusable)."""
    link = LinkServer(address)
    try:
        link.start()
    except (LinkError, OSError) as e:
        logger.warning('Focus: the live focus link is not available (%s); the focus page shows the classic view',
                       type(e).__name__)
        try:
            link.close()
        except Exception:
            pass
        return None
    return link


def _ccd(ccd_info: dict, prop: str, elem: str, key: str) -> Optional[float]:
    try:
        value = ccd_info[prop][elem][key]
    except (KeyError, TypeError):
        return None
    return None if value is None else float(value)


def _gain_info(ccd_info: dict, key: str) -> Optional[float]:
    """ccd_info['GAIN_INFO'] is one level shallower than the INDI properties: {'min': .., 'max': ..}."""
    try:
        value = ccd_info['GAIN_INFO'][key]
    except (KeyError, TypeError):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _gain_limits(ccd_info: dict):
    """(min, max) of the camera's gain, or (None, None) when the range means nothing: indi.py reports min = max = -1
    for a camera without gain control, and an upside-down range is no range. A fixed gain (min == max >= 0) stays."""
    low = _gain_info(ccd_info, 'min')
    high = _gain_info(ccd_info, 'max')
    if low is None or high is None or high < 0 or high < low:
        return None, None
    return low, high


def default_region(width: int, height: int):
    """A centred square of at most VIDEO_REGION_PX, on the Bayer grid."""
    size = min(VIDEO_REGION_PX, width, height)
    size -= size % 2
    x = (width - size) // 2
    y = (height - size) // 2
    return x - x % 2, y - y % 2, size, size


class FocusGlue(object):

    def __init__(self, camera, link, *, camera_interface: str, device_name: str, ccd_info: dict, cfa,
                 gain: Callable[[], float], state_path: str, reconfigure: Callable[[], None],
                 exposure_max: Optional[float] = None,
                 notify: Optional[Callable[[str, str], None]] = None,
                 epoch: Callable[[], float] = time.time):
        """exposure_max: config CCD_EXPOSURE_MAX, the longest exposure the session may pick (at most 15 s).
        notify(key, message): raises an indi-allsky notification; called when a required camera property could not
        be put back."""
        self.camera = camera
        self.link = link
        self.ccd_info = ccd_info or {}
        self.cfa = cfa
        self.state_path = state_path
        self._gain = gain
        self._reconfigure = reconfigure
        self._exposure_max = exposure_max
        self._notify = notify
        self._epoch = epoch
        video = has_video(camera)
        rig = rigs.decide(camera_interface, device_name, video, link_ok=True)
        self.info = CameraInfo(interface=camera_interface, device_name=device_name, driver_has_video=video, rig=rig)
        self.service = FocusService(link, self.info, self.make_controller, on_end=self.on_end)
        self._saved = None        # the camera state of the running session (video on a real driver only)

    # -- start ---------------------------------------------------------------------------------------------------------

    def make_controller(self, cid: int) -> FocusController:
        rig = self.info.rig
        if rig.picture not in ('video', 'stills'):
            raise RuntimeError('this camera cannot run a focus session')

        self._saved = None
        simulated = callable(getattr(self.camera, 'start_video', None))
        saved = None
        if rig.picture == 'video' and not simulated:
            try:
                saved = save_state(self.camera, video=True)  # reads only: a refusal leaves the camera as it was
            except MissingProperty as e:
                raise StartRefused(MSG_MISSING_PROPERTY.format(e.name))
            left = read_state(self.state_path)
            if left is not None:
                # an earlier session could not put a required property back: the camera holds the session's
                # values, so the file, not the camera, has the settings to return to
                logger.warning('Focus: keeping the camera settings an earlier session could not put back')
                saved = left

        self.camera.abortCcdExposure()  # the normal exposure in progress is discarded

        driver_shortest = _ccd(self.ccd_info, 'CCD_EXPOSURE', 'CCD_EXPOSURE_VALUE', 'min')
        shortest = driver_shortest
        if shortest is None or shortest <= 0:
            shortest = SHORTEST_FLOOR_S
        ceiling = default_ceiling(self._exposure_max)  # the config's longest exposure, not the driver's
        ceiling = max(ceiling, shortest)

        try:
            if rig.picture == 'video':
                if saved is not None:
                    self._saved = saved
                    write_state(self.state_path, self._saved)
                width = _ccd(self.ccd_info, 'CCD_FRAME', 'WIDTH', 'max')
                height = _ccd(self.ccd_info, 'CCD_FRAME', 'HEIGHT', 'max')
                sensor = (int(width), int(height)) if width and height else None
                bits = _ccd(self.ccd_info, 'CCD_INFO', 'CCD_BITSPERPIXEL', 'current')
                source = IndiStreamSource(self.camera, sensor=sensor, bits=int(bits) if bits else None,
                                          bayer=self.cfa)
                roi = default_region(*sensor) if sensor else None
                start = min(max(VIDEO_START_EXPOSURE_S, shortest), ceiling)
            else:
                source = StillsSource(self.camera, self._swap_image_q)
                roi = None
                start = shortest
            gain_min, gain_max = _gain_limits(self.ccd_info)
            return FocusController(
                source, self.link, SessionClock(), rig, False,
                epoch=self._epoch, exposure=start, gain=float(self._gain()), roi=roi,
                search=ExposureSearch(shortest, ceiling, start),
                # the same sources capture checks its own settings against (capture.py: CCD_EXPOSURE, GAIN_INFO)
                exposure_min=driver_shortest if driver_shortest is not None and driver_shortest > 0 else None,
                exposure_max=ceiling,
                gain_min=gain_min,
                gain_max=gain_max,
            )
        except Exception:
            if self._saved is not None:
                remove_state(self.state_path)
                self._saved = None
            raise

    def _swap_image_q(self, new_q):
        old = self.camera.image_q
        self.camera.image_q = new_q
        return old

    # -- end -----------------------------------------------------------------------------------------------------------

    def on_end(self, controller: FocusController) -> None:
        """After the controller stopped the source: put the camera back for the normal schedule."""
        gone = getattr(self.camera, 'disconnected', False) or getattr(self.camera, 'ccd_removed', False)
        try:
            if gone:
                # the state file stays, and capture restores it when it starts again
                logger.warning('Focus: the camera is not connected; its settings are restored at the next start')
                return
            if self._saved is not None:
                saved, self._saved = self._saved, None
                failures = restore_state(self.camera, saved)
                if failures:
                    logger.error('Focus: %d camera properties could not be restored', len(failures))
                required = [name for name in failures if name in REQUIRED + REQUIRED_VIDEO]
                if required:
                    # the state file stays: capture tries the restore again when it starts
                    logger.error('Focus: a required camera property was not restored: %s', ', '.join(required))
                    self._raise_notice(MSG_RESTORE_FAILED.format(', '.join(required)))
                else:
                    # an optional property that failed would only fail again at the next startup
                    remove_state(self.state_path)
        finally:
            if getattr(self.camera, 'blob_sink', None) is not None:
                self.camera.blob_sink = None  # capture's exposures go to the image queue again
            if not gone:
                self._reconfigure()

    def _raise_notice(self, message: str) -> None:
        if self._notify is None:
            return
        try:
            self._notify(RESTORE_NOTICE_KEY, message)
        except Exception as e:
            logger.error('Focus: the notification could not be raised (%s)', type(e).__name__)
