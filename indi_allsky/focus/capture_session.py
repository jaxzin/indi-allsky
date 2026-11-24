"""What the capture worker builds for a focus session: the link, the service, and the camera steps around a session.

`FocusGlue` holds the camera and turns a page's `start` into a `FocusController`:
  1. abort the exposure in progress (it is discarded);
  2. save the camera state next to the link's socket (video only: stills change no camera property);
  3. build the picture source for the rig and the controller.
When the session ends, `on_end` stops nothing itself (the controller already stopped the source), restores the saved
state, deletes the file and asks capture to reconfigure the camera for its normal schedule.

No import of PyIndi, Flask or capture: tests drive this with fake cameras.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable
from typing import Optional

from indi_allsky.focus import rigs
from indi_allsky.focus.controller import CameraInfo
from indi_allsky.focus.controller import FocusController
from indi_allsky.focus.controller import FocusService
from indi_allsky.focus.exposure_search import ExposureSearch
from indi_allsky.focus.exposure_search import default_ceiling
from indi_allsky.focus.indi_stream import IndiStreamSource
from indi_allsky.focus.indi_stream import has_video
from indi_allsky.focus.indi_stream import remove_state
from indi_allsky.focus.indi_stream import restore_state
from indi_allsky.focus.indi_stream import save_state
from indi_allsky.focus.indi_stream import write_state
from indi_allsky.focus.link import LinkError
from indi_allsky.focus.link import LinkServer
from indi_allsky.focus.session import SessionClock
from indi_allsky.focus.sources import StillsSource


logger = logging.getLogger('indi_allsky')


DEFAULT_ADDRESS = '/var/lib/indi-allsky/focus-session.sock'
STATE_SUFFIX = '.state.json'
VIDEO_START_EXPOSURE_S = 0.1
SHORTEST_FLOOR_S = 1e-6      # a driver that reports a minimum exposure of 0 or less
VIDEO_REGION_PX = 1024       # the first region: a centred square, at most this size


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
                 epoch: Callable[[], float] = time.time):
        self.camera = camera
        self.link = link
        self.ccd_info = ccd_info or {}
        self.cfa = cfa
        self.state_path = state_path
        self._gain = gain
        self._reconfigure = reconfigure
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

        self.camera.abortCcdExposure()  # the normal exposure in progress is discarded

        shortest = _ccd(self.ccd_info, 'CCD_EXPOSURE', 'CCD_EXPOSURE_VALUE', 'min')
        if shortest is None or shortest <= 0:
            shortest = SHORTEST_FLOOR_S
        ceiling = default_ceiling(_ccd(self.ccd_info, 'CCD_EXPOSURE', 'CCD_EXPOSURE_VALUE', 'max'))
        ceiling = max(ceiling, shortest)

        self._saved = None
        try:
            if rig.picture == 'video':
                simulated = callable(getattr(self.camera, 'start_video', None))
                if not simulated:
                    self._saved = save_state(self.camera)
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
            return FocusController(
                source, self.link, SessionClock(), rig, False,
                epoch=self._epoch, exposure=start, gain=float(self._gain()), roi=roi,
                search=ExposureSearch(shortest, ceiling, start),
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
        if getattr(self.camera, 'disconnected', False) or getattr(self.camera, 'ccd_removed', False):
            # the camera is gone: the state file stays, and capture restores it when it starts again
            logger.warning('Focus: the camera is not connected; its settings are restored at the next start')
            return
        try:
            if self._saved is not None:
                failures = restore_state(self.camera, self._saved)
                if failures:
                    logger.error('Focus: %d camera properties could not be restored', failures)
                # deleted even after a failure: a later startup would only repeat the same restore
                remove_state(self.state_path)
                self._saved = None
        finally:
            self._reconfigure()
