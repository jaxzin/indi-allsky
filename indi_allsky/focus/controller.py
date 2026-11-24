"""The focus session in the capture process: commands in, pictures and readings out.

`FocusService` lives as long as capture does and is the only reader of `LinkServer.poll_commands()`. With no session
it answers `ping` and `hello`, and turns a `start` from a page with controls into a `FocusController`.

`FocusController` runs one session. Its worker thread takes each `RawFrame` from the picture source, in order, and
when it falls behind it skips to the newest waiting frame (frame numbers still count the skipped ones):

    hot-pixel fix -> gray -> measure (picked stars) or detect (star count) -> preview copy -> clock -> JPEG
    -> link.broadcast_frame, then link.broadcast_json(reading)

Threading: every change of session state happens on the caller's thread (`handle`, `service`, `begin`, `end`), and so
does every call into the source but one. The worker never calls the source; it posts requests (a pick result, an
exposure-search step, a failure) that `service()` applies. `_lock` covers the rules, the ledger, the star set and the
generation counters; the worker takes it once per frame, only to copy a snapshot.

The one exception is `source.whole_frame()`, which can wait many seconds for a picture. A `focus-preview` thread makes
that call and the JPEG, then posts the reply for `service()` to send, so commands (`ping`, `done`) are answered and the
session rules keep ticking meanwhile. One preview at a time: a second `preview` while one is pending gets an `error`.
`end()` stops the source, which releases a waiting `whole_frame()`, and joins the thread; a reply still pending then is
dropped, so nothing follows `ended`.

Saturated stars: a star whose raw peak reaches SATURATION_FRACTION (0.98) of full scale is clipped, so its measured
size is wrong. The peak is the maximum of the hot-pixel-fixed raw (before gray) in a MEASURE_BOX square around the
star. It is checked twice:
  - at the pick, as the median peak over the grading frames. Such a star stays in the set with the state `saturated`
    in the `stars` message, so the page can show why it is not used. It is not dropped, because a shorter exposure
    can bring it back: the per-frame check below decides each frame.
  - on every frame: a star clipped in that frame never adds to `size`, whatever its state, and its ring says
    `saturated`. A star marked `saturated` at the pick counts again on a frame where it is not clipped. When no star
    is used and a wanted one was clipped, the reading's message is MSG_SATURATED.

A page may command only when the hello it registered with carries `'controls': True`. `ping` and `hello` are open to
every page; every other command is refused to a watcher with an `error`.

Messages from a page (`kind` and fields):
    start                           begin a session (FocusService); during one, answered with `state`
    done                            end the session
    ping        id?                 -> pong
    hello                           -> state
    keep_going                      reset the idle timer
    set_camera  exposure, gain, cmd_id?      exposure in s (0 < e <= 3600), gain >= 0
    set_region  roi: [x, y, w, h] | null, cmd_id?   snapped to the Bayer grid (even x, y, w, h)
    preview                         -> preview, later (one at a time; `error` while one is pending)
    pick_stars                      stack 3 frames begun after the press, pick, grade
    star        id, state: keep | drop | pin        pin is exclusive
    min_snr     value               clamped to 3..60
    moved       cmd_id              a page-driven motor move finished; frames begun 0.5 s later carry its id
cmd_id is an int in 0..2**32-1.

Messages to a page:
    state     session ('active' | 'idle'), controls, pages, idle_warning (s or null), rig {page, picture, reason};
              active: motor; idle: driver_has_video, idle_reason (the last session's end reason or null)
    pong      t (server epoch s), id (echoed)
    mode      picture, reason, exposure, gain, roi ([x, y, w, h] or null)
    reading   frame, cmd_id, capture_time, exposure, gain, width, height (gray px; the JPEG is half size),
              measure ('stars' | 'contrast' | 'none'), stars_visible (count before a pick, else null),
              size (median of the used stars, px, or null), stars_used, contrast (or null),
              rings [{id, x, y, size, snr, state}] (size and snr null when the star was not found; state as in
              `stars`, but `saturated` only when this frame clipped the star), message (or null)
    measure   measure (sent when the measure switches)
    stars     stars [{id, x, y, snr1, state: kept | weak | saturated | dropped | pinned}], min_snr
              (precedence: pinned, dropped, saturated, then kept or weak by snr1)
    progress  what ('pick' | 'exposure_search'), message, done; pick also: frame, of, candidates
    preview   jpeg (base64), width, height (the whole sensor, px), jpeg_width, jpeg_height
    error     message, command
    ended     reason ('done' | 'no_page' | 'idle' | 'failed' | 'stopped')
"""

from __future__ import annotations

import base64
import dataclasses
import logging
import math
import queue
import statistics
import threading
import time
from dataclasses import dataclass
from typing import Callable
from typing import Optional

import cv2
import numpy as np

from indi_allsky.focus.exposure_search import FOUND_STARS
from indi_allsky.focus.exposure_search import MSG_NO_STARS
from indi_allsky.focus.exposure_search import ExposureSearch
from indi_allsky.focus.exposure_search import brightness
from indi_allsky.focus.frame import Stretch
from indi_allsky.focus.frame import draw_clock
from indi_allsky.focus.frame import encode_jpeg
from indi_allsky.focus.frame import pack_frame
from indi_allsky.focus.frame import preview
from indi_allsky.focus.measure import DEFAULT_MIN_SNR
from indi_allsky.focus.measure import MEASURE_BOX
from indi_allsky.focus.measure import Candidate
from indi_allsky.focus.measure import Tracker
from indi_allsky.focus.measure import contrast_score
from indi_allsky.focus.measure import detect
from indi_allsky.focus.measure import find_hot_pixels
from indi_allsky.focus.measure import fix_hot_pixels
from indi_allsky.focus.measure import grade
from indi_allsky.focus.measure import measure_star
from indi_allsky.focus.measure import noise_sigma
from indi_allsky.focus.measure import pick
from indi_allsky.focus.measure import to_gray
from indi_allsky.focus.rigs import RigDecision
from indi_allsky.focus.session import CommandLedger
from indi_allsky.focus.session import SessionRules
from indi_allsky.focus.sources import normalise_bayer


logger = logging.getLogger('indi_allsky')


WORKER_POLL_S = 0.1
JOIN_TIMEOUT_S = 5.0
MAX_FRAME_FAILURES = 5
MEASURE_SWITCH_FRAMES = 5     # a new measure must win this many frames in a row (§12 item 7)
PICK_FRAMES = 3
PICK_MAX_STARS = 20
GRADE_SECONDS = 4.0           # grade over about 4 s of frames, at least 3 and at most 10
GRADE_MIN_FRAMES = 3
GRADE_MAX_FRAMES = 10
MOVE_SETTLE_S = 0.5
MIN_SNR_RANGE = (3.0, 60.0)
MAX_EXPOSURE_S = 3600.0
PREVIEW_MAX_PX = 1024
SATURATION_FRACTION = 0.98    # a raw peak at or above this share of full scale is clipped
CMD_ID_MAX = 2 ** 32 - 1

MSG_CONTROLS = 'Only a page with controls can do that.'
MSG_NO_SESSION = 'No focus session is running.'
MSG_UNKNOWN = 'Unknown command.'
MSG_INVALID = 'Not a valid {0}.'
MSG_PICK_BUSY = 'Wait for the star pick to finish.'
MSG_PICK_WAIT = 'Picking stars: waiting for a new picture'
MSG_PICKING = 'Picking stars: frame {0:d} of {1:d} · {2:d} found so far'
MSG_PICK_NONE = 'No stars found to pick. Try a longer exposure.'
MSG_PICKED = 'Picked {0:d} stars.'
MSG_PICK_CANCELLED = 'Star pick cancelled: the region moved.'
MSG_VISIBLE = 'About {0:d} stars visible · press Pick stars to measure'
MSG_OVER = 'The picture is over-exposed. Try a shorter exposure.'
MSG_LOST = 'None of the picked stars is in this picture.'
MSG_PREVIEW_FAILED = 'Could not take a whole-sensor picture.'
MSG_PREVIEW_BUSY = 'A whole-sensor picture is already on its way.'
MSG_SATURATED = 'The picked stars are saturated. Try a shorter exposure.'

OPEN_COMMANDS = ('ping', 'hello')


@dataclass(frozen=True)
class CameraInfo:
    interface: str
    device_name: str
    driver_has_video: bool
    rig: RigDecision


class _Invalid(Exception):
    """A command value out of range; the argument names it for MSG_INVALID."""


def _rig_dict(rig) -> dict:
    return dataclasses.asdict(rig) if rig is not None else None


def _num(v, name, low=None, high=None, low_open=False) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise _Invalid(name)
    v = float(v)
    if low is not None and (v < low or (low_open and v == low)):
        raise _Invalid(name)
    if high is not None and v > high:
        raise _Invalid(name)
    return v


def _int(v, name, low, high) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or not low <= v <= high:
        raise _Invalid(name)
    return v


def _cmd_id(msg) -> Optional[int]:
    if msg.get('cmd_id') is None:
        return None
    return _int(msg['cmd_id'], 'command id', 0, CMD_ID_MAX)


def _roi(v):
    if v is None:
        return None
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        raise _Invalid('region')
    x, y, w, h = (_int(n, 'region', 0, 1 << 20) for n in v)
    x, y, w, h = x - x % 2, y - y % 2, w - w % 2, h - h % 2
    if w < 2 or h < 2:
        raise _Invalid('region')
    return (x, y, w, h)


def _f(v) -> Optional[float]:
    """A JSON-safe float: NaN and infinities become None."""
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def _peak(raw, x, y) -> float:
    """The brightest raw value in the measure box around (x, y)."""
    half = MEASURE_BOX // 2
    xi, yi = int(round(x)), int(round(y))
    box = raw[max(0, yi - half):yi + half + 1, max(0, xi - half):xi + half + 1]
    return float(box.max()) if box.size else 0.0


class _Pick(object):
    """The worker's star pick in progress: stack frames begun after `press`, then grade the picked stars."""

    def __init__(self, gen, press):
        self.gen = gen
        self.press = press
        self.frames = []          # (gray, hot-pixel-fixed raw, saturation level) of each stacked frame
        self.stack = None
        self.cands = None
        self.snrs = None
        self.peaks = None         # per candidate: its raw peak over the saturation level, per grading frame
        self.need = GRADE_MIN_FRAMES


class FocusController(object):

    def __init__(self, source, link, clock, rig: RigDecision, motor_configured: bool, *,
                 epoch: Callable[[], float] = time.time, exposure: float = 1.0, gain: float = 0.0, roi=None,
                 search: Optional[ExposureSearch] = None, saved_camera: Optional[dict] = None):
        self.source = source
        self.link = link
        self.rig = rig
        self.motor_configured = bool(motor_configured)
        self.exposure = float(exposure)
        self.gain = float(gain)
        self.roi = tuple(roi) if roi is not None else None
        self.saved_camera = dict(saved_camera) if saved_camera is not None else {
            'exposure': self.exposure, 'gain': self.gain, 'roi': self.roi}
        self.end_reason = None

        self._epoch = epoch
        self._lock = threading.Lock()
        self.rules = SessionRules(clock)
        self.ledger = CommandLedger(clock=epoch)
        self._stars = []          # [{'id', 'x', 'y', 'snr1', 'saturated', 'choice': keep | drop | pin}]
        self._stars_gen = 0
        self._min_snr = DEFAULT_MIN_SNR
        self._region_gen = 0
        self._region_since = None  # epoch of the last region change
        self._pick_gen = 0
        self._pick_press = None   # epoch of the Pick stars press while a pick runs
        self._search = search
        self._search_on = search is not None

        self._active = False
        self._ended = False
        self._pages = set()
        self._warned = None
        self._requests = queue.Queue()
        self._stop = threading.Event()
        self._thread = None
        self._preview_thread = None

        # worker-only state
        self._frame_no = 0
        self._seen_stars_gen = -1
        self._seen_pick_gen = 0
        self._tracker = None
        self._pick = None
        self._frame_key = None
        self._hot = None
        self._stretch = None
        self._stretch_phase = 'linear'
        self._measure = None
        self._pending_measure = (None, 0)
        self._search_done = False
        self._last_format = (None, 65535.0)

    @property
    def active(self) -> bool:
        return self._active

    # -- life ------------------------------------------------------------------------------------------------------

    def begin(self, by_client: int) -> None:
        with self._lock:
            self.rules.start()
            self._pages = set(self.link.clients())
            for cid in self._pages:
                self.rules.page_connected(cid)
            self._active = True
        logger.info('Focus session started by page %d', by_client)
        if not self._source_call('start', self.source.start, self.roi, self.exposure, self.gain):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name='focus-worker', daemon=True)
        self._thread.start()
        self._send_state()
        self.link.broadcast_json(self._mode_msg())

    def end(self, reason: str) -> None:
        with self._lock:
            if self._ended:
                return
            self._ended = True
            self._active = False
            if self.rules.active:
                self.rules.end(reason)
            self.end_reason = reason
        logger.info('Focus session ended: %s', reason)
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(JOIN_TIMEOUT_S)
        try:
            self.source.stop()  # also releases a waiting whole_frame()
        except Exception:
            logger.exception('Focus: the picture source failed to stop')
        pending = self._preview_thread
        if pending is not None and pending is not threading.current_thread():
            pending.join(JOIN_TIMEOUT_S)
            if pending.is_alive():
                logger.error('Focus: the whole-sensor picture did not stop')
        self.link.broadcast_json({'kind': 'ended', 'reason': reason})

    def service(self) -> None:
        """Page arrivals, the session rules, and the worker's requests. Call it often from the owning thread."""
        if not self._active:
            return
        now_pages = set(self.link.clients())
        with self._lock:
            joined = now_pages - self._pages
            for cid in joined:
                self.rules.page_connected(cid)
            for cid in self._pages - now_pages:
                self.rules.page_left(cid)
            self._pages = now_pages
            reason = self.rules.tick()
            warning = self.rules.idle_warning()
        for cid in sorted(joined):
            self._send_state(cid)
            self.link.send_json(cid, self._mode_msg())
            self.link.send_json(cid, self._stars_msg())
            if self._measure is not None:
                self.link.send_json(cid, {'kind': 'measure', 'measure': self._measure})

        while True:
            try:
                request = self._requests.get_nowait()
            except queue.Empty:
                break
            if not self._active:
                return
            self._apply(request)
        if not self._active:
            return

        if reason is not None:
            self.end(reason)
            return
        warned = None if warning is None else int(math.ceil(warning))
        if warned != self._warned:
            self._warned = warned
            self._send_state()

    # -- commands --------------------------------------------------------------------------------------------------

    def handle(self, client_id: int, msg: dict) -> None:
        kind = msg.get('kind') if isinstance(msg, dict) else None
        if kind not in OPEN_COMMANDS and not self._controls(client_id):
            self._error(client_id, kind, MSG_CONTROLS)
            return
        if kind not in OPEN_COMMANDS:
            with self._lock:
                self.rules.touched()
        handler = getattr(self, '_cmd_' + kind, None) if isinstance(kind, str) else None
        if handler is None:
            self._error(client_id, kind, MSG_UNKNOWN)
            return
        try:
            handler(client_id, msg)
        except _Invalid as e:
            self._error(client_id, kind, MSG_INVALID.format(e.args[0]))

    def _cmd_ping(self, cid, msg):
        self.link.send_json(cid, {'kind': 'pong', 't': time.time(), 'id': msg.get('id')})

    def _cmd_hello(self, cid, msg):
        self._send_state(cid)

    def _cmd_start(self, cid, msg):
        self._send_state(cid)

    def _cmd_keep_going(self, cid, msg):
        self._warned = None
        self._send_state()

    def _cmd_done(self, cid, msg):
        self.end('done')

    def _cmd_set_camera(self, cid, msg):
        exposure = _num(msg.get('exposure'), 'exposure', 0.0, MAX_EXPOSURE_S, low_open=True)
        gain = _num(msg.get('gain'), 'gain', 0.0)
        cmd_id = _cmd_id(msg)
        with self._lock:
            if self._pick_press is not None:
                busy = True
            else:
                busy = False
                self._search_on = False
                if cmd_id is not None:
                    self.ledger.begin(cmd_id)
        if busy:
            self._error(cid, 'set_camera', MSG_PICK_BUSY)
            return
        if not self._source_call('set the exposure', self.source.set_camera, exposure, gain):
            return
        self.exposure, self.gain = exposure, gain
        if cmd_id is not None:
            with self._lock:
                self.ledger.done(cmd_id, 0.0)
        self.link.broadcast_json(self._mode_msg())

    def _cmd_set_region(self, cid, msg):
        roi = _roi(msg.get('roi'))
        cmd_id = _cmd_id(msg)
        with self._lock:
            if cmd_id is not None:
                self.ledger.begin(cmd_id)
            cancelled = self._pick_press is not None
            self._clear_stars_locked()
        if not self._source_call('stop', self.source.stop):
            return
        _drain(self.source.frames)
        with self._lock:
            # a frame begun before now is from the old region; the worker skips it (a stills load in flight can
            # still queue one after the drain)
            self._region_gen += 1
            self._region_since = self._epoch()
        self.roi = roi
        if not self._source_call('start', self.source.start, roi, self.exposure, self.gain):
            return
        if cmd_id is not None:
            with self._lock:
                self.ledger.done(cmd_id, 0.0)
        if cancelled:
            self.link.broadcast_json({'kind': 'progress', 'what': 'pick', 'message': MSG_PICK_CANCELLED,
                                      'done': True, 'frame': 0, 'of': PICK_FRAMES, 'candidates': 0})
        self.link.broadcast_json(self._stars_msg())
        self.link.broadcast_json(self._mode_msg())

    def _cmd_preview(self, cid, msg):
        if self._preview_thread is not None and self._preview_thread.is_alive():
            self._error(cid, 'preview', MSG_PREVIEW_BUSY)
            return
        bayer, _ = self._last_format  # the region crop keeps the Bayer phase, so the stream's pattern holds
        self._preview_thread = threading.Thread(target=self._take_preview, args=(cid, bayer), name='focus-preview',
                                                daemon=True)
        self._preview_thread.start()

    def _take_preview(self, cid, bayer):
        """On the focus-preview thread: wait for the whole sensor, make the reply, post it for service()."""
        try:
            reply = self._preview_msg(self.source.whole_frame(), bayer)
        except Exception:
            if self._stop.is_set():
                return  # the session ended under it
            logger.exception('Focus: whole-sensor picture failed')
            reply = None
        self._post(('preview', cid, reply))

    @staticmethod
    def _preview_msg(raw, bayer):
        gray = to_gray(raw, bayer)
        img = preview(gray, Stretch.from_frame(gray))
        longest = max(img.shape[:2])
        if longest > PREVIEW_MAX_PX:
            f = PREVIEW_MAX_PX / float(longest)
            img = cv2.resize(img, (max(1, int(img.shape[1] * f)), max(1, int(img.shape[0] * f))),
                             interpolation=cv2.INTER_AREA)
        # width and height are the sensor's: the page maps the region it draws on the JPEG back to sensor px
        return {
            'kind': 'preview',
            'jpeg': base64.b64encode(encode_jpeg(img)).decode('ascii'),
            'width': int(raw.shape[1]),
            'height': int(raw.shape[0]),
            'jpeg_width': int(img.shape[1]),
            'jpeg_height': int(img.shape[0]),
        }

    def _cmd_pick_stars(self, cid, msg):
        with self._lock:
            self._pick_gen += 1
            self._pick_press = self._epoch()
            self._search_on = False
        self.link.broadcast_json({'kind': 'progress', 'what': 'pick', 'message': MSG_PICK_WAIT,
                                  'done': False, 'frame': 0, 'of': PICK_FRAMES, 'candidates': 0})

    def _cmd_star(self, cid, msg):
        state = msg.get('state')
        if state not in ('keep', 'drop', 'pin'):
            raise _Invalid('star state')
        with self._lock:
            ids = [s['id'] for s in self._stars]
            star_id = msg.get('id')
            if isinstance(star_id, bool) or star_id not in ids:
                raise _Invalid('star')
            for s in self._stars:
                if s['id'] == star_id:
                    s['choice'] = state
                elif state == 'pin' and s['choice'] == 'pin':
                    s['choice'] = 'keep'
        self.link.broadcast_json(self._stars_msg())

    def _cmd_min_snr(self, cid, msg):
        value = _num(msg.get('value'), 'brightness')
        with self._lock:
            self._min_snr = min(max(value, MIN_SNR_RANGE[0]), MIN_SNR_RANGE[1])
        self.link.broadcast_json(self._stars_msg())

    def _cmd_moved(self, cid, msg):
        cmd_id = _cmd_id(msg)
        if cmd_id is None:
            raise _Invalid('command id')
        with self._lock:
            self.ledger.begin(cmd_id)
            self.ledger.done(cmd_id, MOVE_SETTLE_S)

    # -- messages --------------------------------------------------------------------------------------------------

    def _controls(self, cid) -> bool:
        hello = self.link.client_hello(cid)
        return bool(hello) and hello.get('controls') is True

    def _error(self, cid, command, message):
        self.link.send_json(cid, {'kind': 'error', 'message': message,
                                  'command': command if isinstance(command, str) else None})

    def _send_state(self, cid=None):
        with self._lock:
            pages = len(self._pages)
            warning = self.rules.idle_warning()
        targets = [cid] if cid is not None else self.link.clients()
        for target in targets:
            self.link.send_json(target, {
                'kind': 'state',
                'session': 'active' if self._active else 'idle',
                'controls': self._controls(target),
                'pages': pages,
                'idle_warning': None if warning is None else int(math.ceil(warning)),
                'rig': _rig_dict(self.rig),
                'motor': self.motor_configured,
            })

    def _mode_msg(self):
        return {
            'kind': 'mode',
            'picture': self.rig.picture if self.rig is not None else None,
            'reason': self.rig.reason if self.rig is not None else None,
            'exposure': self.exposure,
            'gain': self.gain,
            'roi': list(self.roi) if self.roi is not None else None,
        }

    def _star_state(self, s, saturation=True) -> str:
        if s['choice'] == 'pin':
            return 'pinned'
        if s['choice'] == 'drop':
            return 'dropped'
        if saturation and s['saturated']:
            return 'saturated'
        return grade(s['snr1'], self._min_snr)

    def _stars_msg(self):
        with self._lock:
            stars = [{'id': s['id'], 'x': s['x'], 'y': s['y'], 'snr1': s['snr1'], 'state': self._star_state(s)}
                     for s in self._stars]
            return {'kind': 'stars', 'stars': stars, 'min_snr': self._min_snr}

    def _clear_stars_locked(self):
        self._stars = []
        self._stars_gen += 1
        self._pick_gen += 1
        self._pick_press = None

    # -- worker requests (applied on the owning thread) ------------------------------------------------------------

    def _source_call(self, what, fn, *args) -> bool:
        """Call the picture source; on an error, log it and end the session as failed."""
        try:
            fn(*args)
            return True
        except Exception:
            logger.exception('Focus: the picture source failed to %s', what)
            self.end('failed')
            return False

    def _post(self, request):
        self._requests.put(request)

    def _apply(self, request):
        what = request[0]
        if what == 'end':
            self.end(request[1])
        elif what == 'stars':
            _, gen, found = request
            with self._lock:
                if gen != self._pick_gen:
                    return
                self._stars = [{'id': i, 'x': x, 'y': y, 'snr1': snr1, 'saturated': sat, 'choice': 'keep'}
                               for i, (x, y, snr1, sat) in enumerate(found)]
                self._stars_gen += 1
                self._pick_press = None
            self.link.broadcast_json(self._stars_msg())
            self.link.broadcast_json({'kind': 'progress', 'what': 'pick', 'message': MSG_PICKED.format(len(found)),
                                      'done': True, 'frame': PICK_FRAMES, 'of': PICK_FRAMES,
                                      'candidates': len(found)})
        elif what == 'preview':
            _, cid, reply = request
            if reply is None:
                self._error(cid, 'preview', MSG_PREVIEW_FAILED)
            else:
                self.link.send_json(cid, reply)
        elif what == 'pick_failed':
            _, gen, message = request
            with self._lock:
                if gen != self._pick_gen:
                    return
                self._pick_press = None
            self.link.broadcast_json({'kind': 'progress', 'what': 'pick', 'message': message, 'done': True,
                                      'frame': PICK_FRAMES, 'of': PICK_FRAMES, 'candidates': 0})
        elif what == 'search':
            step = request[1]
            with self._lock:
                if not self._search_on:
                    return
                if step.done:
                    self._search_on = False
            if step.exposure is not None and not math.isclose(step.exposure, self.exposure):
                if not self._source_call('set the exposure', self.source.set_camera, step.exposure, self.gain):
                    return
                self.exposure = float(step.exposure)
                self.link.broadcast_json(self._mode_msg())
            self.link.broadcast_json({'kind': 'progress', 'what': 'exposure_search', 'message': step.message,
                                      'done': step.done})

    # -- the worker ------------------------------------------------------------------------------------------------

    def _run(self):
        failures = 0
        frames = self.source.frames
        while not self._stop.is_set():
            try:
                frame = frames.get(timeout=WORKER_POLL_S)
            except queue.Empty:
                continue
            count = 1
            while True:  # behind: measure only the newest waiting frame
                try:
                    frame = frames.get_nowait()
                except queue.Empty:
                    break
                count += 1
            self._frame_no += count
            if self._stop.is_set():
                return
            try:
                self._process(frame, self._frame_no)
                failures = 0
            except Exception:
                logger.exception('Focus: frame %d failed', self._frame_no)
                failures += 1
                if failures >= MAX_FRAME_FAILURES:
                    self._post(('end', 'failed'))
                    return

    def _snapshot(self, exposure_start):
        with self._lock:
            return {
                # the state without the pick's saturation mark: each frame checks clipping for itself
                'stars': [(s['id'], s['x'], s['y'], self._star_state(s, saturation=False)) for s in self._stars],
                'stars_gen': self._stars_gen,
                'region_gen': self._region_gen,
                'region_since': self._region_since,
                'pick_gen': self._pick_gen,
                'pick_press': self._pick_press,
                'search_on': self._search_on,
                'cmd_id': self.ledger.effective_before(exposure_start),
            }

    def _process(self, frame, n):
        exposure_start = frame.t_end - frame.exposure
        capture_time = frame.t_end - frame.exposure / 2.0
        snap = self._snapshot(exposure_start)
        if snap['region_since'] is not None and exposure_start < snap['region_since']:
            return  # taken before the region moved
        bayer = normalise_bayer(frame.bayer)
        full_scale = float(frame.full_scale)
        raw = frame.raw
        self._last_format = (bayer, full_scale)

        # a new region, shape or camera setting: find hot pixels again and show the next frame linear
        key = (raw.shape, snap['region_gen'], frame.exposure, frame.gain)
        if key != self._frame_key:
            self._frame_key = key
            self._hot = None
            self._stretch = None
            self._stretch_phase = 'linear'

        step = 2 if bayer else 1
        if self._hot is None:
            self._hot = find_hot_pixels(raw, step)
        fixed = fix_hot_pixels(raw, self._hot, step)
        gray = to_gray(fixed, bayer)
        clip = SATURATION_FRACTION * full_scale
        sigma = noise_sigma(gray)
        level = brightness(gray, full_scale)
        height, width = gray.shape[:2]

        # -- measure -------------------------------------------------------------------------------------------------
        if snap['stars_gen'] != self._seen_stars_gen:
            self._seen_stars_gen = snap['stars_gen']
            self._tracker = Tracker([Candidate(x, y, 0.0, 0) for _, x, y, _ in snap['stars']]) \
                if snap['stars'] else None

        picked = bool(snap['stars'])
        count = None
        rings = []
        size = None
        used = 0
        if picked:
            got = self._tracker.update(gray, sigma)
            positions = self._tracker.positions
            pinned = [sid for sid, _, _, state in snap['stars'] if state == 'pinned']
            use = set(pinned) if pinned else {sid for sid, _, _, state in snap['stars'] if state == 'kept'}
            sizes = []
            clipped = False
            for i, (sid, _, _, state) in enumerate(snap['stars']):
                m = got.get(i)
                x, y = (m.x, m.y) if m is not None else positions[i]
                saturated = state != 'dropped' and _peak(fixed, x, y) >= clip
                rings.append({'id': sid, 'x': _f(x), 'y': _f(y), 'size': _f(m.size) if m else None,
                              'snr': _f(m.snr) if m else None, 'state': 'saturated' if saturated else state})
                if sid in use and saturated:
                    clipped = True
                elif m is not None and sid in use and math.isfinite(m.size):
                    sizes.append(m.size)
            used = len(sizes)
            size = _f(statistics.median(sizes)) if sizes else None
            candidate = 'stars'
        else:
            count = len(detect(gray))
            candidate = 'stars' if count >= FOUND_STARS else ('contrast' if level == 'bright' else 'none')
        measure = self._switch_measure(candidate, immediate=picked)

        contrast = _f(contrast_score(gray)) if measure == 'contrast' else None
        if measure == 'stars':
            if not picked:
                message = MSG_VISIBLE.format(count)
            elif used == 0:
                message = MSG_SATURATED if clipped else MSG_LOST
            else:
                message = None
        elif measure == 'none':
            message = MSG_OVER if level == 'over' else MSG_NO_STARS
        else:
            message = None

        # -- picking and the exposure search ---------------------------------------------------------------------------
        self._step_pick(snap, frame, gray, fixed, clip, sigma, exposure_start)
        if (self._search is not None and snap['search_on'] and not self._search_done and not picked
                and self._pick is None and math.isclose(frame.exposure, self._search.exposure)):
            s = self._search.step(count, level)
            self._search_done = s.done
            self._post(('search', s))

        # -- preview -------------------------------------------------------------------------------------------------
        if self._stretch_phase == 'linear':
            stretch = Stretch(0.0, full_scale, 1.0)
            self._stretch_phase = 'pending'
        elif self._stretch_phase == 'pending':
            self._stretch = Stretch.from_frame(gray)
            self._stretch_phase = 'fixed'
            stretch = self._stretch
        else:
            stretch = self._stretch
        img = preview(gray, stretch)  # a new uint8 image: the clock never touches what was measured
        draw_clock(img, capture_time, n)
        self.link.broadcast_frame(pack_frame(encode_jpeg(img), n, capture_time, frame.exposure, frame.gain,
                                             snap['cmd_id']))
        self.link.broadcast_json({
            'kind': 'reading',
            'frame': n,
            'cmd_id': snap['cmd_id'],
            'capture_time': capture_time,
            'exposure': float(frame.exposure),
            'gain': float(frame.gain),
            'width': int(width),
            'height': int(height),
            'measure': measure,
            'stars_visible': count,
            'size': size,
            'stars_used': used,
            'contrast': contrast,
            'rings': rings,
            'message': message,
        })

    def _switch_measure(self, candidate, immediate):
        if self._measure is None or immediate:
            changed = candidate != self._measure
            self._measure = candidate
            self._pending_measure = (None, 0)
        elif candidate == self._measure:
            self._pending_measure = (None, 0)
            changed = False
        else:
            want, run = self._pending_measure
            run = run + 1 if want == candidate else 1
            self._pending_measure = (candidate, run)
            changed = run >= MEASURE_SWITCH_FRAMES
            if changed:
                self._measure = candidate
                self._pending_measure = (None, 0)
        if changed:
            self.link.broadcast_json({'kind': 'measure', 'measure': self._measure})
        return self._measure

    def _step_pick(self, snap, frame, gray, fixed, clip, sigma, exposure_start):
        if snap['pick_gen'] != self._seen_pick_gen:
            self._seen_pick_gen = snap['pick_gen']
            self._pick = _Pick(snap['pick_gen'], snap['pick_press']) if snap['pick_press'] is not None else None
        p = self._pick
        if p is None:
            return

        if p.cands is None:
            if exposure_start < p.press:
                return
            g = gray.astype(np.float64)
            p.stack = g if p.stack is None else p.stack + g
            p.frames.append((gray, fixed, clip))
            k = len(p.frames)
            found = pick(p.stack / k)
            self.link.broadcast_json({'kind': 'progress', 'what': 'pick',
                                      'message': MSG_PICKING.format(k, PICK_FRAMES, len(found)),
                                      'done': False, 'frame': k, 'of': PICK_FRAMES, 'candidates': len(found)})
            if k < PICK_FRAMES:
                return
            p.cands = found[:PICK_MAX_STARS]
            p.stack = None
            if not p.cands:
                self._post(('pick_failed', p.gen, MSG_PICK_NONE))
                self._pick = None
                return
            p.need = min(max(int(round(GRADE_SECONDS / max(frame.exposure, 1e-6))), GRADE_MIN_FRAMES),
                         GRADE_MAX_FRAMES)
            p.snrs = [[] for _ in p.cands]
            p.peaks = [[] for _ in p.cands]
            stacked, p.frames = p.frames, []
            for g, f, c in stacked:
                self._grade_frame(p, g, f, c, noise_sigma(g))
        else:
            self._grade_frame(p, gray, fixed, clip, sigma)

        if len(p.snrs[0]) >= p.need:
            found = [(float(c.x), float(c.y), float(statistics.median(s)), bool(statistics.median(k) >= 1.0))
                     for c, s, k in zip(p.cands, p.snrs, p.peaks)]
            self._post(('stars', p.gen, found))
            self._pick = None

    @staticmethod
    def _grade_frame(p, gray, fixed, clip, sigma):
        for c, snrs, peaks in zip(p.cands, p.snrs, p.peaks):
            m = measure_star(gray, c.x, c.y, sigma)
            snrs.append(max(float(m.snr), 0.0) if m is not None and math.isfinite(m.snr) else 0.0)
            peaks.append(_peak(fixed, c.x, c.y) / clip)  # >= 1 when clipped


def _drain(q):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return


class FocusService(object):
    """Commands while capture runs; owns at most one `FocusController` at a time."""

    def __init__(self, link, camera_info: CameraInfo, make_controller: Callable[[int], FocusController], *,
                 on_end: Optional[Callable[[FocusController], None]] = None):
        self.link = link
        self.camera_info = camera_info
        self._make = make_controller
        self._on_end = on_end
        self.controller = None
        self.idle_reason = None
        self._known = set()

    @property
    def active(self) -> bool:
        return self.controller is not None and self.controller.active

    def poll(self) -> bool:
        """Drain the commands. True when this call started a session."""
        started = False
        self._reap()
        for cid, msg in self.link.poll_commands():
            if self.controller is not None:
                self.controller.handle(cid, msg)
                self._reap()
                continue
            kind = msg.get('kind') if isinstance(msg, dict) else None
            if kind == 'ping':
                self.link.send_json(cid, {'kind': 'pong', 't': time.time(), 'id': msg.get('id')})
            elif kind == 'hello':
                self._send_idle_state(cid)
            elif kind == 'start':
                hello = self.link.client_hello(cid)
                if not hello or hello.get('controls') is not True:
                    self._error(cid, kind, MSG_CONTROLS)
                    continue
                controller = self._make(cid)
                self.controller = controller
                controller.begin(cid)
                started = controller.active  # False when the source failed to start
            else:
                self._error(cid, kind, MSG_NO_SESSION)

        if self.controller is not None:
            self.controller.service()
            self._reap()
        else:
            now = set(self.link.clients())
            for cid in sorted(now - self._known):
                self._send_idle_state(cid)
            self._known = now
        return started

    def _reap(self):
        controller = self.controller
        if controller is None or controller.active:
            return
        self.controller = None
        self.idle_reason = controller.end_reason
        if self._on_end is not None:
            try:
                self._on_end(controller)
            except Exception:
                logger.exception('Focus: restoring the camera after the session failed')
        self._known = set(self.link.clients())
        for cid in sorted(self._known):
            self._send_idle_state(cid)

    def _send_idle_state(self, cid):
        hello = self.link.client_hello(cid)
        self.link.send_json(cid, {
            'kind': 'state',
            'session': 'idle',
            'controls': bool(hello) and hello.get('controls') is True,
            'pages': len(self.link.clients()),
            'idle_warning': None,
            'rig': _rig_dict(self.camera_info.rig),
            'driver_has_video': bool(self.camera_info.driver_has_video),
            'idle_reason': self.idle_reason,
        })

    def _error(self, cid, command, message):
        self.link.send_json(cid, {'kind': 'error', 'message': message,
                                  'command': command if isinstance(command, str) else None})
