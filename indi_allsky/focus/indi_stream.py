"""Live video from an INDI camera for a focus session, and the camera state saved around it.

`IndiStreamSource` drives the driver's stream (CCD_VIDEO_STREAM) and turns each blob into a `RawFrame`. It works on
any client with `set_number`, `set_switch`, `get_control`, `ccd_device` and a `blob_sink` attribute: the INDI client
calls `blob_sink(data, fmt, t)` for every blob while it is set, instead of making a FITS for the image queue. A client
with `start_video(exposure, roi)` and `stop_video()` (the test cameras) streams by itself, and no property is set.

`save_state` / `restore_state` keep exactly the properties the stream changes (focus_spike.py:27-28). The state is
written next to the link's socket, so a capture process that dies in a session puts the camera back when it starts
again (`focus_startup_restore`).

This module never imports PyIndi: switch states are compared by value (ISS_ON is 1).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from typing import Optional

import numpy as np

from indi_allsky.focus.frame import decode_stream
from indi_allsky.focus.sources import RawFrame
from indi_allsky.focus.sources import normalise_bayer
from indi_allsky.focus.sources import snap_roi


logger = logging.getLogger('indi_allsky')


NUMBERS = ('CCD_CONTROLS', 'CCD_FRAME', 'CCD_BINNING', 'CCD_STREAM_FRAME', 'STREAMING_EXPOSURE', 'LIMITS')
SWITCHES = ('STREAM_FULL_DEPTH', 'CCD_STREAM_ENCODER', 'CCD_VIDEO_FORMAT', 'CCD_CAPTURE_FORMAT', 'CCD_VIDEO_STREAM')

# restore order (focus_spike.py restore): the stream off first, then formats, then numbers; the stream stays off
RESTORE_SWITCHES = ('STREAM_FULL_DEPTH', 'CCD_STREAM_ENCODER', 'CCD_VIDEO_FORMAT', 'CCD_CAPTURE_FORMAT')
RESTORE_NUMBERS = ('CCD_BINNING', 'CCD_FRAME', 'CCD_STREAM_FRAME', 'STREAMING_EXPOSURE', 'LIMITS', 'CCD_CONTROLS')

CONTROL_TIMEOUT_S = 1.0       # finding a property; get_control waits forever on 0
SET_TIMEOUT_S = 5.0           # a property change to settle
WHOLE_FRAME_TIMEOUT_S = 30.0
MAX_QUEUED = 4                # frames waiting for the controller; older ones are dropped
ISS_ON = 1


def _on(state) -> bool:
    return state == ISS_ON or str(state).upper() in ('ON', 'ISS_ON')


def _element_max(e) -> Optional[float]:
    getter = getattr(e, 'getMax', None)
    value = getter() if callable(getter) else getattr(e, 'max', None)
    return None if value is None else float(value)


def _device(client):
    return getattr(client, 'ccd_device', None)


def _numbers(client, name) -> dict:
    ctl = client.get_control(_device(client), name, 'number', timeout=CONTROL_TIMEOUT_S)
    return {e.getName(): float(e.getValue()) for e in ctl}


def _switch_on(client, name) -> list:
    ctl = client.get_control(_device(client), name, 'switch', timeout=CONTROL_TIMEOUT_S)
    return [e.getName() for e in ctl if _on(e.getState())]


def has_video(client) -> bool:
    """True when the camera's driver has a CCD_VIDEO_STREAM property."""
    own = getattr(client, 'has_video', None)
    if callable(own):
        try:
            return bool(own())
        except Exception:
            logger.exception('Focus: could not tell whether the camera has video')
            return False
    try:
        return bool(client.get_control(_device(client), 'CCD_VIDEO_STREAM', 'switch', timeout=CONTROL_TIMEOUT_S))
    except Exception:
        return False


# -- camera state ---------------------------------------------------------------------------------------------------

def save_state(client) -> dict:
    """{'numbers': {property: {element: value}}, 'switches': {property: [elements on]}} for the properties the
    stream changes. A property the driver does not have is left out."""
    state = {'numbers': {}, 'switches': {}}
    for name in NUMBERS:
        try:
            state['numbers'][name] = _numbers(client, name)
        except Exception:
            logger.info('Focus: the camera has no %s to save', name)
    for name in SWITCHES:
        try:
            state['switches'][name] = _switch_on(client, name)
        except Exception:
            logger.info('Focus: the camera has no %s to save', name)
    return state


def restore_state(client, state: dict) -> int:
    """Put the saved values back, the stream off whatever was saved. Returns the number of properties that failed;
    each failure is logged and the rest are still restored."""
    device = _device(client)
    failures = 0
    numbers = state.get('numbers') or {}
    switches = state.get('switches') or {}

    def attempt(what, fn, *args, **kwargs):
        nonlocal failures
        try:
            fn(*args, **kwargs)
        except Exception as e:
            failures += 1
            logger.error('Focus: could not restore %s: %s', what, e)

    if 'CCD_VIDEO_STREAM' in switches:
        attempt('CCD_VIDEO_STREAM', client.set_switch, device, 'CCD_VIDEO_STREAM', on_switches=['STREAM_OFF'])
    for name in RESTORE_SWITCHES:
        on = switches.get(name)
        if on:
            attempt(name, client.set_switch, device, name, on_switches=list(on))
    for name in RESTORE_NUMBERS:
        values = numbers.get(name)
        if values:
            attempt(name, client.set_number, device, name, dict(values), timeout=SET_TIMEOUT_S)
    return failures


def write_state(path: str, state: dict) -> None:
    """Atomically: a crash mid-write leaves the old file or none, never half a file."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def remove_state(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.error('Focus: could not delete the saved camera state: %s', e)


def focus_startup_restore(camera, path: str) -> bool:
    """At capture start: if a session left its saved state (the process died in it), restore the camera and delete
    the file. True when a state was restored. The file is always deleted, so a bad one cannot block every start."""
    if not os.path.exists(path):
        return False
    try:
        try:
            with open(path) as f:
                state = json.load(f)
            if not isinstance(state, dict):
                raise ValueError('not a camera state')
        except (OSError, ValueError) as e:
            logger.error('Focus: the saved camera state could not be read, deleting it: %s', e)
            return False
        logger.warning('Focus: a focus session did not end cleanly; restoring the camera')
        failures = restore_state(camera, state)
        if failures:
            logger.error('Focus: %d camera properties could not be restored', failures)
        return True
    except Exception:
        logger.exception('Focus: restoring the camera at startup failed')
        return False
    finally:
        remove_state(path)


# -- the stream -----------------------------------------------------------------------------------------------------

class IndiStreamSource(object):
    """`PictureSource` over the driver's video stream (spec amendment 5: new settings restart the stream).

    sensor: (width, height) in pixels; read from CCD_FRAME when not given. bits: the camera's bit depth, to tell
    12-bit data stored as is from data shifted up to 16 bits (full_scale is decided on the first frame of a stream
    and kept). bayer: the sensor's pattern, or None for mono.
    """

    def __init__(self, client, *, sensor=None, bits=None, bayer=None):
        self.client = client
        self.bayer = normalise_bayer(bayer)
        self.frames = queue.Queue()
        self._bits = int(bits) if bits else None
        self._sensor = tuple(int(v) for v in sensor) if sensor else None
        self._simulated = callable(getattr(client, 'start_video', None))
        self._ctl = threading.RLock()     # one camera reconfiguration at a time
        self._cond = threading.Condition()
        self._running = False
        self._region = None               # the region the controller asked for
        self._streaming = None            # the region the camera is streaming now
        self._exposure = 0.0
        self._gain = 0.0
        self._full_scale = None
        self._whole_waiting = False
        self._whole = None
        self._fps_max = None

    # -- PictureSource -----------------------------------------------------------------------------------------------

    def start(self, roi, exposure, gain):
        w, h = self._sensor_size()
        region = snap_roi(roi, (h, w)) if roi is not None else (0, 0, w - w % 2, h - h % 2)
        with self._cond:
            self._region = region
            self._exposure = float(exposure)
            self._gain = float(gain)
            self._running = True
        self.client.blob_sink = self._on_blob
        self._configure(region)

    def set_camera(self, exposure, gain):
        with self._cond:
            self._exposure = float(exposure)
            self._gain = float(gain)
            running = self._running
        if not running:
            return
        with self._ctl:
            if self._simulated:
                self._restart_simulated()
                return
            self._stream(False)
            self._set_rate()
            self._stream(True)

    def whole_frame(self, timeout=WHOLE_FRAME_TIMEOUT_S):
        """One full-sensor frame: stream the whole sensor until one arrives, then the region again."""
        w, h = self._sensor_size()
        full = (0, 0, w - w % 2, h - h % 2)
        with self._cond:
            if not self._running:
                raise RuntimeError('the picture source stopped')
            self._whole = None
            self._whole_waiting = True
        try:
            self._configure(full)
            deadline = time.monotonic() + timeout
            with self._cond:
                while self._whole is None:
                    if not self._running:
                        raise RuntimeError('the picture source stopped')
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('no whole-sensor picture')
                    self._cond.wait(remaining)
                return self._whole
        finally:
            with self._cond:
                self._whole_waiting = False
                self._whole = None
                running, region = self._running, self._region
            if running:
                self._configure(region)

    def stop(self):
        with self._cond:
            was = self._running
            self._running = False
            self._cond.notify_all()
        if not was:
            return
        with self._ctl:
            try:
                if self._simulated:
                    self.client.stop_video()
                else:
                    self._stream(False)
            finally:
                if getattr(self.client, 'blob_sink', None) == self._on_blob:
                    self.client.blob_sink = None  # a blob still on its way is dropped by the client

    # -- camera ------------------------------------------------------------------------------------------------------

    def _configure(self, region):
        """The start order (focus_spike.py prep_stream): stream off, region, format, rate, gain, exposure, on."""
        with self._ctl:
            with self._cond:
                if not self._running:
                    return
            if self._simulated:
                self._set_streaming(region)
                self._restart_simulated()
                return
            client, device = self.client, _device(self.client)
            self._stream(False)
            self._set_streaming(region)
            x, y, w, h = region
            client.set_number(device, 'CCD_FRAME', {'X': x, 'Y': y, 'WIDTH': w, 'HEIGHT': h}, timeout=SET_TIMEOUT_S)
            client.set_number(device, 'CCD_STREAM_FRAME', {'X': 0, 'Y': 0, 'WIDTH': w, 'HEIGHT': h},
                              timeout=SET_TIMEOUT_S)
            client.set_switch(device, 'CCD_STREAM_ENCODER', on_switches=['RAW'])
            client.set_switch(device, 'STREAM_FULL_DEPTH', on_switches=['FULL_DEPTH_16BIT'])
            self._set_rate()
            self._stream(True)

    def _set_rate(self):
        client, device = self.client, _device(self.client)
        with self._cond:
            exposure, gain = self._exposure, self._gain
        client.set_number(device, 'LIMITS', {'LIMITS_PREVIEW_FPS': self._fps(exposure)}, timeout=SET_TIMEOUT_S)
        client.set_number(device, 'CCD_CONTROLS', {'Gain': gain}, timeout=SET_TIMEOUT_S)
        client.set_number(device, 'STREAMING_EXPOSURE',
                          {'STREAMING_EXPOSURE_VALUE': exposure, 'STREAMING_DIVISOR_VALUE': 1},
                          timeout=SET_TIMEOUT_S)

    def _stream(self, on):
        self.client.set_switch(_device(self.client), 'CCD_VIDEO_STREAM',
                               on_switches=['STREAM_ON' if on else 'STREAM_OFF'])

    def _restart_simulated(self):
        with self._cond:
            exposure, region = self._exposure, self._streaming
        self.client.stop_video()
        self.client.start_video(exposure, region)

    def _set_streaming(self, region):
        with self._cond:
            self._streaming = tuple(region)
            self._full_scale = None  # decided again on the first frame of this stream

    def _fps(self, exposure) -> float:
        """One picture per exposure, never more than the driver allows."""
        fps = 1.0 / exposure
        if self._fps_max is None:
            try:
                ctl = self.client.get_control(_device(self.client), 'LIMITS', 'number', timeout=CONTROL_TIMEOUT_S)
                for e in ctl:
                    if e.getName() == 'LIMITS_PREVIEW_FPS':
                        self._fps_max = _element_max(e)
            except Exception:
                logger.warning('Focus: the driver gave no preview frame rate limit')
        if self._fps_max is not None and self._fps_max > 0:
            fps = min(fps, self._fps_max)
        return fps

    def _sensor_size(self):
        if self._sensor is None:
            ctl = self.client.get_control(_device(self.client), 'CCD_FRAME', 'number', timeout=CONTROL_TIMEOUT_S)
            sizes = {e.getName(): _element_max(e) for e in ctl}
            self._sensor = (int(sizes['WIDTH']), int(sizes['HEIGHT']))
        return self._sensor

    # -- blobs (on the INDI client's thread) -------------------------------------------------------------------------

    def _on_blob(self, data, fmt, t):
        with self._cond:
            if not self._running or self._streaming is None:
                return
            region, exposure, gain = self._streaming, self._exposure, self._gain
            wanted = self._region
            whole = self._whole_waiting and region[2:] == self._sensor_size_even()
        raw, full_scale = decode_stream(data, region[2], region[3], fmt)
        if raw is None:
            logger.debug('Focus: a blob that is not a frame of the stream was dropped')
            return
        mosaic = raw.dtype == np.uint16 or len(data) == region[2] * region[3]  # not a debayered or JPEG frame
        if raw.dtype == np.uint16:
            full_scale = self._latch_full_scale(raw)
        raw = np.array(raw, copy=True)
        with self._cond:
            if whole:
                self._whole = raw
                self._cond.notify_all()
                return
            if region != wanted or self._whole_waiting:
                return  # streaming some other region for a moment
        frame = RawFrame(raw=raw, full_scale=full_scale, bayer=self.bayer if mosaic else None, t_end=float(t),
                         exposure=exposure, gain=gain)
        while self.frames.qsize() >= MAX_QUEUED:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                break
        self.frames.put(frame)

    def _sensor_size_even(self):
        w, h = self._sensor
        return (w - w % 2, h - h % 2)

    def _latch_full_scale(self, raw) -> float:
        """65535 unless the camera's bits are fewer and the data is plainly not shifted up to 16 bits: its peak fits
        in those bits and some value is not a multiple of the shift. Decided once per stream."""
        with self._cond:
            if self._full_scale is not None:
                return self._full_scale
        full = 65535.0
        if self._bits is not None and self._bits < 16:
            top = (1 << self._bits) - 1
            step = 1 << (16 - self._bits)
            if int(raw.max()) <= top and bool(np.any(raw % step)):
                full = float(top)
        with self._cond:
            if self._full_scale is None:
                self._full_scale = full
            return self._full_scale
