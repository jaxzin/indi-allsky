from __future__ import annotations

import json
import os
import threading
import time

import numpy as np
import pytest

from indi_allsky.focus import indi_stream
from indi_allsky.focus.indi_stream import NUMBERS
from indi_allsky.focus.indi_stream import SWITCHES
from indi_allsky.focus.indi_stream import IndiStreamSource
from indi_allsky.focus.indi_stream import focus_startup_restore
from indi_allsky.focus.indi_stream import has_video
from indi_allsky.focus.indi_stream import restore_state
from indi_allsky.focus.indi_stream import save_state


SENSOR = (640, 480)


class Elem(object):
    def __init__(self, name, value=0.0, low=0.0, high=1e9, on=False):
        self._name, self._value, self._min, self._max, self._on = name, value, low, high, on

    def getName(self):
        return self._name

    def getValue(self):
        return self._value

    def getMin(self):
        return self._min

    def getMax(self):
        return self._max

    def getState(self):
        return 1 if self._on else 0


def default_numbers():
    return {
        'CCD_CONTROLS': {'Gain': (100.0, 0, 600), 'Offset': (10.0, 0, 255)},
        'CCD_FRAME': {'X': (0.0, 0, 640), 'Y': (0.0, 0, 480), 'WIDTH': (640.0, 1, 640), 'HEIGHT': (480.0, 1, 480)},
        'CCD_BINNING': {'HOR_BIN': (1.0, 1, 4), 'VER_BIN': (1.0, 1, 4)},
        'CCD_STREAM_FRAME': {'X': (0.0, 0, 640), 'Y': (0.0, 0, 480), 'WIDTH': (640.0, 1, 640),
                             'HEIGHT': (480.0, 1, 480)},
        'STREAMING_EXPOSURE': {'STREAMING_EXPOSURE_VALUE': (0.1, 0.000032, 3600),
                               'STREAMING_DIVISOR_VALUE': (1.0, 1, 100)},
        'LIMITS': {'LIMITS_BUFFER_MAX': (512.0, 1, 65536), 'LIMITS_PREVIEW_FPS': (10.0, 1, 30)},
        'CCD_EXPOSURE': {'CCD_EXPOSURE_VALUE': (1.0, 0.000032, 3600)},
    }


def default_switches():
    return {
        'STREAM_FULL_DEPTH': {'FULL_DEPTH_8BIT': True, 'FULL_DEPTH_16BIT': False},
        'CCD_STREAM_ENCODER': {'RAW': False, 'MJPEG': True},
        'CCD_VIDEO_FORMAT': {'ASI_IMG_RAW8': False, 'ASI_IMG_RAW16': True},
        'CCD_CAPTURE_FORMAT': {'ASI_IMG_RAW8': False, 'ASI_IMG_RAW16': True},
        'CCD_VIDEO_STREAM': {'STREAM_ON': False, 'STREAM_OFF': True},
        'CCD_ABORT_EXPOSURE': {'ABORT': False},
    }


class FakeClient(object):
    """Just enough of IndiClient: set_number, set_switch, get_control, blob_sink, ccd_device."""

    def __init__(self, numbers=None, switches=None):
        self.numbers = default_numbers() if numbers is None else numbers
        self.switches = default_switches() if switches is None else switches
        self.ccd_device = object()
        self.blob_sink = None
        self.calls = []
        self.lock = threading.Lock()
        self.aborts = 0

    def get_control(self, device, name, ctl_type, timeout=None):
        assert device is self.ccd_device
        assert timeout is not None and timeout > 0  # 0 waits forever in IndiClient.get_control
        if ctl_type == 'number' and name in self.numbers:
            return [Elem(k, v[0], v[1], v[2]) for k, v in self.numbers[name].items()]
        if ctl_type == 'switch' and name in self.switches:
            return [Elem(k, on=v) for k, v in self.switches[name].items()]
        raise TimeoutError('Timeout finding control {0}'.format(name))

    def set_number(self, device, name, values, sync=True, timeout=None):
        assert device is self.ccd_device
        if name not in self.numbers:
            raise TimeoutError(name)
        with self.lock:
            self.calls.append(('number', name, dict(values)))
        for k, v in values.items():
            low, high = self.numbers[name][k][1:]
            self.numbers[name][k] = (float(v), low, high)

    def set_switch(self, device, name, on_switches=[], off_switches=[], sync=True, timeout=None):
        assert device is self.ccd_device
        if name not in self.switches:
            raise TimeoutError(name)
        with self.lock:
            self.calls.append(('switch', name, tuple(on_switches)))
        for k in self.switches[name]:
            self.switches[name][k] = k in on_switches

    def abortCcdExposure(self):
        self.aborts += 1

    def taken(self):
        with self.lock:
            calls, self.calls = self.calls, []
        return calls


def stream(on):
    return ('switch', 'CCD_VIDEO_STREAM', ('STREAM_ON' if on else 'STREAM_OFF',))


def test_indi_stream_start_order():
    client = FakeClient()
    src = IndiStreamSource(client, sensor=SENSOR)
    src.start((100, 50, 256, 128), 0.2, 150)
    assert client.taken() == [
        stream(False),
        ('number', 'CCD_FRAME', {'X': 100, 'Y': 50, 'WIDTH': 256, 'HEIGHT': 128}),
        ('number', 'CCD_STREAM_FRAME', {'X': 0, 'Y': 0, 'WIDTH': 256, 'HEIGHT': 128}),
        ('switch', 'CCD_STREAM_ENCODER', ('RAW',)),
        ('switch', 'STREAM_FULL_DEPTH', ('FULL_DEPTH_16BIT',)),
        ('number', 'LIMITS', {'LIMITS_PREVIEW_FPS': 5.0}),
        ('number', 'CCD_CONTROLS', {'Gain': 150.0}),
        ('number', 'STREAMING_EXPOSURE', {'STREAMING_EXPOSURE_VALUE': 0.2, 'STREAMING_DIVISOR_VALUE': 1}),
        stream(True),
    ]
    assert client.blob_sink is not None
    src.stop()


def test_driver_specific_properties_are_skipped_when_absent():
    # STREAM_FULL_DEPTH and CCD_CONTROLS.Gain are ZWO's; LIMITS comes with INDI's stream manager. A driver without
    # one still streams, and IndiClient.set_switch/set_number would otherwise wait out their full default timeout.
    client = FakeClient()
    for name in ('STREAM_FULL_DEPTH',):
        del client.switches[name]
    for name in ('LIMITS', 'CCD_CONTROLS'):
        del client.numbers[name]
    src = IndiStreamSource(client, sensor=SENSOR)
    src.start((0, 0, 64, 64), 0.2, 150)
    src.set_camera(0.5, 10)
    names = [c[1] for c in client.taken()]
    assert not {'STREAM_FULL_DEPTH', 'LIMITS', 'CCD_CONTROLS'} & set(names)
    assert names.count('STREAMING_EXPOSURE') == 2
    src.stop()


def test_indi_stream_whole_sensor_when_no_region():
    client = FakeClient()
    src = IndiStreamSource(client, sensor=SENSOR)
    src.start(None, 0.1, 0)
    frame = [c for c in client.taken() if c[1] == 'CCD_FRAME'][0]
    assert frame[2] == {'X': 0, 'Y': 0, 'WIDTH': 640, 'HEIGHT': 480}
    src.stop()


def fps_set(calls):
    return [c[2]['LIMITS_PREVIEW_FPS'] for c in calls if c[1] == 'LIMITS']


def test_indi_stream_fps_always_capped():
    client = FakeClient()  # the driver's LIMITS_PREVIEW_FPS max is 30
    src = IndiStreamSource(client, sensor=SENSOR)
    src.start(None, 0.001, 0)  # 1000 fps asked
    assert fps_set(client.taken()) == [30.0]
    src.set_camera(0.0001, 0)
    assert fps_set(client.taken()) == [30.0]
    src.set_camera(2.0, 0)  # long exposures never ask for more than one picture per exposure
    assert fps_set(client.taken()) == [0.5]
    src.set_camera(0.05, 0)
    assert fps_set(client.taken()) == [20.0]
    src.stop()


def test_set_camera_restarts_stream():
    client = FakeClient()
    src = IndiStreamSource(client, sensor=SENSOR)
    src.start((0, 0, 64, 64), 0.1, 0)
    client.taken()
    src.set_camera(0.25, 300)
    assert client.taken() == [
        stream(False),
        ('number', 'LIMITS', {'LIMITS_PREVIEW_FPS': 4.0}),
        ('number', 'CCD_CONTROLS', {'Gain': 300.0}),
        ('number', 'STREAMING_EXPOSURE', {'STREAMING_EXPOSURE_VALUE': 0.25, 'STREAMING_DIVISOR_VALUE': 1}),
        stream(True),
    ]
    src.stop()


def test_saved_state_is_exact_property_list():
    assert NUMBERS == ('CCD_CONTROLS', 'CCD_FRAME', 'CCD_BINNING', 'CCD_STREAM_FRAME', 'STREAMING_EXPOSURE', 'LIMITS')
    assert SWITCHES == ('STREAM_FULL_DEPTH', 'CCD_STREAM_ENCODER', 'CCD_VIDEO_FORMAT', 'CCD_CAPTURE_FORMAT',
                        'CCD_VIDEO_STREAM')
    client = FakeClient()
    state = save_state(client)
    assert set(state) == {'numbers', 'switches'}
    assert set(state['numbers']) == set(NUMBERS)  # CCD_EXPOSURE is on the device but not saved
    assert set(state['switches']) == set(SWITCHES)  # CCD_ABORT_EXPOSURE neither
    assert state['numbers']['CCD_CONTROLS'] == {'Gain': 100.0, 'Offset': 10.0}
    assert state['switches']['CCD_STREAM_ENCODER'] == ['MJPEG']
    json.dumps(state)  # it goes to a file
    assert client.taken() == []  # saving only reads


def test_saved_state_skips_a_missing_property():
    client = FakeClient()
    del client.numbers['LIMITS']
    state = save_state(client)
    assert 'LIMITS' not in state['numbers']
    assert set(state['numbers']) == set(NUMBERS) - {'LIMITS'}


def test_restore_puts_values_back_and_never_starts_the_stream():
    client = FakeClient()
    state = save_state(client)
    state['switches']['CCD_VIDEO_STREAM'] = ['STREAM_ON']  # saved while some other client streamed
    src = IndiStreamSource(client, sensor=SENSOR)
    src.start((0, 0, 64, 64), 0.5, 400)
    src.stop()
    client.taken()
    assert restore_state(client, state) == 0
    calls = client.taken()
    assert calls[0] == stream(False)
    assert stream(True) not in calls
    assert [c[1] for c in calls[1:5]] == ['STREAM_FULL_DEPTH', 'CCD_STREAM_ENCODER', 'CCD_VIDEO_FORMAT',
                                          'CCD_CAPTURE_FORMAT']
    assert [c[1] for c in calls[5:]] == ['CCD_BINNING', 'CCD_FRAME', 'CCD_STREAM_FRAME', 'STREAMING_EXPOSURE',
                                         'LIMITS', 'CCD_CONTROLS']
    assert client.numbers['CCD_CONTROLS']['Gain'][0] == 100.0
    assert client.numbers['CCD_FRAME']['WIDTH'][0] == 640.0
    assert client.switches['CCD_STREAM_ENCODER'] == {'RAW': False, 'MJPEG': True}


def test_restore_goes_on_past_a_failing_property():
    client = FakeClient()
    state = save_state(client)
    state['numbers']['NOT_THERE'] = {'A': 1.0}
    del client.numbers['CCD_BINNING']
    assert restore_state(client, state) == 1  # CCD_BINNING failed; NOT_THERE is not on the list
    names = [c[1] for c in client.taken()]
    assert 'CCD_CONTROLS' in names and 'NOT_THERE' not in names


def test_startup_restores_after_crash(tmp_path):
    client = FakeClient()
    state = save_state(client)
    path = str(tmp_path / 'focus-session.state.json')
    with open(path, 'w') as f:
        json.dump(state, f)
    client.numbers['CCD_CONTROLS']['Gain'] = (400.0, 0, 600)  # the session's gain, left by a crash
    assert focus_startup_restore(client, path) is True
    assert client.numbers['CCD_CONTROLS']['Gain'][0] == 100.0
    assert not os.path.exists(path)


def test_startup_restore_without_a_file_does_nothing(tmp_path):
    client = FakeClient()
    assert focus_startup_restore(client, str(tmp_path / 'focus-session.state.json')) is False
    assert client.taken() == []


def test_startup_restore_deletes_a_corrupt_file(tmp_path):
    client = FakeClient()
    path = str(tmp_path / 'focus-session.state.json')
    with open(path, 'w') as f:
        f.write('{not json')
    assert focus_startup_restore(client, path) is False
    assert not os.path.exists(path)
    assert client.taken() == []


def test_has_video():
    assert has_video(FakeClient()) is True
    client = FakeClient()
    del client.switches['CCD_VIDEO_STREAM']
    assert has_video(client) is False

    class Says(object):
        def has_video(self):
            return True

    assert has_video(Says()) is True


def u16(img):
    return np.ascontiguousarray(img, dtype='<u2').tobytes()


def test_blobs_become_frames_of_the_region():
    client = FakeClient()
    src = IndiStreamSource(client, sensor=SENSOR, bayer='rggb')
    src.start((10, 20, 64, 32), 0.1, 50)
    img = np.arange(64 * 32, dtype=np.uint16).reshape(32, 64) * 16
    client.blob_sink(u16(img), '.stream', 1234.5)
    frame = src.frames.get(timeout=1)
    assert frame.raw.shape == (32, 64)
    assert np.array_equal(frame.raw, img)
    assert frame.bayer == 'RGGB' and src.bayer == 'RGGB'
    assert (frame.t_end, frame.exposure, frame.gain) == (1234.5, 0.1, 50.0)
    assert frame.full_scale == 65535.0
    client.blob_sink(b'not a frame', '.stream', 1235.0)  # dropped
    assert src.frames.empty()
    src.stop()


def test_unscaled_12_bit_stream_has_12_bit_full_scale():
    client = FakeClient()
    src = IndiStreamSource(client, sensor=SENSOR, bits=12)
    src.start((0, 0, 32, 32), 0.1, 0)
    img = np.full((32, 32), 300, dtype=np.uint16)
    img[5, 5] = 4095
    img[6, 6] = 301  # not a multiple of 16: the data is not shifted up to 16 bits
    client.blob_sink(u16(img), '.stream', 1.0)
    assert src.frames.get(timeout=1).full_scale == 4095.0
    img[6, 6] = 70  # the scale is kept for the whole stream
    client.blob_sink(u16(img * 16), '.stream', 2.0)
    assert src.frames.get(timeout=1).full_scale == 4095.0
    src.stop()


def test_shifted_12_bit_stream_has_16_bit_full_scale():
    client = FakeClient()
    src = IndiStreamSource(client, sensor=SENSOR, bits=12)
    src.start((0, 0, 32, 32), 0.1, 0)
    img = (np.arange(32 * 32, dtype=np.uint16).reshape(32, 32) % 4096) * 16
    client.blob_sink(u16(img), '.stream', 1.0)
    assert src.frames.get(timeout=1).full_scale == 65535.0
    src.stop()


def test_frames_queue_keeps_only_the_newest():
    client = FakeClient()
    src = IndiStreamSource(client, sensor=SENSOR)
    src.start((0, 0, 8, 8), 0.1, 0)
    for i in range(20):
        client.blob_sink(u16(np.full((8, 8), i)), '.stream', float(i))
    got = []
    while not src.frames.empty():
        got.append(src.frames.get_nowait().t_end)
    assert len(got) <= indi_stream.MAX_QUEUED and got[-1] == 19.0
    src.stop()


def test_whole_frame_streams_the_sensor_then_the_region_again():
    client = FakeClient()
    src = IndiStreamSource(client, sensor=(64, 48))
    src.start((8, 8, 16, 16), 0.1, 0)
    client.taken()
    result = {}

    def take():
        result['raw'] = src.whole_frame(timeout=5.0)

    t = threading.Thread(target=take)
    t.start()
    deadline = time.monotonic() + 5
    while not [c for c in client.calls if c[1] == 'CCD_FRAME'] and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.05)
    client.blob_sink(u16(np.full((48, 64), 7)), '.stream', 1.0)
    t.join(5)
    assert result['raw'].shape == (48, 64)
    frames = [c[2] for c in client.taken() if c[1] == 'CCD_FRAME']
    assert frames == [{'X': 0, 'Y': 0, 'WIDTH': 64, 'HEIGHT': 48}, {'X': 8, 'Y': 8, 'WIDTH': 16, 'HEIGHT': 16}]
    assert src.frames.empty()  # the whole-sensor picture is not a region frame
    src.stop()


def test_stop_releases_whole_frame_and_the_sink():
    client = FakeClient()
    src = IndiStreamSource(client, sensor=(64, 48))
    src.start(None, 0.1, 0)
    errors = []

    def take():
        try:
            src.whole_frame(timeout=10.0)
        except Exception as e:
            errors.append(e)

    t = threading.Thread(target=take)
    t.start()
    time.sleep(0.1)
    src.stop()
    t.join(2)
    assert not t.is_alive() and errors
    assert client.blob_sink is None
    assert client.taken()[-1] == stream(False)
    with pytest.raises(RuntimeError):
        src.whole_frame(timeout=1.0)


class VideoCamera(FakeClient):
    """A client that streams by itself, like the test cameras."""

    def __init__(self):
        super(VideoCamera, self).__init__()
        self.video = []

    def start_video(self, exposure, roi):
        self.video.append(('start', exposure, tuple(roi)))

    def stop_video(self):
        self.video.append(('stop',))


def test_simulated_camera_streams_through_start_video():
    cam = VideoCamera()
    src = IndiStreamSource(cam, sensor=SENSOR)
    src.start((0, 0, 32, 32), 0.1, 0)
    src.set_camera(0.2, 0)
    src.stop()
    assert cam.video == [('stop',), ('start', 0.1, (0, 0, 32, 32)), ('stop',), ('start', 0.2, (0, 0, 32, 32)),
                         ('stop',)]
    assert cam.taken() == []  # no INDI properties
