from __future__ import annotations

import json
import os
import queue
import time
from types import SimpleNamespace

import numpy as np
import pytest

from indi_allsky import constants
from indi_allsky.focus.capture_session import FocusGlue
from indi_allsky.focus.capture_session import open_link
from indi_allsky.focus.capture_session import state_path_for
from indi_allsky.focus.controller import focus_sleep
from indi_allsky.focus.controller import focus_wait_turn
from indi_allsky.focus.controller import run_session_pass
from indi_allsky.focus.link import LinkClient
from indi_allsky.focus.link import LinkServer
from tests.focus.test_controller import Page
from tests.focus.test_controller import wait_for
from tests.focus.test_indi_stream import FakeClient


CCD_INFO = {
    'CCD_EXPOSURE': {'CCD_EXPOSURE_VALUE': {'min': 0.000032, 'max': 3600.0}},
    'CCD_INFO': {'CCD_BITSPERPIXEL': {'current': 16}},
    'CCD_FRAME': {'WIDTH': {'max': 640}, 'HEIGHT': {'max': 480}},
}


class FakeMiscDb(object):
    def __init__(self):
        self.states = []

    def setState(self, key, value):
        self.states.append((key, value))

    def last(self, key):
        values = [v for k, v in self.states if k == key]
        return values[-1] if values else None


class Camera(FakeClient):
    """A video camera with a 30 s normal exposure under way."""

    def __init__(self):
        super(Camera, self).__init__()
        self.image_q = queue.Queue()
        self.exposing = True
        self.disconnected = False
        self.ccd_removed = False

    def abortCcdExposure(self):
        self.aborts += 1
        self.exposing = False

    def getCcdExposureStatus(self):
        return (not self.exposing), ('BUSY' if self.exposing else 'READY')


@pytest.fixture
def rig(sock_path, closers):
    link = LinkServer(sock_path)
    link.start()
    closers.append(link)
    camera = Camera()
    reconfigured = []
    glue = FocusGlue(
        camera, link,
        camera_interface='test_rotating_stars', device_name='Rotating Stars Test Camera',
        ccd_info=CCD_INFO, cfa=None, gain=lambda: 7.0,
        state_path=state_path_for(sock_path),
        reconfigure=lambda: reconfigured.append(True),
    )
    worker = SimpleNamespace(focus=glue.service, _shutdown=False, _miscDb=FakeMiscDb())

    def page():
        before = set(link.clients())
        client = LinkClient(sock_path)
        client.connect(hello={'controls': True})
        closers.append(client)
        assert wait_for(lambda: set(link.clients()) - before)
        return Page(client)

    return SimpleNamespace(link=link, camera=camera, glue=glue, worker=worker, page=page,
                           reconfigured=reconfigured, state_path=state_path_for(sock_path))


def start_session(r):
    page = r.page()
    page.client.send({'kind': 'start'})
    assert wait_for(lambda: run_session_pass(r.worker) and r.worker.focus.active, timeout=3.0)
    return page


def test_state_file_sits_next_to_the_socket():
    assert state_path_for('/var/lib/indi-allsky/focus-session.sock') == \
        '/var/lib/indi-allsky/focus-session.state.json'


def test_start_during_long_exposure_breaks_wait(rig):
    page = rig.page()
    assert run_session_pass(rig.worker) is False  # idle: the normal schedule goes on
    page.client.send({'kind': 'start'})
    time.sleep(0.05)  # the command is on its way
    t0 = time.monotonic()
    left = False
    while time.monotonic() - t0 < 11:  # capture's wait loop, around a 30 s exposure
        if focus_wait_turn(rig.worker):
            left = True
            break
    elapsed = time.monotonic() - t0
    assert left and elapsed < 0.2
    assert rig.camera.aborts == 1  # counted before the session ends
    assert rig.worker.focus.active
    assert os.path.exists(rig.state_path)
    rig.worker.focus.stop('done')


def test_start_saves_the_camera_state_next_to_the_socket(rig):
    start_session(rig)
    with open(rig.state_path) as f:
        saved = json.load(f)
    assert saved['numbers']['CCD_CONTROLS']['Gain'] == 100.0
    assert saved['switches']['CCD_VIDEO_STREAM'] == ['STREAM_OFF']
    assert rig.camera.blob_sink is not None
    rig.worker.focus.stop('done')


def test_service_feeds_watchdog(rig):
    start_session(rig)
    db = rig.worker._miscDb
    db.states.clear()
    later = time.monotonic() + 60  # past the status throttle
    t0 = time.monotonic()
    assert run_session_pass(rig.worker, clock=lambda: later) is True
    assert time.monotonic() - t0 >= 0.04  # it paces the loop
    assert db.last('STATUS') == constants.STATUS_FOCUSING
    assert db.last('WATCHDOG') is not None
    written = len(db.states)
    assert run_session_pass(rig.worker, clock=lambda: later + 0.1) is True
    assert len(db.states) == written  # not a database write 20 times a second
    rig.worker.focus.stop('done')


def test_stop_ends_session_and_restores(rig):
    page = start_session(rig)
    rig.camera.taken()
    rig.worker._shutdown = True  # SIGTERM, SIGINT or SIGHUP
    assert run_session_pass(rig.worker) is True
    assert not rig.worker.focus.active
    assert page.json('ended')['reason'] == 'stopped'
    calls = rig.camera.taken()
    assert ('switch', 'CCD_VIDEO_STREAM', ('STREAM_OFF',)) in calls
    assert [c[1] for c in calls if c[0] == 'number'][-1] == 'CCD_CONTROLS'
    assert ('switch', 'CCD_VIDEO_STREAM', ('STREAM_ON',)) not in calls
    assert rig.camera.numbers['CCD_CONTROLS']['Gain'][0] == 100.0
    assert rig.camera.blob_sink is None
    assert not os.path.exists(rig.state_path)
    assert rig.reconfigured == [True]
    assert rig.worker._miscDb.last('STATUS') == constants.STATUS_RUNNING
    assert run_session_pass(rig.worker) is False  # the normal schedule resumes


def test_done_from_the_page_ends_and_restores(rig):
    page = start_session(rig)
    page.client.send({'kind': 'done'})
    assert wait_for(lambda: not run_session_pass(rig.worker) and not rig.worker.focus.active, timeout=3.0)
    assert page.json('ended')['reason'] == 'done'
    assert not os.path.exists(rig.state_path)
    assert rig.reconfigured == [True]


def test_disconnected_camera_keeps_the_state_file_for_startup(rig):
    start_session(rig)
    rig.camera.disconnected = True
    rig.worker.focus.stop('stopped')
    assert os.path.exists(rig.state_path)
    assert rig.reconfigured == []


def test_a_failing_poll_ends_the_session_not_capture(rig):
    start_session(rig)

    def broken():
        raise RuntimeError('bug')

    rig.worker.focus.controller.service = broken
    assert run_session_pass(rig.worker) is True
    assert not rig.worker.focus.active
    assert not os.path.exists(rig.state_path)


def test_failed_start_still_leaves_the_wait_and_cleans_up(rig):
    def broken(*args, **kwargs):
        raise RuntimeError('no camera')

    rig.camera.set_number = broken  # the stream cannot start
    page = rig.page()
    page.client.send({'kind': 'start'})
    time.sleep(0.05)
    assert wait_for(lambda: focus_wait_turn(rig.worker), timeout=2.0)  # the exposure was aborted all the same
    assert rig.camera.aborts == 1
    assert not rig.worker.focus.active
    assert not os.path.exists(rig.state_path)
    assert focus_wait_turn(rig.worker) is False


def test_classic_rig_refuses_start_without_touching_the_camera(sock_path, closers):
    link = LinkServer(sock_path)
    link.start()
    closers.append(link)
    camera = Camera()
    glue = FocusGlue(camera, link, camera_interface='indi_passive', device_name='x', ccd_info=CCD_INFO, cfa=None,
                     gain=lambda: 0.0, state_path=state_path_for(sock_path), reconfigure=lambda: None)
    worker = SimpleNamespace(focus=glue.service, _shutdown=False, _miscDb=FakeMiscDb())
    client = LinkClient(sock_path)
    client.connect(hello={'controls': True})
    closers.append(client)
    page = Page(client)
    assert wait_for(lambda: link.clients())
    client.send({'kind': 'start'})
    err = page.json('error', pump=lambda: run_session_pass(worker))
    assert err['command'] == 'start'
    assert camera.aborts == 0 and not glue.service.active
    client.send({'kind': 'start'})
    time.sleep(0.05)
    assert focus_wait_turn(worker) is False  # nothing was aborted: capture keeps waiting for its exposure


def test_stills_rig_uses_the_image_queue(sock_path, closers):
    link = LinkServer(sock_path)
    link.start()
    closers.append(link)
    camera = Camera()
    camera.setCcdExposure = lambda *a, **k: None
    original_q = camera.image_q
    glue = FocusGlue(camera, link, camera_interface='indi_accumulator', device_name='x', ccd_info=CCD_INFO,
                     cfa=None, gain=lambda: 0.0, state_path=state_path_for(sock_path), reconfigure=lambda: None)
    worker = SimpleNamespace(focus=glue.service, _shutdown=False, _miscDb=FakeMiscDb())
    client = LinkClient(sock_path)
    client.connect(hello={'controls': True})
    closers.append(client)
    assert wait_for(lambda: link.clients())
    client.send({'kind': 'start'})
    assert wait_for(lambda: run_session_pass(worker) and glue.service.active, timeout=3.0)
    assert camera.image_q is not original_q
    assert camera.blob_sink is None
    glue.service.stop('done')
    assert camera.image_q is original_q


def test_open_link_in_use_gives_none(sock_path, closers):
    first = open_link(sock_path)
    closers.append(first)
    assert first is not None
    assert open_link(sock_path) is None


def test_open_link_on_a_directory_gives_none(tmp_path):
    assert open_link(str(tmp_path)) is None


def test_paused_sleep_ends_on_start_and_answers_pages(rig):
    page = rig.page()
    page.client.send({'kind': 'ping', 'id': 3})
    t0 = time.monotonic()
    assert focus_sleep(rig.worker, 0.3) is False  # nothing to start: the whole sleep
    assert time.monotonic() - t0 >= 0.3
    assert page.json('pong')['id'] == 3  # answered during the sleep
    page.client.send({'kind': 'start'})
    t0 = time.monotonic()
    assert focus_sleep(rig.worker, 31) is True
    assert time.monotonic() - t0 < 1.0
    rig.worker.focus.stop('done')


def test_sleep_without_a_link_just_sleeps():
    slept = []
    assert focus_sleep(SimpleNamespace(focus=None), 31, sleep=slept.append) is False
    assert slept == [31]


def test_blob_sink_bypasses_image_queue(fake_pyindi):
    indi = fake_pyindi('indi')

    class Q(object):
        def put(self, *a, **k):
            raise AssertionError('image_q touched')

    class Blob(object):
        def __init__(self, data, fmt):
            self._data, self._fmt = data, fmt

        def getblobdata(self):
            return self._data

        def getFormat(self):
            return self._fmt

    client = indi.IndiClient.__new__(indi.IndiClient)
    client.image_q = Q()
    client.exposureStartTime = 0
    got = []
    client.blob_sink = lambda data, fmt, t: got.append((data, fmt, t))
    client.processBlob(Blob(b'\x00\x01' * 8, '.stream'))
    assert got and got[0][:2] == (b'\x00\x01' * 8, '.stream') and isinstance(got[0][2], float)

    def raising(data, fmt, t):
        raise RuntimeError('sink bug')

    client.blob_sink = raising
    client.processBlob(Blob(b'xx', '.stream'))  # never kills the INDI thread

    client.blob_sink = None
    client.processBlob(Blob(b'xx', '.stream'))  # a late stream frame with no session: dropped
    client.processBlob(Blob(b'xx', '.stream.z'))
    assert indi.IndiClient.blob_sink is None  # off unless a session sets it


def test_indi_client_has_video_checks_the_property(fake_pyindi):
    indi = fake_pyindi('indi')

    class Prop(object):
        def __init__(self, valid):
            self.valid = valid

        def isValid(self):
            return self.valid

        def __bool__(self):
            return True

    class Device(object):
        def __init__(self, prop):
            self.prop = prop

        def getSwitch(self, name):
            return self.prop if name == 'CCD_VIDEO_STREAM' else None

    client = indi.IndiClient.__new__(indi.IndiClient)
    client._ccd_device = Device(Prop(True))
    assert client.has_video() is True
    client._ccd_device = Device(Prop(False))
    assert client.has_video() is False
    client._ccd_device = Device(None)
    assert client.has_video() is False
    client._ccd_device = object()  # FakeIndiCcd and friends have no switches
    assert client.has_video() is False


def test_test_camera_streams_frames(fake_pyindi):
    tc = fake_pyindi('test_cameras')

    class Tiny(tc.IndiClientTestCameraBase):
        def updateImage(self, binning):
            img = np.zeros((48, 64, 3), dtype=np.uint16)
            img[20:23, 30:33] = 30000
            self._image = img

    cam = Tiny.__new__(Tiny)
    got = []
    cam.blob_sink = lambda data, fmt, t: got.append((data, fmt, t))
    cam.sim_position = 0
    assert cam.has_video() is True
    cam.start_video(0.02, (16, 8, 32, 32))
    assert wait_for(lambda: len(got) >= 3, timeout=3.0)
    cam.stop_video()
    n = len(got)
    time.sleep(0.1)
    assert len(got) <= n + 1
    data, fmt, t = got[0]
    assert fmt == '.stream' and len(data) == 32 * 32 * 2
    raw = np.frombuffer(data, dtype='<u2').reshape(32, 32)
    assert raw[12:15, 14:17].min() == 30000  # the crop starts at (16, 8)

    cam.sim_position = 40  # out of focus: the star spreads
    got.clear()
    cam.start_video(0.02, (16, 8, 32, 32))
    assert wait_for(lambda: len(got) >= 1, timeout=3.0)
    cam.stop_video()
    blurred = np.frombuffer(got[0][0], dtype='<u2').reshape(32, 32)
    assert blurred.max() < raw.max()
    assert (blurred > 1000).sum() > (raw > 1000).sum()
