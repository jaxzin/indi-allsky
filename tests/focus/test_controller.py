from __future__ import annotations

import queue
import time

import numpy as np
import pytest

from indi_allsky.focus import controller as controller_mod
from indi_allsky.focus.controller import CameraInfo
from indi_allsky.focus.controller import FocusController
from indi_allsky.focus.controller import FocusService
from indi_allsky.focus.exposure_search import ExposureSearch
from indi_allsky.focus.exposure_search import MSG_NO_STARS
from indi_allsky.focus.frame import Stretch
from indi_allsky.focus.frame import unpack_header
from indi_allsky.focus.link import LinkClient
from indi_allsky.focus.link import LinkServer
from indi_allsky.focus.measure import grade
from indi_allsky.focus.rigs import RigDecision
from indi_allsky.focus.sources import RawFrame
from tests.focus import synth


SHAPE = (256, 256)
RIG = RigDecision('new', 'stills', 'test rig')
GRID = synth.grid_stars(SHAPE, 1.5, 20000)
# 16 stars from faint to bright: the faint ones grade weak at higher brightness limits
GRADED = [synth.Star(s.x, s.y, s.sigma, f) for s, f in zip(GRID, np.geomspace(300, 30000, len(GRID)))]


def wait_for(cond, timeout=5.0):
    deadline = time.monotonic() + timeout
    while True:
        value = cond()
        if value or time.monotonic() >= deadline:
            return value
        time.sleep(0.01)


class FakeClock(object):
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t


class Epoch(object):
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeSource(object):
    def __init__(self):
        self.frames = queue.Queue()
        self.calls = []
        self.whole = synth.frame(SHAPE)

    def start(self, roi, exposure, gain):
        self.calls.append(('start', roi, exposure, gain))

    def set_camera(self, exposure, gain):
        self.calls.append(('set_camera', exposure, gain))

    def whole_frame(self):
        self.calls.append(('whole_frame',))
        return self.whole

    def stop(self):
        self.calls.append(('stop',))

    def kinds(self):
        return [c[0] for c in self.calls]


def raw_frame(stars=(), t_end=1001.0, exposure=1.0, gain=100.0, background=synth.BACKGROUND, seed=synth.SEED):
    return RawFrame(synth.frame(SHAPE, stars=stars, background=background, seed=seed), 65535.0, 'RGGB',
                    t_end, exposure, gain)


class Page(object):
    """A LinkClient plus everything it has received so far."""

    def __init__(self, client):
        self.client = client
        self.got = []

    def next(self, pred, timeout=5.0, pump=None):
        """The oldest unclaimed message matching pred, reading more as needed. `pump` runs between reads."""
        deadline = time.monotonic() + timeout
        seen = 0
        while True:
            for i in range(seen, len(self.got)):
                if pred(self.got[i]):
                    return self.got.pop(i)
            seen = len(self.got)
            if time.monotonic() >= deadline:
                raise AssertionError('timed out; got {0!r}'.format([_short(m) for m in self.got]))
            if pump is not None:
                pump()
            msg = self.client.recv(timeout=0.05)
            if msg is not None:
                self.got.append(msg)

    def json(self, kind, timeout=5.0, pump=None, **match):
        return self.next(lambda m: m[0] == 'json' and m[1].get('kind') == kind
                         and all(m[1].get(k) == v for k, v in match.items()), timeout, pump)[1]

    def frame(self, timeout=5.0):
        return unpack_header(self.next(lambda m: m[0] == 'frame', timeout)[1])

    def drain(self, wait=0.2):
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            msg = self.client.recv(timeout=0.02)
            if msg is not None:
                self.got.append(msg)
        out, self.got = self.got, []
        return out


def _short(m):
    return ('frame', len(m[1])) if m[0] == 'frame' else m[1]


@pytest.fixture
def link(sock_path, closers):
    server = LinkServer(sock_path)
    server.start()
    closers.append(server)
    return server


@pytest.fixture
def connect(link, sock_path, closers):
    def _connect(controls=True):
        before = set(link.clients())
        client = LinkClient(sock_path)
        client.connect(hello={'controls': True} if controls else None)
        closers.append(client)
        assert wait_for(lambda: set(link.clients()) - before)
        (cid,) = set(link.clients()) - before
        return cid, Page(client)
    return _connect


@pytest.fixture
def source():
    return FakeSource()


@pytest.fixture
def epoch():
    return Epoch()


@pytest.fixture
def make(link, source, epoch, closers):
    def _make(**kw):
        ctrl = FocusController(source, link, FakeClock(), RIG, False, epoch=epoch,
                               exposure=kw.pop('exposure', 1.0), gain=kw.pop('gain', 100.0), **kw)
        closers.append(_Ender(ctrl))
        return ctrl
    return _make


class _Ender(object):
    def __init__(self, ctrl):
        self.ctrl = ctrl

    def close(self):
        if self.ctrl.active:
            self.ctrl.end('stopped')


def reading(page, frame_no=None, ctrl=None):
    match = {} if frame_no is None else {'frame': frame_no}
    return page.json('reading', pump=ctrl.service if ctrl else None, **match)


def run_pick(ctrl, page, source, epoch, stars=GRADED, exposure=2.0):
    """Press Pick stars at epoch 1000 and feed three frames begun after it; return the `stars` message."""
    epoch.t = 1000.0
    ctrl.handle(page.cid, {'kind': 'pick_stars'})
    for i in range(3):  # one at a time: a worker that falls behind skips frames
        source.frames.put(raw_frame(stars, t_end=1000.0 + exposure * (i + 1), exposure=exposure))
        page.json('progress', what='pick', frame=i + 1, pump=ctrl.service)
    return page.json('stars', pump=ctrl.service, timeout=10.0)


@pytest.fixture
def session(connect, make, source, epoch):
    """A begun controller and one page with controls."""
    cid, page = connect()
    page.cid = cid
    ctrl = make(exposure=2.0)
    ctrl.begin(cid)
    page.json('state')
    page.json('mode')
    return ctrl, page


# --- frames and readings ----------------------------------------------------------------------------------------

def test_frame_then_reading_order(session, source):
    ctrl, page = session
    source.frames.put(raw_frame(GRID, t_end=1003.0, exposure=2.0))
    msgs = []
    page.next(lambda m: msgs.append(m) or (m[0] == 'json' and m[1].get('kind') == 'reading'))
    frames = [unpack_header(m[1]) for m in msgs if m[0] == 'frame']
    assert len(frames) == 1
    head = frames[0]
    assert msgs[-1][1]['frame'] == head['frame_no'] == 1
    assert head['capture_time'] == pytest.approx(1002.0)  # mid-exposure: t_end - exposure / 2
    assert head['exposure'] == pytest.approx(2.0)
    assert msgs.index(next(m for m in msgs if m[0] == 'frame')) < len(msgs) - 1


def test_reading_measured_before_clock(connect, make, source, epoch, monkeypatch):
    def run():
        cid, page = connect()
        page.cid = cid
        ctrl = make(exposure=2.0)
        ctrl.begin(cid)
        run_pick(ctrl, page, source, epoch)
        source.frames.put(raw_frame(GRADED, t_end=1010.0, exposure=2.0))
        got = reading(page, 4, ctrl)
        ctrl.end('stopped')
        page.client.close()
        return got

    plain = run()

    def white(img, capture_time, frame_no):
        img[...] = 255

    monkeypatch.setattr(controller_mod, 'draw_clock', white)
    assert run() == plain
    assert plain['measure'] == 'stars' and plain['size'] > 0


# --- picking --------------------------------------------------------------------------------------------------------

def test_pick_uses_frames_begun_after_press(session, source, epoch):
    ctrl, page = session
    epoch.t = 1000.0
    ctrl.handle(page.cid, {'kind': 'pick_stars'})
    # begun at 999.5, before the press: it holds an extra star and must not be stacked
    extra = synth.Star(120.3, 72.2, 1.5, 20000)
    source.frames.put(raw_frame(GRID + [extra], t_end=1001.5, exposure=2.0))
    reading(page, 1, ctrl)
    for i in range(3):
        source.frames.put(raw_frame(GRID, t_end=1002.0 + 2.0 * (i + 1), exposure=2.0))
        progress = page.json('progress', what='pick', pump=ctrl.service, frame=i + 1)
        assert progress['of'] == 3 and progress['done'] is False
    stars = page.json('stars', pump=ctrl.service)
    assert len(stars['stars']) == len(GRID)
    assert not any(abs(s['x'] - extra.x) < 6 and abs(s['y'] - extra.y) < 6 for s in stars['stars'])
    for s in stars['stars']:
        assert s['snr1'] > 0
        assert s['state'] == grade(s['snr1'], stars['min_snr'])
    assert page.json('progress', what='pick', pump=ctrl.service, done=True)['message'] == 'Picked 16 stars.'


def test_min_snr_regrades_without_repick(session, source, epoch):
    ctrl, page = session
    first = run_pick(ctrl, page, source, epoch)
    page.json('progress', what='pick', done=True, pump=ctrl.service)
    page.drain()
    calls = list(source.calls)
    ctrl.handle(page.cid, {'kind': 'min_snr', 'value': 30})
    again = page.json('stars')
    assert again['min_snr'] == 30
    assert [(s['id'], s['x'], s['y'], s['snr1']) for s in again['stars']] == \
        [(s['id'], s['x'], s['y'], s['snr1']) for s in first['stars']]
    states = [s['state'] for s in again['stars']]
    assert states == [grade(s['snr1'], 30) for s in again['stars']]
    assert 'weak' in states and 'kept' in states
    assert source.calls == calls
    assert not [m for m in page.drain() if m[0] == 'json' and m[1].get('kind') == 'progress']

    ctrl.handle(page.cid, {'kind': 'min_snr', 'value': 500})
    assert page.json('stars')['min_snr'] == 60  # the slider's range is 3 to 60


def test_drop_and_pin(session, source, epoch):
    ctrl, page = session
    stars = run_pick(ctrl, page, source, epoch, stars=GRID)
    kept = [s['id'] for s in stars['stars'] if s['state'] == 'kept']
    assert len(kept) > 3
    target = kept[0]

    ctrl.handle(page.cid, {'kind': 'star', 'id': target, 'state': 'pin'})
    assert {s['id']: s['state'] for s in page.json('stars')['stars']}[target] == 'pinned'
    source.frames.put(raw_frame(GRID, t_end=1010.0, exposure=2.0))
    r = reading(page, 4, ctrl)
    rings = {ring['id']: ring for ring in r['rings']}
    assert r['stars_used'] == 1
    assert r['size'] == pytest.approx(rings[target]['size'])

    ctrl.handle(page.cid, {'kind': 'star', 'id': target, 'state': 'drop'})
    assert {s['id']: s['state'] for s in page.json('stars')['stars']}[target] == 'dropped'
    source.frames.put(raw_frame(GRID, t_end=1012.0, exposure=2.0))
    r = reading(page, 5, ctrl)
    assert r['stars_used'] == len(kept) - 1
    assert {ring['id']: ring['state'] for ring in r['rings']}[target] == 'dropped'


def test_set_camera_restarts_and_resets_stretch(session, source, monkeypatch):
    ctrl, page = session
    used = []
    real = controller_mod.preview

    def spy(gray, stretch):
        used.append(stretch)
        return real(gray, stretch)

    monkeypatch.setattr(controller_mod, 'preview', spy)
    linear = Stretch(0.0, 65535.0, 1.0)

    for n in (1, 2, 3):
        source.frames.put(raw_frame(GRID, t_end=1000.0 + 2 * n, exposure=2.0))
        reading(page, n, ctrl)
    assert used[0] == linear
    assert used[1] != linear and used[1].gamma == pytest.approx(0.8)
    assert used[2] is used[1]

    ctrl.handle(page.cid, {'kind': 'set_camera', 'exposure': 4.0, 'gain': 50, 'cmd_id': 3})
    assert ('set_camera', 4.0, 50.0) in source.calls
    mode = page.json('mode')
    assert (mode['exposure'], mode['gain']) == (4.0, 50.0)
    for n in (4, 5):
        source.frames.put(raw_frame(GRID, t_end=1100.0 + 4 * n, exposure=4.0, gain=50.0))
        reading(page, n, ctrl)
    assert used[3] == linear
    assert used[4] != linear


# --- what the number shows ------------------------------------------------------------------------------------------

def test_contrast_when_no_stars_and_bright(session, source):
    ctrl, page = session
    source.frames.put(raw_frame((), background=20000.0, exposure=2.0))
    r = reading(page, 1, ctrl)
    assert r['measure'] == 'contrast'
    assert r['contrast'] > 0
    assert r['size'] is None


def test_no_stars_dark_message(session, source):
    ctrl, page = session
    source.frames.put(raw_frame((), exposure=2.0))
    r = reading(page, 1, ctrl)
    assert r['measure'] == 'none'
    assert r['message'] == MSG_NO_STARS


def test_stars_visible_before_a_pick(session, source):
    ctrl, page = session
    source.frames.put(raw_frame(GRID, exposure=2.0))
    r = reading(page, 1, ctrl)
    assert r['measure'] == 'stars'
    assert r['stars_visible'] == len(GRID)
    assert r['size'] is None
    assert r['message'] == 'About 16 stars visible · press Pick stars to measure'


def test_measure_switch_needs_several_frames(session, source):
    ctrl, page = session
    source.frames.put(raw_frame((), exposure=2.0))
    assert reading(page, 1, ctrl)['measure'] == 'none'
    for n in range(2, 6):
        source.frames.put(raw_frame((), background=20000.0, exposure=2.0))
        assert reading(page, n, ctrl)['measure'] == 'none'
    source.frames.put(raw_frame((), background=20000.0, exposure=2.0))
    assert reading(page, 6, ctrl)['measure'] == 'contrast'


def test_measure_message_on_switch(session, source):
    ctrl, page = session
    source.frames.put(raw_frame((), exposure=2.0))
    assert page.json('measure')['measure'] == 'none'
    for n in range(2, 7):
        source.frames.put(raw_frame((), background=20000.0, exposure=2.0))
        reading(page, n, ctrl)
    assert page.json('measure')['measure'] == 'contrast'


# --- commands -------------------------------------------------------------------------------------------------------

def test_watcher_cannot_command(session, connect, source):
    ctrl, page = session
    wid, watcher = connect(controls=False)
    calls = list(source.calls)
    ctrl.handle(wid, {'kind': 'set_camera', 'exposure': 8.0, 'gain': 1})
    err = watcher.json('error')
    assert err['command'] == 'set_camera'
    assert err['message']
    assert source.calls == calls
    assert ctrl.exposure == 2.0

    ctrl.handle(wid, {'kind': 'ping', 'id': 9})
    pong = watcher.json('pong')
    assert pong['id'] == 9 and abs(pong['t'] - time.time()) < 5


def test_bad_values_get_an_error(session, source):
    ctrl, page = session
    calls = list(source.calls)
    for msg in ({'kind': 'set_camera', 'exposure': -1, 'gain': 1},
                {'kind': 'set_camera', 'exposure': 'x', 'gain': 1},
                {'kind': 'moved', 'cmd_id': -4},
                {'kind': 'star', 'id': 99, 'state': 'pin'},
                {'kind': 'set_region', 'roi': [1, 2]},
                {'kind': 'no_such_thing'}):
        ctrl.handle(page.cid, msg)
        assert page.json('error')['command'] == msg['kind']
    assert source.calls == calls


def test_set_camera_refused_during_pick(session, source, epoch):
    ctrl, page = session
    ctrl.handle(page.cid, {'kind': 'pick_stars'})
    ctrl.handle(page.cid, {'kind': 'set_camera', 'exposure': 4.0, 'gain': 50})
    assert page.json('error')['message'] == 'Wait for the star pick to finish.'
    assert 'set_camera' not in source.kinds()


def test_set_region_restarts_source_and_clears_stars(session, source, epoch):
    ctrl, page = session
    run_pick(ctrl, page, source, epoch)
    ctrl.handle(page.cid, {'kind': 'set_region', 'roi': [11, 21, 101, 81], 'cmd_id': 4})
    assert page.json('stars')['stars'] == []
    assert page.json('mode')['roi'] == [10, 20, 100, 80]
    assert source.calls[-2:] == [('stop',), ('start', (10, 20, 100, 80), 2.0, 100.0)]


def test_preview_sends_whole_frame(session, source):
    ctrl, page = session
    ctrl.handle(page.cid, {'kind': 'preview'})
    msg = page.json('preview')
    assert (msg['width'], msg['height']) == SHAPE[::-1]
    assert msg['jpeg']


def test_moved_marks_later_frames(session, source, epoch):
    ctrl, page = session
    epoch.t = 1000.0
    ctrl.handle(page.cid, {'kind': 'moved', 'cmd_id': 7})  # settled at 1000.5
    source.frames.put(raw_frame(GRID, t_end=1002.4, exposure=2.0))  # began 1000.4
    head = page.frame()
    assert head['cmd_id'] == 0
    assert reading(page, 1, ctrl)['cmd_id'] == 0
    source.frames.put(raw_frame(GRID, t_end=1002.5, exposure=2.0))  # began 1000.5
    head = page.frame()
    assert head['cmd_id'] == 7
    assert reading(page, 2, ctrl)['cmd_id'] == 7


def test_falls_behind_keeps_newest(connect, make, source):
    cid, page = connect()
    ctrl = make(exposure=2.0)
    for i in range(4):
        source.frames.put(raw_frame(GRID, t_end=1002.0 + i, exposure=2.0))
    ctrl.begin(cid)
    head = page.frame()
    assert head['frame_no'] == 4
    assert head['capture_time'] == pytest.approx(1004.0)
    assert reading(page, 4, ctrl)['frame'] == 4
    later = page.drain(0.3)
    assert not [m for m in later if m[0] == 'frame']
    assert not [m for m in later if m[0] == 'json' and m[1].get('kind') == 'reading']


def test_exposure_search_lengthens_until_stars(connect, make, source):
    cid, page = connect()
    ctrl = make(exposure=1.0, search=ExposureSearch(0.001, 15.0, 1.0))
    ctrl.begin(cid)
    source.frames.put(raw_frame((), exposure=1.0))
    step = page.json('progress', what='exposure_search', pump=ctrl.service)
    assert step['message'] == 'No stars yet. Trying 2 s.' and step['done'] is False
    assert ('set_camera', 2.0, 100.0) in source.calls
    source.frames.put(raw_frame(GRID, exposure=2.0, t_end=1003.0))
    step = page.json('progress', what='exposure_search', pump=ctrl.service)
    assert step['done'] is True and step['message'] == 'Found 16 stars at 2 s.'


def test_done_ends_and_stops_source(session, source):
    ctrl, page = session
    ctrl.handle(page.cid, {'kind': 'done'})
    assert page.json('ended')['reason'] == 'done'
    assert not ctrl.active
    assert source.kinds()[-1] == 'stop'


def test_no_page_ends_session(connect, make, source, link):
    cid, page = connect()
    clock = FakeClock()
    ctrl = FocusController(source, link, clock, RIG, False, epoch=Epoch(), exposure=1.0, gain=1.0)
    ctrl.begin(cid)
    page.client.close()
    assert wait_for(lambda: not link.clients())
    ctrl.service()
    assert ctrl.active
    clock.t = 121.0
    ctrl.service()
    assert not ctrl.active
    assert ctrl.end_reason == 'no_page'


# --- the service ----------------------------------------------------------------------------------------------------

INFO = CameraInfo(interface='indi', device_name='CCD Simulator', driver_has_video=True, rig=RIG)


def test_service_idle_answers_ping_and_state(link, connect):
    service = FocusService(link, INFO, lambda cid: pytest.fail('no session expected'))
    cid, page = connect(controls=False)
    page.client.send({'kind': 'hello'})
    page.client.send({'kind': 'ping', 'id': 1})
    state = page.json('state', pump=service.poll)
    assert state['session'] == 'idle'
    assert state['rig'] == {'page': 'new', 'picture': 'stills', 'reason': 'test rig'}
    assert state['driver_has_video'] is True
    assert state['controls'] is False
    assert page.json('pong', pump=service.poll)['id'] == 1

    page.client.send({'kind': 'set_camera', 'exposure': 1, 'gain': 1})
    assert page.json('error', pump=service.poll)['command'] == 'set_camera'


def test_service_start_from_watcher_refused(link, connect):
    made = []
    service = FocusService(link, INFO, lambda cid: made.append(cid))
    cid, page = connect(controls=False)
    page.client.send({'kind': 'start'})
    assert page.json('error', pump=service.poll)['command'] == 'start'
    assert made == []


def test_service_start_returns_true_once(link, connect, source, closers):
    made = []

    def make_controller(cid):
        ctrl = FocusController(source, link, FakeClock(), RIG, False, epoch=Epoch(), exposure=1.0, gain=1.0)
        closers.append(_Ender(ctrl))
        made.append(ctrl)
        return ctrl

    service = FocusService(link, INFO, make_controller)
    cid, page = connect()
    page.client.send({'kind': 'start'})
    page.client.send({'kind': 'start'})
    results = []
    page.json('mode', pump=lambda: results.append(service.poll()))
    wait_for(lambda: results.append(service.poll()) or len(results) > 20)
    assert results.count(True) == 1
    assert len(made) == 1 and made[0].active
    assert ('start', None, 1.0, 1.0) in source.calls

    page.client.send({'kind': 'done'})
    ended = page.json('ended', pump=service.poll)
    assert ended['reason'] == 'done'
    idle = page.json('state', pump=service.poll, session='idle', idle_reason='done')
    assert idle['controls'] is True and not service.active


def test_source_that_fails_to_start_ends_the_session(connect, make, source):
    cid, page = connect()

    def broken(roi, exposure, gain):
        raise RuntimeError('no camera')

    source.start = broken
    ctrl = make()
    ctrl.begin(cid)
    assert not ctrl.active
    assert ctrl.end_reason == 'failed'
    assert page.json('ended')['reason'] == 'failed'
