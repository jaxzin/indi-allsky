"""The live Focus page's pure helpers (`focus_session.js`), run with node.

Pure tier. Skipped where node is not installed. The frame header is built with the same `struct` format as
`indi_allsky/focus/frame.py`, so the page and capture cannot drift apart unnoticed.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from indi_allsky.focus import frame


NODE = shutil.which('node')
SCRIPT = Path(__file__).resolve().parents[2] / 'indi_allsky' / 'flask' / 'static' / 'js' / 'focus_session.js'

pytestmark = pytest.mark.skipif(NODE is None, reason='node is not installed')


def run(body):
    """Run `body` with the helpers as `h`; it must print one JSON value."""
    code = 'const h = require({0:s});\n{1:s}\n'.format(json.dumps(str(SCRIPT)), body)
    out = subprocess.run([NODE, '-e', code], capture_output=True, text=True, timeout=30, check=True)
    return json.loads(out.stdout)


def test_parse_header_matches_capture():
    header = struct.pack('>4sBIdffI', b'IAFF', 1, 4242, 1764000000.125, 0.25, 300.0, 77)
    assert len(header) == frame.HEADER.size
    buf = list(header + b'\xff\xd8jpeg')

    got = run('const b = new Uint8Array({0:s}).buffer; console.log(JSON.stringify(h.parseHeader(b)));'.format(
        json.dumps(buf)))

    assert got == {'frame_no': 4242, 'capture_time': 1764000000.125, 'exposure': 0.25, 'gain': 300.0, 'cmd_id': 77}


def test_parse_header_refuses_other_bytes():
    bad_magic = list(struct.pack('>4sBIdffI', b'XXXX', 1, 1, 0.0, 0.0, 0.0, 0))
    bad_version = list(struct.pack('>4sBIdffI', b'IAFF', 2, 1, 0.0, 0.0, 0.0, 0))

    got = run('console.log(JSON.stringify([{0:s}, {1:s}, [1, 2, 3]].map('
              'a => h.parseHeader(new Uint8Array(a).buffer))));'.format(json.dumps(bad_magic), json.dumps(bad_version)))

    assert got == [None, None, None]


def test_clock_offset_uses_the_shortest_round_trip():
    got = run('const s = [h.clockSample(10, 10.4, 1000.5), h.clockSample(20, 20.02, 1010.01),'
              ' h.clockSample(30, 30.3, 1020.0)]; console.log(JSON.stringify(h.bestOffset(s)));')

    assert got == pytest.approx(990.0)


def test_status_state():
    got = run('console.log(JSON.stringify([h.statusState(null, 0.1, false), h.statusState(0.05, 0.1, false),'
              ' h.statusState(0.05, 0.1, true), h.statusState(0.9, 0.1, false), h.statusState(1.1, 0.1, false),'
              ' h.statusState(6.1, 2, true)]));')

    assert got == ['waiting', 'live', 'waiting', 'live', 'stalled', 'stalled']


def test_ended_text_maps_codes_and_keeps_sentences():
    sentence = 'The live focus page cannot reach the capture program.'
    got = run('console.log(JSON.stringify(["done", "slow", {0:s}, null].map(h.endedText)));'.format(
        json.dumps(sentence)))

    assert got[0].startswith('The focus session ended')
    assert 'keep up' in got[1]
    assert got[2] == sentence
    assert got[3] == ''


def test_nearest_ring_within_screen_pixels():
    rings = [{'id': 0, 'x': 100, 'y': 100}, {'id': 1, 'x': 130, 'y': 100}]
    got = run('const r = {0:s}; console.log(JSON.stringify([h.nearestRing(r, 104, 100, 1, 12),'
              ' h.nearestRing(r, 120, 100, 1, 12), h.nearestRing(r, 115, 100, 2, 12)]));'.format(json.dumps(rings)))

    assert got == [0, 1, None]


def test_next_cmd_id_rises_and_wraps():
    got = run('console.log(JSON.stringify([h.nextCmdId(0, 0), h.nextCmdId(5, 9), h.nextCmdId(12, 3),'
              ' h.nextCmdId(4294967295, 0)]));')

    assert got == [1, 10, 13, 1]


def test_words():
    got = run('console.log(JSON.stringify([h.fmtAge(0.214), h.fmtAge(3.42), h.fmtAge(12.6), h.liveText("video", 0.1),'
              ' h.liveText("stills", 2), h.fmtExposure(0.075), h.fmtGain(300)]));')

    assert got == ['0.21 s old', '3.4 s old', '13 s old', 'Live video · 10 pictures a second',
                   'Still pictures · 2 s each', '75 ms', '300']


def test_live_rate_is_what_arrives_not_one_over_exposure():
    got = run('console.log(JSON.stringify([h.liveText("video", 0.001, 0.1), h.liveText("video", 0.5, 0.1),'
              ' h.liveText("video", null, null), h.liveText("stills", null, null)]));')

    assert got == ['Live video · 10 pictures a second', 'Live video · 2 pictures a second', 'Live video',
                   'Still pictures']


def test_pending_commands_add_and_clear_on_the_picture_that_shows_them():
    got = run('let p = []; p = h.pendingAdd(p, 5, "camera"); p = h.pendingAdd(p, 6, "motor");'
              ' p = h.pendingAdd(p, 8, "region");'
              ' console.log(JSON.stringify([p.length, h.pendingShown(p, 4).length, h.pendingShown(p, 6),'
              ' h.pendingShown(p, 9)]));')

    assert got == [3, 3, [{'id': 8, 'kind': 'region'}], []]


def test_pending_drop_when_the_command_never_left():
    got = run('const p = [{id: 5, kind: "camera"}, {id: 6, kind: "motor"}];'
              ' console.log(JSON.stringify([h.pendingDrop(p, 6), h.pendingDrop(p, 99)]));')

    assert got == [[{'id': 5, 'kind': 'camera'}], [{'id': 5, 'kind': 'camera'}, {'id': 6, 'kind': 'motor'}]]


def test_pending_refused_by_id_or_newest_of_its_kind():
    got = run('const p = [{id: 5, kind: "camera"}, {id: 6, kind: "motor"}, {id: 7, kind: "camera"}];'
              ' console.log(JSON.stringify([h.pendingRefused(p, "set_camera", 5).map(x => x.id),'
              ' h.pendingRefused(p, "set_camera", null).map(x => x.id),'
              ' h.pendingRefused(p, "moved", undefined).map(x => x.id),'
              ' h.pendingRefused(p, "start", null).map(x => x.id)]));')

    assert got == [[6, 7], [5, 6], [5, 7], [5, 6, 7]]


def test_exposure_choices_stay_inside_the_camera_limits():
    got = run('const L = [0.001, 0.01, 0.1, 1, 5, 10];'
              ' console.log(JSON.stringify([h.exposureChoices(L, 0.1, 0.005, 2),'
              ' h.exposureChoices(L, 0.1, null, null)]));')

    assert got == [[0.005, 0.01, 0.1, 1, 2], [0.001, 0.01, 0.1, 1, 5, 10]]


def test_gain_range_from_the_camera_with_a_fine_step_for_small_ranges():
    got = run('console.log(JSON.stringify([h.gainRange(0, 10, 2), h.gainRange(0, 510, 7),'
              ' h.gainRange(null, null, 100)]));')

    assert got[0] == {'min': 0, 'max': 10, 'step': pytest.approx(0.1)}
    assert got[1] == {'min': 0, 'max': 510, 'step': 1}
    assert got[2] == {'min': 0, 'max': 600, 'step': 1}


def test_gain_range_for_a_fixed_gain_camera_is_fixed():
    got = run('console.log(JSON.stringify([h.gainRange(1, 1, 1), h.gainRange(0, 0, 0), h.gainRange(0, 10, 2).fixed,'
              ' h.gainRange(null, null, 0).fixed]));')

    assert got[0] == {'min': 1, 'max': 1, 'step': 1, 'fixed': True}
    assert got[1] == {'min': 0, 'max': 0, 'step': 1, 'fixed': True}
    assert got[2:] == [None, None]


def test_gain_sent_stays_inside_the_camera_limits():
    got = run('const g = (lo, hi, cur, v) => h.clampGain(h.gainRange(lo, hi, cur), v);'
              ' console.log(JSON.stringify([g(1, 1, 100, 100), g(0, 300, 500, 500), g(0, 300, 500, 120),'
              ' g(null, null, 100, 100), g(null, null, 700, 700), h.gainRange(0, 300, 500).max]));')

    assert got == [1, 300, 120, 100, 700, 300]


def test_region_can_move_in_both_modes_even_without_a_region_yet():
    got = run('console.log(JSON.stringify([h.canMoveRegion({picture: "stills", roi: null}),'
              ' h.canMoveRegion({picture: "video", roi: [0, 0, 1024, 1024]}),'
              ' h.canMoveRegion({picture: "video", roi: null}), h.canMoveRegion(null),'
              ' h.canMoveRegion({picture: "classic", roi: null})]));')

    assert got == [True, True, True, False, False]


def test_fractional_gain_reads_sensibly():
    got = run('console.log(JSON.stringify([h.fmtGain(1.25), h.fmtGain(1.2), h.fmtGain(12.34)]));')

    assert got == ['1.25', '1.2', '12.3']


def test_first_picture_is_not_stalled_before_its_download():
    got = run('console.log(JSON.stringify([h.expectedGap(2, []), h.expectedGap(0.1, [0.5, 0.2, 0.3]),'
              ' h.expectedGap(2, [0.5])]));')

    assert got == [4, 0.3, 2]


def test_region_box_on_a_whole_frame_picture_is_centred_and_on_the_bayer_grid():
    got = run('console.log(JSON.stringify([h.regionBox(3000, 2000, null), h.regionBox(801, 601, null),'
              ' h.regionBox(3000, 2000, [2500, 100, 1024, 1024])]));')

    assert got[0] == {'x': 988, 'y': 488, 'w': 1024, 'h': 1024}
    assert got[1] == {'x': 100, 'y': 0, 'w': 600, 'h': 600}
    assert got[2] == {'x': 1976, 'y': 100, 'w': 1024, 'h': 1024}


def test_star_states_in_words():
    got = run('console.log(JSON.stringify(["kept", "weak", "dropped", "pinned", "saturated", "odd"]'
              '.map(h.starWord)));')

    assert got == ['measured', 'too faint', 'left out', 'the only one measured', 'too bright', 'odd']
