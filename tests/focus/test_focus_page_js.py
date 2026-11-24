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
