from __future__ import annotations

import cv2
import numpy as np
import pytest

from indi_allsky.focus import frame
from indi_allsky.focus.frame import HEADER
from indi_allsky.focus.frame import Stretch


def test_decode_u16_stream():
    rng = np.random.default_rng(1)
    src = rng.integers(0, 65535, (6, 8), dtype=np.uint16)
    img, full = frame.decode_stream(src.astype('<u2').tobytes(), 8, 6, '')
    assert full == 65535.0
    assert img.dtype == np.uint16
    assert np.array_equal(img, src)


def test_decode_8bit_and_rgb24_streams():
    src = np.arange(48, dtype=np.uint8).reshape(6, 8)
    img, full = frame.decode_stream(src.tobytes(), 8, 6, '')
    assert full == 255.0
    assert np.array_equal(img, src)

    rgb = np.dstack([src, src, src])
    img, full = frame.decode_stream(rgb.tobytes(), 8, 6, '')
    assert full == 255.0
    assert img.shape == (6, 8)
    assert np.array_equal(img, src)


def test_decode_jpeg_stream():
    src = np.full((32, 48), 120, np.uint8)
    ok, enc = cv2.imencode('.jpg', src)
    assert ok
    img, full = frame.decode_stream(enc.tobytes(), 48, 32, 'image/jpeg')
    assert full == 255.0
    assert img.shape == (32, 48)
    assert abs(int(img.mean()) - 120) <= 2


def test_decode_wrong_length_returns_none():
    img, full = frame.decode_stream(b'\x00' * 13, 8, 6, '')
    assert img is None
    assert full is None


def test_decode_corrupt_jpeg_returns_none():
    img, full = frame.decode_stream(b'\xff\xd8garbage', 8, 6, '')
    assert img is None
    assert full is None


def test_maxpool_keeps_one_pixel_star_brightness():
    g = np.full((64, 64), 100, np.uint16)
    g[31, 17] = 1000
    # a plain 16-bit -> 8-bit scale would put 1000 at 3; stretched to [100, 1000] it must be 255
    out = frame.preview(g, Stretch(black=100.0, white=1000.0, gamma=1.0))
    assert out.shape == (32, 32)
    assert out.dtype == np.uint8
    assert out.max() == 255
    assert (out == 255).sum() == 1
    assert out[15, 8] == 255


def test_maxpool_drops_odd_trailing_row_and_column():
    g = np.zeros((5, 7), np.uint8)
    assert frame.preview(g, None).shape == (2, 3)


def test_preview_linear_scales_by_dtype_range():
    g16 = np.array([[0, 65535], [257, 257 * 100]], np.uint16)
    out = frame.preview(g16, None)
    assert out.shape == (1, 1)
    assert out[0, 0] == 255
    g8 = np.array([[10, 200], [30, 40]], np.uint8)
    assert frame.preview(g8, None)[0, 0] == 200


def test_stretch_from_frame_levels():
    rng = np.random.default_rng(2)
    g = rng.normal(1000, 20, (200, 200)).clip(0, 65535).astype(np.uint16)
    g[100, 100] = 60000
    s = Stretch.from_frame(g)
    assert s.gamma == 0.8
    assert abs(s.black - np.percentile(g, 5)) < 1e-6
    # the 99.995th percentile of 40000 px sits below the single hot pixel, so median + 60 sigma (~2200) wins
    assert 2000 < s.white < 2500
    out = frame.preview(g, s)
    assert out.max() == 255


def test_clock_height_is_one_fifteenth():
    h, w = 480, 640
    img = np.zeros((h, w), np.uint8)
    frame.draw_clock(img, 1700000000.123, 42)
    rows = np.where(img.any(axis=1))[0]
    cols = np.where(img.any(axis=0))[0]
    assert rows.size
    assert rows.max() - rows.min() + 1 <= h // 15 + 4
    assert rows.min() > h - 2 * (h // 15)    # bottom
    assert cols.min() < 10                   # left


def test_clock_draws_on_colour_image_in_place():
    img = np.full((300, 400, 3), 90, np.uint8)
    frame.draw_clock(img, 0.0, 1)
    assert img.max() == 255
    assert (img == 0).any()


def test_clock_text_is_utc():
    # 1700000000 is 2023-11-14 22:13:20 UTC
    assert frame.clock_text(1700000000.5, 7) == '22:13:20.500 #7'


def test_encode_jpeg_is_valid_jpeg_gray_and_colour():
    for img in (np.full((40, 60), 128, np.uint8), np.full((40, 60, 3), 128, np.uint8)):
        data = frame.encode_jpeg(img)
        assert data[:2] == b'\xff\xd8'
        back = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        assert back.shape[:2] == (40, 60)


def test_encode_jpeg_simplejpeg_path(monkeypatch):
    calls = {}

    class Fake:
        @staticmethod
        def encode_jpeg(img, quality, colorspace):
            calls['args'] = (img.shape, quality, colorspace)
            return b'\xff\xd8fake'

    monkeypatch.setattr(frame, 'simplejpeg', Fake)
    assert frame.encode_jpeg(np.zeros((4, 6), np.uint8)) == b'\xff\xd8fake'
    assert calls['args'] == ((4, 6, 1), 85, 'GRAY')


def test_header_round_trip():
    assert HEADER.size == 29
    jpeg = b'\xff\xd8abc\xff\xd9'
    buf = frame.pack_frame(jpeg, 9, 1700000000.25, 0.5, 120.0, 77)
    assert buf[:4] == b'IAFF'
    assert buf[HEADER.size:] == jpeg
    hdr = frame.unpack_header(buf)
    assert hdr == {
        'magic': b'IAFF', 'version': 1, 'frame_no': 9, 'capture_time': 1700000000.25,
        'exposure': 0.5, 'gain': 120.0, 'cmd_id': 77,
    }


def test_unpack_header_rejects_bad_magic():
    buf = bytearray(frame.pack_frame(b'x', 1, 0.0, 1.0, 1.0, 0))
    buf[0:4] = b'NOPE'
    with pytest.raises(ValueError):
        frame.unpack_header(bytes(buf))


def test_unpack_header_rejects_bad_version_and_short_buffer():
    buf = bytearray(frame.pack_frame(b'x', 1, 0.0, 1.0, 1.0, 0))
    buf[4] = 2
    with pytest.raises(ValueError):
        frame.unpack_header(bytes(buf))
    with pytest.raises(ValueError):
        frame.unpack_header(b'IAFF\x01')
