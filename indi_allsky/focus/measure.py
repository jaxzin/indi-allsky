"""Full-resolution star measuring (spec amendment 12), ported from the focus prototype.

Per frame: find hot pixels on the RAW mosaic (once per stream) and fix them, demosaic to gray, take the noise, then
measure each picked star in a small box. Picking runs `detect` on a stack of longer frames.

Every threshold is a keyword argument whose default is the prototype's value.
"""

from __future__ import annotations

import functools
import logging
from typing import NamedTuple
from typing import Optional

import cv2
import numpy as np


logger = logging.getLogger('indi_allsky')


DEFAULT_MIN_SNR = 8.0     # single-frame S/N a picked star needs to be kept (the prototype's --min-snr)
TRACK_SNR = 4.0           # peak S/N a tracked star needs in a frame (--track-snr)
MAX_SHIFT = 6.0           # px a tracked star may move between frames (--max-shift)

HOT_SIGMA = 6.0
HOT_NEIGHBOUR_FRACTION = 0.25

DETECT_SIGMA = 6.0
BACKGROUND_SIGMA_PX = 8.0
SMOOTH_SIGMA_PX = 1.2

PICK_MAX_AREA = 900
PICK_EDGE_PX = 24
PICK_SAME_STAR_PX = 6.0

MEASURE_BOX = 25
MEASURE_CLIP_SIGMA = 2.0
PEAK_HALF = 3             # the peak S/N is taken in the central (2 * 3 + 1) = 7x7
APERTURE_PASSES = 3
APERTURE_SCALE = 2.5
MIN_APERTURE_PX = 3.0

_BAYER_TO_GRAY = {
    'RGGB': cv2.COLOR_BayerRG2GRAY,
    'GRBG': cv2.COLOR_BayerGR2GRAY,
    'GBRG': cv2.COLOR_BayerGB2GRAY,
    'BGGR': cv2.COLOR_BayerBG2GRAY,
}

_NEIGHBOURS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)


class Candidate(NamedTuple):
    """A source found by `detect`. `flux` is the smoothed, background-subtracted value at the centroid, in units of
    that smoothed image's noise (the prototype's ranking value)."""
    x: float
    y: float
    flux: float
    area: int


class StarReading(NamedTuple):
    """One star in one frame: centroid (px), size (2 x brightness-weighted mean radius, full-res px) and peak S/N."""
    x: float
    y: float
    size: float
    snr: float


# --- hot pixels -------------------------------------------------------------------------------------------------

def find_hot_pixels(
    raw: np.ndarray,
    step: int,
    sigma: float = HOT_SIGMA,
    neighbour_fraction: float = HOT_NEIGHBOUR_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """(ys, xs) of hot pixels on the RAW mosaic. `step` is the distance to the nearest same-colour pixel: 2 for
    Bayer, 1 for mono.

    A pixel is hot when it exceeds the same-colour 3x3 median by more than `sigma` x the noise while the largest
    excess among its 8 immediate neighbours (any colour) is under `neighbour_fraction` of its own: a real star,
    however sharp, spills some light into them. Find them on the RAW mosaic, before demosaicing smears them.
    Hot pixels sit still, so this runs once per stream.
    """
    r = raw.astype(np.float32)
    ex = np.empty_like(r)  # excess over the 3x3 median of the same colour
    for dy in range(step):
        for dx in range(step):
            plane = np.ascontiguousarray(r[dy::step, dx::step])
            ex[dy::step, dx::step] = plane - cv2.medianBlur(plane, 3)
    sig = noise_sigma(ex)
    nb = cv2.dilate(ex, _NEIGHBOURS)
    ys, xs = np.nonzero((ex > sigma * sig) & (nb < neighbour_fraction * ex))
    h, w = r.shape
    keep = (ys >= step) & (xs >= step) & (ys < h - step) & (xs < w - step)
    return ys[keep], xs[keep]


def fix_hot_pixels(raw: np.ndarray, hot: tuple[np.ndarray, np.ndarray], step: int) -> np.ndarray:
    """A float32 copy of `raw` with each hot pixel replaced by the median of its 8 same-colour neighbours."""
    r = raw.astype(np.float32)
    ys, xs = hot
    if len(ys):
        nbs = np.stack([r[ys + dy, xs + dx] for dy in (-step, 0, step) for dx in (-step, 0, step) if dy or dx])
        r[ys, xs] = np.median(nbs, axis=0)
    return r


# --- gray and noise ---------------------------------------------------------------------------------------------

def to_gray(raw: np.ndarray, bayer: Optional[str]) -> np.ndarray:
    """Full-resolution float32 luminance. `bayer` is the CFA pattern ('RGGB', ...) or None for mono.

    Not the 'drop the brightest of 2x2' trick: that throws away the one pixel a sharp star lands on.
    """
    if bayer is None:
        return raw.astype(np.float32)
    try:
        code = _BAYER_TO_GRAY[bayer]
    except KeyError:
        raise ValueError('unknown bayer pattern: %r' % (bayer,)) from None
    mosaic = np.clip(raw, 0, 65535).astype(np.uint16)
    return cv2.cvtColor(mosaic, code).astype(np.float32)


def noise_sigma(gray: np.ndarray) -> float:
    """1.4826 x the median absolute deviation over every 4th pixel."""
    s = gray[::4, ::4]
    return 1.4826 * float(np.median(np.abs(s - np.median(s)))) + 1e-6


# --- detection and picking --------------------------------------------------------------------------------------

def detect(
    gray: np.ndarray,
    *,
    sigma: float = DETECT_SIGMA,
    background_px: float = BACKGROUND_SIGMA_PX,
    smooth_px: float = SMOOTH_SIGMA_PX,
) -> list[Candidate]:
    """Compact sources: subtract a Gaussian background, smooth, threshold, find connected components.

    `sigma` is the threshold in units of the noise of the smoothed, background-subtracted image, which is measured
    here (over every pixel, as the prototype did); it is not the gray image's noise. Keyword-only so that a noise
    value passed by position fails loudly.
    """
    g = gray.astype(np.float32, copy=False)
    hp = g - cv2.GaussianBlur(g, (0, 0), background_px)
    sm = cv2.GaussianBlur(hp, (0, 0), smooth_px)
    sig = 1.4826 * float(np.median(np.abs(sm - np.median(sm)))) + 1e-6
    n, _, stats, cent = cv2.connectedComponentsWithStats((sm > sigma * sig).astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        cx, cy = float(cent[i][0]), float(cent[i][1])
        out.append(Candidate(cx, cy, float(sm[int(cy), int(cx)] / sig), int(stats[i, cv2.CC_STAT_AREA])))
    return out


def pick(
    stack: np.ndarray,
    *,
    sigma: float = DETECT_SIGMA,
    max_area: int = PICK_MAX_AREA,
    edge: int = PICK_EDGE_PX,
    same_star: float = PICK_SAME_STAR_PX,
) -> list[Candidate]:
    """Stars in a stacked frame, brightest first.

    Rejects components over `max_area` px (a glint or a blob), anything within `edge` px of the frame edge, and
    anything within `same_star` px of a brighter kept candidate (the same star split in two, not a neighbour).
    `sigma` is as in `detect`.
    """
    h, w = stack.shape[:2]
    cands = [
        c for c in detect(stack, sigma=sigma)
        if c.area <= max_area and edge <= c.x < w - edge and edge <= c.y < h - edge
    ]
    cands.sort(key=lambda c: (c.flux, c.x, c.y, c.area), reverse=True)
    out: list[Candidate] = []
    for c in cands:
        if not any((c.x - o.x) ** 2 + (c.y - o.y) ** 2 <= same_star ** 2 for o in out):
            out.append(c)
    return out


# --- per-star measuring -----------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def _grid(half: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = (a.astype(np.float32) for a in np.mgrid[0:2 * half + 1, 0:2 * half + 1])
    yy.setflags(write=False)
    xx.setflags(write=False)
    return yy, xx


def measure_star(
    gray: np.ndarray,
    x: float,
    y: float,
    sigma: float,
    box: int = MEASURE_BOX,
    *,
    clip_sigma: float = MEASURE_CLIP_SIGMA,
    peak_half: int = PEAK_HALF,
    passes: int = APERTURE_PASSES,
    aperture_scale: float = APERTURE_SCALE,
    min_aperture: float = MIN_APERTURE_PX,
) -> Optional[StarReading]:
    """Centroid, size and peak S/N of the star in a `box` x `box` window at (x, y). `sigma` is the gray noise.

    None when the box leaves the frame or holds no light above the clip. No S/N gate here: `Tracker` applies one.
    """
    half = box // 2
    xi, yi = int(round(x)), int(round(y))
    if not (half <= xi < gray.shape[1] - half and half <= yi < gray.shape[0] - half):
        return None
    b = gray[yi - half:yi + half + 1, xi - half:xi + half + 1]
    edge = np.concatenate([b[0], b[-1], b[:, 0], b[:, -1]])
    bg = float(np.median(edge))
    w = np.clip(b - bg - clip_sigma * sigma, 0, None)
    f = float(w.sum())
    # peak near where the star should be, not anywhere in the box: the brightest of ~600 noise pixels alone is
    # ~3 sigma, which let faint stars 'pass' on noise and measure the noise as a big blur
    c0, c1 = half - peak_half, half + peak_half + 1
    peak = float((b[c0:c1, c0:c1].max() - bg) / sigma)
    if f <= 0:
        return None
    yy, xx = _grid(half)
    bx, by = float((w * xx).sum() / f), float((w * yy).sum() / f)
    # Shrink the aperture to fit the star: over the whole box, noise far from a faint sharp star outweighs it and
    # makes it look bigger than a bright soft one.
    ap = float(half)
    size = 0.0
    for _ in range(passes):
        r = np.hypot(xx - bx, yy - by)
        wa = np.where(r <= ap, w, 0.0)
        fa = float(wa.sum())
        if fa <= 0:
            return None
        bx, by = float((wa * xx).sum() / fa), float((wa * yy).sum() / fa)
        size = 2.0 * float((wa * np.hypot(xx - bx, yy - by)).sum() / fa)
        ap = min(float(half), max(min_aperture, aperture_scale * size))
    return StarReading(xi - half + bx, yi - half + by, size, peak)


# --- tracking ---------------------------------------------------------------------------------------------------

class Tracker:
    """Follows picked stars frame to frame, each in a box around where it was last seen.

    A star counts in a frame when its peak S/N is at least `track_snr` and it moved no more than `max_shift` px.
    Stars not found move by the median shift of those that were, so the sky's rotation is followed.
    """

    def __init__(
        self,
        stars: list[Candidate],
        track_snr: float = TRACK_SNR,
        max_shift: float = MAX_SHIFT,
        box: int = MEASURE_BOX,
    ) -> None:
        self.track_snr = track_snr
        self.max_shift = max_shift
        self.box = box
        self._pos = [(float(s.x), float(s.y)) for s in stars]

    @property
    def positions(self) -> list[tuple[float, float]]:
        """Where each star is expected next, in the order of the initial star list."""
        return list(self._pos)

    def update(self, gray: np.ndarray, sigma: float) -> dict[int, StarReading]:
        """Readings for the stars found in this frame, keyed by index into the initial star list."""
        got: dict[int, StarReading] = {}
        for i, (x, y) in enumerate(self._pos):
            m = measure_star(gray, x, y, sigma, self.box)
            if m is None or m.snr < self.track_snr:
                continue
            if (m.x - x) ** 2 + (m.y - y) ** 2 <= self.max_shift ** 2:
                got[i] = m
        if got:
            dx = float(np.median([m.x - self._pos[i][0] for i, m in got.items()]))
            dy = float(np.median([m.y - self._pos[i][1] for i, m in got.items()]))
            self._pos = [(got[i].x, got[i].y) if i in got else (x + dx, y + dy)
                         for i, (x, y) in enumerate(self._pos)]
        return got


# --- contrast and grading ---------------------------------------------------------------------------------------

def contrast_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian: the measure the focus page uses today (views.py `JsonFocusView`)."""
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def grade(snr1: float, min_snr: float = DEFAULT_MIN_SNR) -> str:
    """'weak' when a picked star's single-frame S/N (median over the grading frames) is under `min_snr`, else
    'kept'. A faint star measured at the tracking exposure is mostly noise, and its size does not follow focus."""
    return 'weak' if snr1 < min_snr else 'kept'
