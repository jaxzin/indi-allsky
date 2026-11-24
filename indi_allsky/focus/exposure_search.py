"""Exposure search (spec 6.2): double until stars show, halve while over-exposed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FOUND_STARS = 5

# Starting thresholds, as fractions of full scale. Tune on real scenes (spec 11, 12).
OVER_FRACTION = 0.80
BRIGHT_FRACTION = 0.15

MAX_CEILING_S = 15.0

MSG_NO_STARS = 'No stars to measure. Try a longer exposure.'
MSG_SHORTEST_OVER = 'The camera\'s shortest exposure is still over-exposed.'
MSG_BRIGHT = 'The picture is bright enough to focus on picture contrast.'


@dataclass(frozen=True)
class SearchStep:
    exposure: float | None  # the exposure to use next; on done, the one the search settled on
    done: bool
    message: str


def brightness(gray: np.ndarray, full_scale: float) -> str:
    """Return 'over', 'bright' or 'dark' from the median level of the picture."""
    level = float(np.median(gray)) / float(full_scale)

    if level >= OVER_FRACTION:
        return 'over'

    if level >= BRIGHT_FRACTION:
        return 'bright'

    return 'dark'


def default_ceiling(ccd_exposure_max: float | None) -> float:
    """CCD_EXPOSURE_MAX from config, or 15 s, whichever is lower."""
    if ccd_exposure_max is None:
        return MAX_CEILING_S

    return min(float(ccd_exposure_max), MAX_CEILING_S)


class ExposureSearch(object):

    def __init__(self, shortest: float, ceiling: float, start: float) -> None:
        self.shortest = float(shortest)
        self.ceiling = float(ceiling)
        self.exposure = float(start)
        self._final: SearchStep | None = None


    def _finish(self, message: str) -> SearchStep:
        self._final = SearchStep(self.exposure, True, message)
        return self._final


    def step(self, star_count: int, level: str) -> SearchStep:
        if self._final is not None:
            return self._final

        if level == 'over':
            if self.exposure <= self.shortest:
                self.exposure = self.shortest
                return self._finish(MSG_SHORTEST_OVER)

            self.exposure = max(self.exposure / 2, self.shortest)
            return SearchStep(
                self.exposure,
                False,
                'Over-exposed. Trying {0:g} s.'.format(self.exposure),
            )

        if star_count >= FOUND_STARS:
            return self._finish('Found {0:d} stars at {1:g} s.'.format(star_count, self.exposure))

        if level == 'bright':
            return self._finish(MSG_BRIGHT)

        if self.exposure >= self.ceiling:
            self.exposure = self.ceiling
            return self._finish(MSG_NO_STARS)

        self.exposure = min(self.exposure * 2, self.ceiling)
        return SearchStep(
            self.exposure,
            False,
            'No stars yet. Trying {0:g} s.'.format(self.exposure),
        )
