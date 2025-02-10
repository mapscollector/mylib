"""Module for interpolation methods."""

from abc import abstractmethod
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import scipy as sp


class InterpolationType(IntEnum):
    """Interpolation type."""

    LINEAR = 0
    CUBIC_SPLINE = 1


@dataclass
class Interpolator:
    """Base class for interpolators."""

    x: float
    y: float

    @abstractmethod
    def __call__(self, x: float) -> float:
        pass

    def interpolate(self, x: float) -> float:
        """Execute interpolation."""
        return self(x)


class LinearInterpolator(Interpolator):
    """Linear interpolator."""

    def __call__(self, x: float) -> float:
        return np.interp(x, self.x, self.y)


class CubicSplineInterpolator(Interpolator):
    """Cubic spline interpolator (based on scipy)."""

    def __init__(self, x: float, y: float) -> float:
        super().__init__(x, y)
        self.tck = sp.interpolate.splrep(self.x, self.y, s=0)

    def __call__(self, x: float) -> float:
        return sp.interpolate.splev(x, self.tck, der=0)


class SplineInterpolator(Interpolator):
    """Base class for spline interpolation not covered in scipy."""

    def __init__(self, x, y):
        super().__init__(x, y)
        self.params = np.array(self.fit(x, y))

    def __call__(self, x):
        idx = self.index(x)
        xx = self.powers(x - self.x[idx])
        theta = self.params[idx]

        return np.multiply(theta, xx).sum(axis=1)

    @classmethod
    @abstractmethod
    def fit(cls, x: float, y: float) -> None:
        """Fit spline parameters."""

    def index(self, t: float) -> list[int]:
        """Return the index idx in the grid such that t >= x[idx] for t in (x[0], x[-1])."""

        idx = [0] * len(t)
        t = np.minimum(t, self.x[-1])

        for i, val in enumerate(t):
            idx[i] = next(x for x, v in enumerate(self.x) if v >= val) - 1

        return np.maximum(idx, 0)

    @classmethod
    def powers(cls, dx):
        """Returns array of ones and three consecutive powers of x."""

        x = np.array([np.ones(dx.shape), dx, np.power(dx, 2), np.power(dx, 3)])

        return x.transpose()


# pylint: disable=too-few-public-methods


class InterpolatorFactory:
    """Factory of interpolators."""

    FACTORY = {
        InterpolationType.LINEAR: LinearInterpolator,
        InterpolationType.CUBIC_SPLINE: CubicSplineInterpolator,
    }

    @classmethod
    def get(cls, x, y, method: InterpolationType = InterpolationType.LINEAR):
        """Return interpolator based on the interpolation type."""
        return cls.FACTORY[method](x, y)


# pylint: enable=too-few-public-methods
