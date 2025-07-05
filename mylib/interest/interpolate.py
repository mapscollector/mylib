"""Interest rate curve interpolation."""

from enum import IntEnum

import numpy as np

from mylib.interest.convert import ConversionType, ConverterFactory
from mylib.interest.rates import discount_factor, implied_rate
from mylib.math.interp import InterpolationType, InterpolatorFactory


# pylint: disable=too-few-public-methods

class InterestRateCurveInterpolator:
    """Base class of interest rate curves interpolators.

    Interest rate interpolation combines interpolation and conversion.

    Converter converts discount factors to interpolated values.
    Interpolator perform interpolation over interpolated values according to the method
    defined in the interpolator.
    Once interpolation is done converter reverts its result back to discount factor.
    """

    def __init__(self, t, df, interp_type, conversion_type):
        self.converter_ = ConverterFactory.get(conversion_type)
        self.interp_ = InterpolatorFactory.get(
            t, 
            self.converter_.convert(t, df), 
            interp_type
        )

    def __call__(self, t):
        """Interpolate discount factors.

        Interpolates iwthin the range of [min(t), max(t)].
        Outside the range of [min(t), max(t)] extrapolates flat rates.
        """

        t = np.array(t)
        min_t = np.min(self.interp_.x)
        max_t = np.max(self.interp_.x)
        t_int = np.minimum(np.maximum(t, min_t), max_t)

        y = self.interp_(t_int)
        df = self.converter_.revert(t_int, y)
        rate = implied_rate(t_int, df)

        return discount_factor(t, rate)


class LogDiscountCurveInterpolator(InterestRateCurveInterpolator):
    """Linear interpolation of log-discount factors."""

    def __init__(self, t, df):
        super().__init__(t, df, InterpolationType.LINEAR, ConversionType.LOGARITHMIC)


class LinearRatesCurveInterpolator(InterestRateCurveInterpolator):
    """Linear interpolation of zero rates."""

    def __init__(self, t, df):
        super().__init__(t, df, InterpolationType.LINEAR, ConversionType.ZERO_RATES)


class LinearDiscountCurveInterpolator(InterestRateCurveInterpolator):
    """Linear interpolation of discount factors."""

    def __init__(self, t, df):
        super().__init__(t, df, InterpolationType.LINEAR, ConversionType.IDENTITY)


class CubicSplineRatesCurveInterpolator(InterestRateCurveInterpolator):
    """Cubic spline interpolation of zero rates."""

    def __init__(self, t, df):
        super().__init__(
            t, df, InterpolationType.CUBIC_SPLINE, ConversionType.ZERO_RATES
        )


class CurveInterpolationType(IntEnum):
    """Enumerator for interest rate curve interpolation methods."""

    LOG_DISCOUNT_INTERPOLATION = 0  # This method is known under several names.
    RAW_INTERPOLATION = 0           # Log-discount interpolation = raw interpolation.
    LINEAR_RATES_INTERPOLATION = 1
    LINEAR_DISCOUNT_INTERPOLATION = 2
    CUBIC_SPLINE_RATES_INTERPOLATION = 3


class CurveInterpolatorFactory:
    """Factory of curve interpolators."""

    FACTORY = {
        CurveInterpolationType.RAW_INTERPOLATION: LogDiscountCurveInterpolator,
        CurveInterpolationType.LINEAR_RATES_INTERPOLATION: LinearRatesCurveInterpolator,
        CurveInterpolationType.LINEAR_DISCOUNT_INTERPOLATION: LinearDiscountCurveInterpolator,
        CurveInterpolationType.CUBIC_SPLINE_RATES_INTERPOLATION: CubicSplineRatesCurveInterpolator,
    }

    @classmethod
    def get(
        cls,
        t,
        df,
        interp: CurveInterpolationType = CurveInterpolationType.RAW_INTERPOLATION,
    ):
        """Return ready curve interpolator."""
        return cls.FACTORY[interp](t, df)
