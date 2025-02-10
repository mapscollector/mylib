"""Discount factor conversion."""

from abc import abstractmethod
from enum import IntEnum

import numpy as np

from mylib.interest.rates import discount_factor, implied_rate


class ConversionType(IntEnum):
    """Conversion types enumerator."""

    IDENTITY = 0
    LOGARITHMIC = 1
    ZERO_RATES = 2
    INVERSION = 3
    RATE_TIME = 4


class Converter:
    """Base class for discount factor converters.

    Converters are used in interest rate interpolation.
    Interest rates interpolation assumes that some function f(x) of discount factor is interpolated.
    Converters convert discount factors into interpolated values and back.

    convert():  converts discount factor to interpolated value
    revert():   reverts from interpolated value back to discount factor
    """

    @classmethod
    @abstractmethod
    def convert(cls, t, x):
        """Convert to"""

    @classmethod
    @abstractmethod
    def revert(cls, t, y):
        """Revert from"""

    @classmethod
    def is_correct(cls, t, x):
        """Check if conversion and reversion work properly."""
        y = cls.convert(t, x)
        return np.isclose(x, cls.revert(t, y))


class IdentityConverter(Converter):
    """Identity converter - dummy (no) conversion."""

    @classmethod
    def convert(cls, t, x):
        return x

    @classmethod
    def revert(cls, t, y):
        return y


class LogConverter(Converter):
    """Log discount factor converter."""

    @classmethod
    def convert(cls, t, x):
        return np.log(x)

    @classmethod
    def revert(cls, t, y):
        return np.exp(y)


class ZeroRatesConverter(Converter):
    """Zero rates converter."""

    @classmethod
    def convert(cls, t, x):
        return implied_rate(t, x)

    @classmethod
    def revert(cls, t, y):
        return discount_factor(t, y)


class InversionConverter(Converter):
    """Inversion converter."""

    @classmethod
    def convert(cls, t, x):
        return 1.0 / x

    @classmethod
    def revert(cls, t, y):
        return cls.convert(t, y)


class RateTimeConverter(LogConverter):
    """Rate(cont. comp.)*time converter."""

    @classmethod
    def convert(cls, t, x):
        return -LogConverter.convert(t, x)

    @classmethod
    def revert(cls, t, y):
        return LogConverter.revert(t, -y)


# pylint: disable=too-few-public-methods


class ConverterFactory:
    """Factory of converters."""

    FACTORY = {
        ConversionType.IDENTITY: IdentityConverter,
        ConversionType.LOGARITHMIC: LogConverter,
        ConversionType.ZERO_RATES: ZeroRatesConverter,
        ConversionType.INVERSION: InversionConverter,
        ConversionType.RATE_TIME: RateTimeConverter,
    }

    @classmethod
    def get(cls, method: ConversionType = ConversionType.IDENTITY):
        """Return converter based on conversion method."""
        return cls.FACTORY[method]
