"""Basic option models."""

from abc import abstractmethod, ABC
from enum import IntEnum
from scipy.stats import norm
from scipy.optimize import brentq

import numpy as np


class OptionType(IntEnum):
    """Option type enumerator."""

    CALL = 1
    PUT = -1


# pylint: disable=too-many-arguments
class OptionFormula(ABC):
    """Base class for option formulas."""

    @abstractmethod
    @staticmethod
    def value(df, f, k, t, sigma, opt_type: OptionType=OptionType.CALL):
        """Calculate option value."""

    @abstractmethod
    @staticmethod
    def delta(df, f, k, t, sigma, opt_type: OptionType=OptionType.CALL):
        """Calculate option delta."""

    @abstractmethod
    @staticmethod
    def gamma(df, f, k, t, sigma, opt_type: OptionType=OptionType.CALL):
        """Calculate option delta."""

    @abstractmethod
    @staticmethod
    def vega(df, f, k, t, sigma, opt_type: OptionType=OptionType.CALL):
        """Calculate option delta."""

    @abstractmethod
    @staticmethod
    def vanna(df, f, k, t, sigma, opt_type: OptionType=OptionType.CALL):
        """Calculate option vanna."""

    @abstractmethod
    @staticmethod
    def volga(df, f, k, t, sigma, opt_type: OptionType=OptionType.CALL):
        """Calculate option volga."""

    @classmethod
    def implied_volatility(cls, df, f, k, t, v, opt_type: OptionType=OptionType.CALL):
        """Calculate implied volatility."""

        b = 2.0
        while cls.value(df, f, k, t, b, opt_type) < v:
            b +=1.0

        def err(x):
            return cls.value(df, f, k, t, x, opt_type) - v

        return brentq(err, 0.0001, b, maxiter=1000)


class BlackFormula(OptionFormula):
    """Base class for option formulas."""

    @staticmethod
    def _precalc(f, k, std):

        d1 = np.log(f / k) + 0.5 * std**2
        d1 = d1 / std
        d2 = d1 - std

        return d1, d2

    @staticmethod
    def value(df, f, k, t, sigma, opt_type: OptionType=OptionType.CALL):
        """Calculate option value."""

        std = sigma * np.sqrt(t)
        d1, d2 = BlackFormula._precalc(f, k, std)

        return opt_type * df * (
            f * norm.cdf(opt_type * d1) - k * norm.cdf(opt_type * d2)
            )

    @staticmethod
    def delta(df, f, k, t, sigma, opt_type: OptionType=OptionType.CALL):
        """Calculate option delta."""

        std = sigma * np.sqrt(t)
        d1, _ = BlackFormula._precalc(f, k, std)

        return opt_type * df * norm.cdf(opt_type * d1)

    @staticmethod
    def gamma(df, f, k, t, sigma, opt_type: OptionType=OptionType.CALL):
        """Calculate option gamma."""

        std = sigma * np.sqrt(t)
        d1, _ = BlackFormula._precalc(f, k, std)

        return df * norm.pdf(d1) / (f * std)

    @staticmethod
    def vega(df, f, k, t, sigma, opt_type: OptionType=OptionType.CALL):
        """Calculate option vega."""

        std = sigma * np.sqrt(t)
        d1, _ = BlackFormula._precalc(f, k, std)

        return df * f * norm.pdf(d1) * np.sqrt(t)

    @staticmethod
    def vanna(df, f, k, t, sigma, opt_type: OptionType=OptionType.CALL):
        """Calculate option vanna."""

        std = sigma * np.sqrt(t)
        d1, d2 = BlackFormula._precalc(f, k, std)

        return -df * norm.pdf(d1) * d2 / sigma

    @staticmethod
    def volga(df, f, k, t, sigma, opt_type: OptionType=OptionType.CALL):
        """Calculate option volga."""

        std = sigma * np.sqrt(t)
        d1, d2 = BlackFormula._precalc(f, k, std)

        return BlackFormula.vega(df, f, k, t, sigma) * d1 * d2 / sigma


class BachelierFormula(OptionFormula):
    """Base class for option formulas."""

    @staticmethod
    def _precalc(f, k, t, sigma):

        std = sigma * np.sqrt(t)
        d0 = (f - k) / std

        return d0, std

    @staticmethod
    def value(df, f, k, t, sigma, opt_type: OptionType=OptionType.CALL):
        """Calculate option value."""

        d0, std = BachelierFormula._precalc(f, k, t, sigma)

        return opt_type * df * (
            (f - k) * norm.cdf(opt_type * d0) + opt_type * std * norm.pdf(d0)
            )

    @staticmethod
    def delta(df, f, k, t, sigma, opt_type: OptionType=OptionType.CALL):
        """Calculate option delta."""

        d0, _ = BachelierFormula._precalc(f, k, t, sigma)

        return opt_type * df * norm.cdf(opt_type * d0)

    @staticmethod
    def gamma(df, f, k, t, sigma, opt_type: OptionType=OptionType.CALL):
        """Calculate option gamma."""

        d0, std = BachelierFormula._precalc(f, k, t, sigma)

        return df * norm.pdf(d0) / std

    @staticmethod
    def vega(df, f, k, t, sigma, opt_type: OptionType=OptionType.CALL):
        """Calculate option vega."""

        d0, _ = BachelierFormula._precalc(f, k, t, sigma)

        return norm.pdf(d0) * np.sqrt(t)

    @staticmethod
    def vanna(df, f, k, t, sigma, opt_type: OptionType=OptionType.CALL):
        """Calculate option vanna."""

        d0, _ = BachelierFormula._precalc(f, k, t, sigma)

        return -norm.pdf(d0) * d0 / sigma

    @staticmethod
    def volga(df, f, k, t, sigma, opt_type: OptionType=OptionType.CALL):
        """Calculate option vega."""

        d0, std = BachelierFormula._precalc(f, k, t, sigma)

        return norm.pdf(d0) * d0**2 * t / std


class VolatilityConverter:
    """Volatility converter converts implied volatility levels between Bachelier and Black model."""

    @classmethod
    def to_bachelier(cls, df, f, k, t, sigma, opt_type: OptionType=OptionType.CALL):
        """Convert Black implied volatility to Bachelier one."""

        v = BlackFormula.value(df, f, k, t, sigma, opt_type)

        return BachelierFormula.implied_volatility(f, k, df, t, v, opt_type)

    @classmethod
    def to_black(cls, df, f, k, t, sigma, opt_type: OptionType=OptionType.CALL):
        """Convert Bachelier implied volatility to Black one."""

        v = BachelierFormula.value(df, f, k, t, sigma, opt_type)

        return BlackFormula.implied_volatility(f, k, df, t, v, opt_type)
