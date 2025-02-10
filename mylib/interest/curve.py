"""Interest rate curves."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from mylib.interest.rates import Compound, implied_rate, discount_factor
from mylib.interest.interpolate import CurveInterpolationType, CurveInterpolatorFactory


class InterestRateCurve(ABC):
    """Base class for all interest rate curves."""

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def discount_factor(self, t):
        """Return discount factor."""

    def zero_rate(self, t, comp: Compound = Compound.CONTINUOUS):
        """Return zero rate."""

        return implied_rate(t, self.discount_factor(t), comp)

    def forward_rate(self, t1, t2, comp: Compound = Compound.CONTINUOUS):
        """Return forward rate from time t1 to time t2."""

        df1 = self.discount_factor(t1)
        df2 = self.discount_factor(t2)

        return implied_rate(t2 - t1, df2 / df1, comp)


@dataclass
class FlatCurve(InterestRateCurve):
    """Interest rate curve with constant level of interest rate."""

    rate: float
    comp: Compound = Compound.CONTINUOUS

    def discount_factor(self, t):
        return discount_factor(t, self.rate, self.comp)


class DiscountCurve(InterestRateCurve):
    """Interpolated discount curve."""

    def __init__(
        self,
        maturity: np.array,
        discount: np.array,
        interp: CurveInterpolationType = CurveInterpolationType.RAW_INTERPOLATION
    ) -> None:

        assert len(maturity) == len(discount)
        assert len(maturity) > 1

        self.maturity = maturity
        self.discount = discount
        self.interp_ = CurveInterpolatorFactory.get(maturity, discount, interp)

    def __str__(self) -> str:
        """Human readable representation of DiscountCurve object."""

        rate = 100.0 * implied_rate(self.maturity, self.discount)

        header = " Maturity | Discount | Interest "

        s = "\n" + "-" * len(header)
        s += "\n" + header
        s += "\n" + "-" * len(header)

        for i, _ in enumerate(self.maturity):
            s += f"\n{self.maturity[i]: 9.4f} | {self.discount[i]:.6f} | {rate[i]:7.4f}%"

        return s + "\n"

    def discount_factor(self, t):
        return self.interp_(t)
