"""Parametric interest rate curve models."""

from abc import abstractmethod
from dataclasses import dataclass

import numpy as np

from mylib.interest.rates import Compound, discount_factor
from mylib.interest.curve import InterestRateCurve


class ParametricInterestRateModel:
    """Base class for parametric interest rate models."""

    def __init__(self, model) -> None:
        pass

    @abstractmethod
    def inst_forward(self, t):
        """Return instantenuous forward rate at time t."""

    @abstractmethod
    def rate(self, t0, t1):
        """Return continuously compounded interest rate over period (t0, t1)."""


@dataclass
class NelsonSiegelModel(ParametricInterestRateModel):
    """Nelson-Siegel instrest rate curve model."""

    beta0_: float
    beta1_: float
    beta2_: float
    lambda_: float

    def inst_forward(self, t):
        """Return instantenuous forward rate at point t."""
        y = t / self.lambda_
        return self.beta0_ + (self.beta1_ + self.beta2_ * y) * np.exp(-y)

    def rate(self, t0, t1):
        """Return continuously compounded interest rate over period (t0, t1)."""

        dt = t1 - t0
        y1 = np.exp(-t0 / self.lambda_)
        y2 = np.exp(-t1 / self.lambda_)
        f1 = self.lambda_ * (y1 - y2) / dt
        f2 = ((self.lambda_ + t0) * y1 - (self.lambda_ + t1) * y2) / dt

        return self.beta0_ + self.beta1_ * f1 + self.beta2_ * f2


@dataclass
class SvenssonModel(ParametricInterestRateModel):
    """Nelson-Siegel instrest rate curve model."""

    beta0_: float
    beta1_: float
    beta2_: float
    beta3_: float
    lambda1_: float
    lambda2_: float

    def inst_forward(self, t):
        """Return instantenuous forward rate at point t."""

        y1 = t / self.lambda1_
        y2 = t / self.lambda2_

        return (
            self.beta0_
            + self.beta1_ * np.exp(-y1)
            + self.beta2_ * y1 * np.exp(-y1)
            + self.beta3_ * y2 * np.exp(-y2)
        )

    def rate(self, t0, t1):
        """Return continuously compounded interest rate over period (t0, t1)."""

        dt = t1 - t0
        y1 = np.exp(-t0 / self.lambda1_)
        y2 = np.exp(-t1 / self.lambda1_)
        y3 = np.exp(-t0 / self.lambda2_)
        y4 = np.exp(-t1 / self.lambda2_)

        f1 = self.lambda1_ * (y1 - y2) / dt
        f2 = ((self.lambda1_ + t0) * y1 - (self.lambda1_ + t1) * y2) / dt
        f3 = ((self.lambda2_ + t0) * y3 - (self.lambda2_ + t1) * y4) / dt

        return self.beta0_ + self.beta1_ * f1 + self.beta2_ * f2 + self.beta3_ * f3


class ParametricInterestRateCurve(InterestRateCurve):
    """Parametric interest rate curve.

    Parametric interest rate curve is based on ParametricInterestRateModel:
    * Nelson-Siegel model
    * Svensson model
    """

    def __init__(self, model: ParametricInterestRateModel) -> None:
        super().__init__()
        self.model = model

    def discount_factor(self, t):
        """Return discount factor for maturity t."""
        return discount_factor(t, self.model.rate(0.0, t),
                               comp=Compound.CONTINUOUS)
