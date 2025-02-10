"""Short rate stochastic models."""

from abc import abstractmethod, ABC
from scipy.stats import ncx2
from scipy.optimize import minimize

import numpy as np

from mylib.interest.curve import InterestRateCurve
from mylib.interest.rates import Compound
from mylib.model.option import OptionType, BlackFormula


#pylint: disable=too-few-public-methods

class ShortRateModel(ABC):
    """Base class for all short rate models."""

    @abstractmethod
    def process(self):
        """Return the process representation."""


class NonArbitrageModel(ShortRateModel):
    """Base class for all short rate non-arbitrage models."""

    def __init__(self, int_rate_curve: InterestRateCurve) -> None:
        self.ts = int_rate_curve

    def instantenuous_fwd(self, t, dt=0.0001):
        """Return instantenuous forward rate for time t."""

        return self.ts.forward_rate(t, t+dt, comp=Compound.CONTINUOUS)


class GeneralEquilibriumModel(ShortRateModel):
    """Base class for all short rate general equilibrium models."""

    def __init__(self, r0: float) -> None:
        self.r0 = r0

#pylint: enable=too-few-public-methods

class AffineModel(ShortRateModel):
    """Base class for all affine short rate models."""

    @abstractmethod
    def a(self, t0, t1):
        """Return A(t0, t1) factor."""

    @abstractmethod
    def b(self, t0, t1):
        """Return B(t0, t1) factor."""

    def alpha(self, t0, t1):
        """Return \alpha(t0, t1) factor."""
        return -np.log(self.a(t0, t1)) / (t1 - t0)

    def beta(self, t0, t1):
        """Return \beta(t0, t1) factor."""
        return self.b(t0, t1) / (t1 - t0)


class OneFactorAffineModel(AffineModel):
    """Base class for one factor short rete affine models."""

    @abstractmethod
    def expectation(self, t0, r0, t1):
        """Return expected value of r0."""

    @abstractmethod
    def variance(self, t0, r0, t1):
        """Return variance of r0."""

    def _jamshidian(self, t, ti, c, y):
        """Return r^* using Jamshidian trick."""

        def error(x):

            return np.square(np.sum(c * self.discount(t, ti, x) - y))

        res = minimize(error, np.array(0.0))

        return res.x[0]

    def discount(self, t0, r0, t1):
        """Return (state dependent) discount factor."""

        return self.a(t0, t1) * np.exp(-self.b(t0, t1) * r0)

    def rate(self, t0, r0, t1):
        """Return (state dependent) continuously compounded zero rate."""

        return self.alpha(t0, t1) + self.beta(t0, t1) * r0

    def discount_bond_opt(self, t0, r0, t1, t2, k, opt: OptionType=OptionType.CALL):
        """Return the price of zero coupon bond option."""

        p1 = self.discount(t0, r0, t1)
        p2 = self.discount(t0, r0, t2)
        sigma = np.sqrt(self.variance(t0, r0, t1)) * self.b(t1, t2) / t1

        return BlackFormula.value(1.0, p2, p1 * k, t1, sigma, opt)

    def caplet(self, t0, r0, t1, t2, tau, k, opt: OptionType=OptionType.CALL):
        """Return the price of a caplet/floorlet."""

        acc = 1.0 + k * tau

        return acc * self.discount_bond_opt(t0, r0, t1, t2, 1.0 / acc, -opt)

    def capfloor(self, t0, r0, t1, t2, tau, k, opt: OptionType=OptionType.CALL):
        """Return the price of a cap/floor."""

        return np.sum(self.caplet(t0, r0, t1, t2, tau, k, opt))

    def swaption(self, t0, r0, t1, t2, tau, k, opt: OptionType=OptionType.CALL):
        """Return the price of a swaption."""

        c = k * tau
        c[-1] += 1

        r_star = self._jamshidian(t1[0], t2, c, 1.0)
        x = self.discount(t1[0], t2, r_star)

        return np.sum(c * self.discount_bond_opt(t0, r0, t1[0], t2, x, -opt))


class VasicekModel(OneFactorAffineModel, GeneralEquilibriumModel):
    """Implementation of interest rate Vasicek model."""

    def __init__(self, r0: float, k: float, theta: float, sigma: float) -> None:
        super().__init__(r0)
        self.k = k
        self.theta = theta
        self.sigma = sigma

    def a(self, t0, t1):

        b = self.b(t0, t1)
        tmp = (self.theta - 0.5 * np.square(self.sigma / self.k)) * (b - t1 + t0)
        tmp -= 0.25 * np.square(self.sigma * b) / self.k

        return np.exp(tmp)

    def b(self, t0, t1):

        return (1 - np.exp(-self.k * (t1 - t0))) / self.k

    def expectation(self, t0, r0, t1):
        """Return expected value of short rate."""

        return r0 * np.exp(-self.k * (t1 - t0)) + self.theta * (1 - np.exp(-self.k * (t1 - t0)))

    def variance(self, t0, r0, t1):
        """Return variance of short rate."""

        return 0.5 * np.square(self.sigma) * (1 - np.exp(-2.0 * self.k * (t1 - t0))) / self.k

    def process(self):
        return None


class CoxIngersollRossModel(OneFactorAffineModel, GeneralEquilibriumModel):
    """CIR model."""

    def __init__(self, r0, k, theta, sigma):
        GeneralEquilibriumModel.__init__(self, r0)
        self.k = k
        self.theta = theta
        self.sigma = sigma

    def a(self, t0, t1):

        h = self._h()
        num = 2 * h * np.exp(0.5 * (self.k + h) * (t1 - t0))
        den = 2 * h + (self.k + h) * (np.exp((t1 - t0) * h) - 1.0)
        x = num / den

        return x ** (2.0 * self.k * self.theta / self.sigma**2)

    def b(self, t0, t1):

        h = self._h()
        x = np.exp((t1 - t0) * h) - 1.0

        return 2 * x / (2 * h + (self.k + h) * x)

    def expectation(self, t0, r0, t1):

        return r0 * np.exp(-self.k * (t1 - t0)) + self.theta * (1.0 - np.exp(-self.k * (t1 - t0))
        )

    def variance(self, t0, r0, t1):

        x0 = np.exp(-self.k * (t1 - t0))
        x1 = x0 - np.square(x0)
        x2 = np.square(1 - x0)

        return np.square(self.sigma) / self.k * (r0, x1 + 0.5 * self.theta * x2)

    def _h(self):
        """Return h - helper function."""

        return np.sqrt(self.k ** 2 + 2 * self.sigma ** 2)

    def discount_bond_opt(self, t0, r0, t1, t2, k, opt: OptionType = OptionType.CALL):

        p1 = self.discount(t0, r0, t2)
        p2 = self.discount(t0, r0, t1)

        b = self.b(t1, t2)
        h = self._h()
        rho = 2 * k / (np.square(self.sigma) * (np.exp(k * (t1 - t0)) - 1))
        psi = (self.k + h) / np.square(self.sigma)
        r = np.log(self.a(t1, t2) / k) / self.b(t1, t2)

        df = 4.0 * self.k * self.theta / np.square(self.sigma)
        x1 = 2.0 * np.square(rho) * self.r0 * np.exp(h * t1)
        x2 = rho + psi

        chi1 = ncx2.cdf(2 * r * (x2 + b), df, x1 / (x2 + b))
        chi2 = ncx2.cdf(2 * r * x2, df, x1 / x2)

        call = p1 * chi1 - k * p2 * chi2

        if opt == OptionType.CALL:
            return call

        return call - p1 + k * p2

    def process(self):
        return None
