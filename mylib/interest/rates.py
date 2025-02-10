"""Module with basic interest rate operations."""

from enum import IntEnum

import numpy as np


class Compound(IntEnum):
    """Compounding methods."""

    CONTINUOUS = -1
    SIMPLE = 0
    ANNUAL = 1
    SEMIANNUAL = 2
    QUARTERLY = 4
    MONTHLY = 12


def implied_rate(t: np.array, df: np.array, comp: Compound = Compound.CONTINUOUS):
    """Calculate implied zero rate given time t, discount factor df and compounding comp."""

    if comp == Compound.CONTINUOUS:
        return -np.log(df) / t
    if comp == Compound.SIMPLE:
        return (1.0 / df - 1) / t
    if comp in Compound:
        return (df ** (-1.0 / (t * comp)) - 1) * comp

    raise ValueError


def capitalization(t: np.array, r: np.array, comp: Compound = Compound.CONTINUOUS):
    """Calculate capitalization give time t, rate r and compounding comp."""

    if comp == Compound.CONTINUOUS:
        return np.exp(r * t)
    if comp == Compound.SIMPLE:
        return 1.0 + r * t
    if comp in Compound:
        return (1.0 + r / comp) ** (t * comp)

    raise ValueError


def discount_factor(t: np.array, r: np.array, comp: Compound = Compound.CONTINUOUS):
    """Calculate discount factor given time t, rate r and compounding comp."""

    return 1.0 / capitalization(t, r, comp)
