"""..."""

from dataclasses import dataclass
from enum import Enum


@dataclass
class SaCcrParameters:
    """..."""

    alpha: float = 1.40


class AssetClass(IntEnum):
    """..."""

    INTEREST_RATE
    FOREIGN_EXCHANGE
    CREDIT
    EQUITY
    COMMODITY
