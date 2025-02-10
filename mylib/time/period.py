"""Time periods."""

from enum import IntEnum


class PeriodUnit(IntEnum):
    """Period units enumerator."""
    DAY = 1
    WEEK = 2
    MONTH = 3
    YEAR = 4


class Period:
    """n"""

    def __init__(self, count: int, unit: PeriodUnit):
        self.count = count
        self.unit = unit

    def __str__(self):

        return f"{self.count} {self.unit.name.capitalize()}{'s' if self.count > 1 else ''}"

    def __add__(self, obj):

        assert self.unit == obj.unit

        return Period(self.count + obj.count, self.unit)
