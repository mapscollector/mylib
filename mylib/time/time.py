"""Time operations."""

from enum import IntEnum


class DayCount(IntEnum):
    """Day count conventions enumerator."""
    ACT_365 = 0
    ACT_360 = 1
    THIRTY_360 = 2
    ACT_ACT = 3



def year_fraction(d1, d2, day_count: DayCount=DayCount.ACT_360):
    """Calculate year fraction between d1 and d2 in line with day_count convention."""
    pass
