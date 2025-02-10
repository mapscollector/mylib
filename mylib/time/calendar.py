"""Calendar operations."""

from abc import abstractmethod
from enum import IntEnum


def is_leap_year(y):
    """Check if a given year y is a leap year."""
    return y % 4 == 0


class DayOfWeek(IntEnum):
    """Days of week enumerator."""
    MONDAY      = 0
    TUESDAY     = 1
    WEDNESDAY   = 2
    THURSDAY    = 3
    FRIDAY      = 4
    SATURDAY    = 5
    SUNDAY      = 6


class Month(IntEnum):
    """Months enumerator."""
    JANUARY     = 1
    FEBRUARY    = 2
    MARCH       = 3
    APRIL       = 4
    MAY         = 5
    JUNE        = 6
    JULY        = 7
    AUGUST      = 8
    SEPTEMBER   = 9
    OCTOBER     = 10
    NOVEMBER    = 11
    DECEMBER    = 12


class BusinessDayConvention(IntEnum):
    """Business days conventions enumerator."""
    NO_ADJUSTMENT = 0
    PRECEDING = 1
    MODIFIED_PRECEDING = 2
    FOLLOWING = 3
    MODIFIED_FOLLOWING = 4


class Holiday:
    """Base class for holidays."""

    def __init__(self, name=None):
        self.name = name

    @abstractmethod
    def __eq__(self, d):
        pass


class HolidayFixed(Holiday):
    """Fixed holiday."""

    month:  Month
    day:    int

    def __init__(self, month, day, name=None):
        super.__init__(self, name)
        self.month = month
        self.day = day

    def __eq__(self, d):
        return False


class HolidayVariable(Holiday):
    """Variable holiday."""

    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

    def __eq__(self, d):
        return False


class Calendar:
    """Calendar."""

#pylint: disable=dangerous-default-value
    def __init__(self, fixed = [], variable = [],
                 weekend = [DayOfWeek.SATURDAY, DayOfWeek.SUNDAY]):
        self.fixed = fixed
        self.variable = variable
        self.weekend = weekend
#pylint: enable=dangerous-default-value

    def is_working_day(self, d):
        """Check if day is a working day."""
        return not(self.is_weekend(d) or self.is_holiday(d))

    def is_weekend(self, d):
        """Check if day is a weekend day."""
        return not d in self.weekend

    def is_holiday(self, d):
        """Check if day is a holiday."""
        return self.is_holiday_fixed(d) or self.is_holiday_variable(d)

    def is_holiday_fixed(self, d):
        """Check if day is a fixed holiday."""
        return d in self.fixed

    def is_holiday_variable(self, d):
        """Check if day is a variable holiday."""
        return d in self.variable
