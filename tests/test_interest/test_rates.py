"""Test mylib.interest.rates module."""

import pytest
import numpy as np

from mylib.interest.rates import discount_factor


# test types
# test values
# test conversion

def test_implied_rate():
    """Some test."""
    assert pytest.approx(discount_factor(1.0, 0.04) - np.exp(-0.04)) == 0
