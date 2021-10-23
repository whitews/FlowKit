"""
Unit tests for Gate sub-classes
"""
import unittest
import numpy as np

from flowkit import Dimension
from .gating_strategy_tests import poly1_gate

test_data_range1 = np.linspace(0.0, 10.0, 101)


class GateTestCase(unittest.TestCase):
    """Tests Gate objects"""
    def test_get_gate_dimension(self):
        dim = poly1_gate.get_dimension('FL2-H')

        self.assertIsInstance(dim, Dimension)
