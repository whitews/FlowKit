"""
Unit tests for base Gate class methods
"""

import unittest

from flowkit import Dimension
from tests.test_config import poly1_gate


class GateTestCase(unittest.TestCase):
    """Tests Gate objects"""

    def test_get_gate_dimension(self):
        dim = poly1_gate.get_dimension("FL2-H")

        self.assertIsInstance(dim, Dimension)

    def test_get_gate_dimension_ids(self):
        dim_ids = poly1_gate.get_dimension_ids()

        self.assertListEqual(dim_ids, ["FL2-H", "FL3-H"])
