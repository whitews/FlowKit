import unittest
import os
import sys

sys.path.append(os.path.abspath('../..'))

from flowkit import Matrix, calculate_compensation_from_beads


class SampleUtilsTestCase(unittest.TestCase):
    """Tests for Session class"""
    def test_calculate_comp_from_beads(self):
        bead_dir = "examples/4_color_beads"
        comp = calculate_compensation_from_beads(bead_dir)

        self.assertIsInstance(comp, Matrix)