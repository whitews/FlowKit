import unittest
import sys
import os

sys.path.append(os.path.abspath('..'))

from flowkit import Session, Matrix


class SessionTestCase(unittest.TestCase):
    """Tests for Session class"""
    def test_calculate_comp_from_beads(self):
        bead_dir = "examples/4_color_beads"
        fks = Session()
        comp = fks.calculate_compensation_from_beads(bead_dir)

        self.assertIsInstance(comp, Matrix)
