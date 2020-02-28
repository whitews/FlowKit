import unittest
import sys
import os

sys.path.append(os.path.abspath('..'))

from flowkit import Session, Sample, Matrix, gates

fcs_file_paths = [
    "examples/100715.fcs",
    "examples/109567.fcs",
    "examples/113548.fcs"
]


class SessionTestCase(unittest.TestCase):
    """Tests for Session class"""
    def test_load_samples_from_list_of_paths(self):
        fks = Session(fcs_samples=fcs_file_paths)

        self.assertEqual(len(fks.sample_lut.keys()), 3)
        self.assertIsInstance(fks.get_sample('100715.fcs'), Sample)

    def test_load_samples_from_list_of_samples(self):
        samples = [Sample(file_path) for file_path in fcs_file_paths]
        fks = Session(fcs_samples=samples)

        self.assertEqual(len(fks.sample_lut.keys()), 3)
        self.assertIsInstance(fks.get_sample('100715.fcs'), Sample)

    def test_load_wsp_single_poly(self):
        wsp_path = "examples/simple_line_example/simple_poly_and_rect.wsp"
        fcs_path = "examples/simple_line_example/data_set_simple_line_100.fcs"

        fks = Session(fcs_samples=fcs_path)
        fks.import_flowjo_workspace(wsp_path)

        self.assertIsInstance(
            fks.get_gate_by_reference(
                'my_group',
                'data_set_simple_line_100.fcs',
                'poly1'),
            gates.PolygonGate
        )

    def test_calculate_comp_from_beads(self):
        bead_dir = "examples/4_color_beads"
        fks = Session()
        comp = fks.calculate_compensation_from_beads(bead_dir)

        self.assertIsInstance(comp, Matrix)
