import unittest
import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('..'))

from flowkit import Session, Sample, Matrix, Dimension, gates, transforms
from .prog_gating_unit_tests import data1_sample, poly1_gate, poly1_vertices, comp_matrix_01, asinh_xform1

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

        sample_ids = ["100715.fcs", "109567.fcs", "113548.fcs"]
        self.assertListEqual(fks.get_sample_ids(), sample_ids)

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
            fks.get_gate(
                'my_group',
                'data_set_simple_line_100.fcs',
                'poly1'),
            gates.PolygonGate
        )

        gate_ids = {'rect1', 'poly1'}
        self.assertSetEqual(set(fks.get_gate_ids('my_group')), gate_ids)

    def test_load_wsp_single_ellipse(self):
        wsp_path = "examples/simple_line_example/single_ellipse_51_events.wsp"
        fcs_path = "examples/simple_line_example/data_set_simple_line_100.fcs"

        fks = Session(fcs_samples=fcs_path)
        fks.import_flowjo_workspace(wsp_path)

        self.assertIsInstance(
            fks.get_gate(
                'All Samples',
                'data_set_simple_line_100.fcs',
                'ellipse1'),
            gates.EllipsoidGate
        )

        fks.analyze_samples(sample_group='All Samples')
        results = fks.get_gating_results('All Samples', 'data_set_simple_line_100.fcs')
        gate_count = results.get_gate_count('ellipse1')
        self.assertEqual(gate_count, 48)

    def test_load_wsp_single_quad(self):
        wsp_path = "examples/simple_diamond_example/simple_diamond_example_quad_gate.wsp"
        fcs_path = "examples/simple_diamond_example/test_data_diamond_01.fcs"

        fks = Session(fcs_samples=fcs_path)
        fks.import_flowjo_workspace(wsp_path)

        # FlowJo quadrant gates are not true quadrant gates, rather a collection of rectangle gates
        self.assertIsInstance(
            fks.get_gate(
                'All Samples',
                'test_data_diamond_01.fcs',
                'Q1: channel_A- , channel_B+'),
            gates.RectangleGate
        )

        fks.analyze_samples(sample_group='All Samples')
        results = fks.get_gating_results('All Samples', 'test_data_diamond_01.fcs')

        gate_count_q1 = results.get_gate_count('Q1: channel_A- , channel_B+')
        gate_count_q2 = results.get_gate_count('Q2: channel_A+ , channel_B+')
        gate_count_q3 = results.get_gate_count('Q3: channel_A+ , channel_B-')
        gate_count_q4 = results.get_gate_count('Q4: channel_A- , channel_B-')
        self.assertEqual(gate_count_q1, 49671)
        self.assertEqual(gate_count_q2, 50596)
        self.assertEqual(gate_count_q3, 50330)
        self.assertEqual(gate_count_q4, 49403)

    def test_get_sample_groups(self):
        wsp_path = "examples/simple_line_example/simple_poly_and_rect.wsp"
        fcs_path = "examples/simple_line_example/data_set_simple_line_100.fcs"

        fks = Session(fcs_samples=fcs_path)
        fks.import_flowjo_workspace(wsp_path)

        groups = fks.get_sample_groups()
        groups_truth = ['default', 'my_group']

        self.assertListEqual(groups, groups_truth)

        fks.add_sample_group('group2')
        groups_truth.append('group2')
        groups = fks.get_sample_groups()

        self.assertListEqual(groups, groups_truth)

    def test_get_comp_matrix(self):
        fks = Session(fcs_samples=data1_sample)
        fks.add_comp_matrix(comp_matrix_01)
        comp_mat = fks.get_comp_matrix('default', 'B07', 'MySpill')

        self.assertIsInstance(comp_mat, Matrix)

    def test_get_transform(self):
        fks = Session(fcs_samples=data1_sample)
        fks.add_transform(asinh_xform1)
        comp_mat = fks.get_transform('default', 'B07', 'AsinH_10000_4_1')

        self.assertIsInstance(comp_mat, transforms.AsinhTransform)

    @staticmethod
    def test_add_poly1_gate():
        fks = Session(fcs_samples=data1_sample)
        fks.add_gate(poly1_gate)
        fks.analyze_samples()
        result = fks.get_gating_results('default', data1_sample.original_filename)

        res_path = 'examples/gate_ref/truth/Results_Polygon1.txt'
        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        np.testing.assert_array_equal(truth, result.get_gate_indices('Polygon1'))

    @staticmethod
    def test_add_matrix_poly4_gate():
        fks = Session(fcs_samples=data1_sample)

        fks.add_comp_matrix(comp_matrix_01)

        dim1 = Dimension('PE', compensation_ref='MySpill')
        dim2 = Dimension('PerCP', compensation_ref='MySpill')
        dims = [dim1, dim2]

        poly_gate = gates.PolygonGate('Polygon4', None, dims, poly1_vertices)
        fks.add_gate(poly_gate)

        res_path = 'examples/gate_ref/truth/Results_Polygon4.txt'
        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        fks.analyze_samples()
        result = fks.get_gating_results('default', data1_sample.original_filename)

        np.testing.assert_array_equal(truth, result.get_gate_indices('Polygon4'))

    @staticmethod
    def test_add_transform_asinh_range1_gate():
        fks = Session(fcs_samples=data1_sample)
        fks.add_transform(asinh_xform1)

        dim1 = Dimension('FL1-H', 'uncompensated', 'AsinH_10000_4_1', range_min=0.37, range_max=0.63)
        dims = [dim1]

        rect_gate = gates.RectangleGate('ScaleRange1', None, dims)
        fks.add_gate(rect_gate)

        res_path = 'examples/gate_ref/truth/Results_ScaleRange1.txt'
        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        fks.analyze_samples()
        result = fks.get_gating_results('default', data1_sample.original_filename)

        np.testing.assert_array_equal(truth, result.get_gate_indices('ScaleRange1'))

    def test_calculate_comp_from_beads(self):
        bead_dir = "examples/4_color_beads"
        fks = Session()
        comp = fks.calculate_compensation_from_beads(bead_dir)

        self.assertIsInstance(comp, Matrix)

    def test_get_ambiguous_gate_objects(self):
        wsp_path = "examples/8_color_data_set/8_color_ICS.wsp"
        fcs_path = "examples/8_color_data_set/fcs_files/101_DEN084Y5_15_E01_008_clean.fcs"
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'
        sample_grp = 'DEN'
        gate_id = 'TNFa+'
        gate_path = ['root', 'Time', 'Singlets', 'aAmine-', 'CD3+', 'CD4+']

        fks = Session(fcs_samples=fcs_path)
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=True)

        fks.analyze_samples(sample_grp)
        gate_indices = fks.get_gate_indices(sample_grp, sample_id, gate_id, gate_path=gate_path)

        self.assertIsInstance(gate_indices, np.ndarray)
        self.assertEqual(np.sum(gate_indices), 21)

    def test_analyze_single_sample(self):
        wsp_path = "examples/8_color_data_set/8_color_ICS_simple.wsp"
        fcs_path = "examples/8_color_data_set/fcs_files"
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'
        sample_grp = 'DEN'

        fks = Session(fcs_samples=fcs_path)
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=True)

        sample_ids = fks.get_group_sample_ids(sample_grp)
        self.assertEqual(len(sample_ids), 3)

        fks.analyze_samples(sample_grp, sample_id=sample_id)
        report = fks.get_group_report(sample_grp)

        self.assertEqual(report.index.get_level_values('sample').nunique(), 1)
