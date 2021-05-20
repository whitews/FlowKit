"""
Tests for FlowJo 10 workspace files
"""
import unittest
import os
import numpy as np
from flowkit import Session, gates, transforms


class FlowJoWSPTestCase(unittest.TestCase):
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
        wsp_gates_tuple = fks.get_gate_ids('my_group')
        wsp_gate_ids = set([g[0] for g in wsp_gates_tuple])
        self.assertSetEqual(wsp_gate_ids, gate_ids)

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

        fks.analyze_samples(group_name='All Samples')
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

        fks.analyze_samples(group_name='All Samples')
        results = fks.get_gating_results('All Samples', 'test_data_diamond_01.fcs')

        gate_count_q1 = results.get_gate_count('Q1: channel_A- , channel_B+')
        gate_count_q2 = results.get_gate_count('Q2: channel_A+ , channel_B+')
        gate_count_q3 = results.get_gate_count('Q3: channel_A+ , channel_B-')
        gate_count_q4 = results.get_gate_count('Q4: channel_A- , channel_B-')
        self.assertEqual(gate_count_q1, 49671)
        self.assertEqual(gate_count_q2, 50596)
        self.assertEqual(gate_count_q3, 50330)
        self.assertEqual(gate_count_q4, 49403)

    def test_wsp_biex_transform(self):
        wsp_path = "examples/simple_diamond_example/test_data_diamond_biex_rect.wsp"
        fcs_path = "examples/simple_diamond_example/test_data_diamond_01.fcs"

        fks = Session(fcs_samples=fcs_path)
        fks.import_flowjo_workspace(wsp_path)

        self.assertIsInstance(
            fks.get_gate(
                'All Samples',
                'test_data_diamond_01.fcs',
                'upper_right'),
            gates.RectangleGate
        )

        fks.analyze_samples(group_name='All Samples')
        results = fks.get_gating_results('All Samples', 'test_data_diamond_01.fcs')
        gate_count = results.get_gate_count('upper_right')
        self.assertEqual(gate_count, 50605)

    def test_wsp_fasinh_transform(self):
        wsp_path = "examples/simple_diamond_example/test_data_diamond_asinh_rect.wsp"
        fcs_path = "examples/simple_diamond_example/test_data_diamond_01.fcs"

        fks = Session(fcs_samples=fcs_path)
        fks.import_flowjo_workspace(wsp_path)

        self.assertIsInstance(
            fks.get_gate(
                'All Samples',
                'test_data_diamond_01.fcs',
                'upper_right'),
            gates.RectangleGate
        )

        fks.analyze_samples(group_name='All Samples')
        results = fks.get_gating_results('All Samples', 'test_data_diamond_01.fcs')
        gate_count = results.get_gate_count('upper_right')
        self.assertEqual(gate_count, 50559)

    def test_wsp_fasinh_transform_v2(self):
        wsp_path = "examples/simple_diamond_example/test_data_diamond_asinh_rect2.wsp"
        fcs_path = "examples/simple_diamond_example/test_data_diamond_01.fcs"

        fks = Session(fcs_samples=fcs_path)
        fks.import_flowjo_workspace(wsp_path)

        self.assertIsInstance(
            fks.get_gate(
                'All Samples',
                'test_data_diamond_01.fcs',
                'upper_right'),
            gates.RectangleGate
        )

        fks.analyze_samples(group_name='All Samples')
        results = fks.get_gating_results('All Samples', 'test_data_diamond_01.fcs')
        gate_count = results.get_gate_count('upper_right')
        self.assertEqual(gate_count, 50699)

    def test_wsp_biex_transform_width_interpolation(self):
        neg = 1.0
        width = -7.943282

        # this LUT exists for only the single negative value of 1.0
        lut_file_name = "tr_biex_l256_w%.6f_n%.6f_m4.418540_r262144.000029.csv" % (width, neg)
        lut_file_path = os.path.join('examples', 'flowjo_xforms', lut_file_name)
        y, x = np.loadtxt(lut_file_path, delimiter=',', usecols=(0, 1), skiprows=1, unpack=True)

        biex_xform = transforms.WSPBiexTransform('biex', negative=neg, width=width)

        test_y = biex_xform.apply(x)

        mean_pct_diff = 100. * np.mean(np.abs(test_y[1:] - y[1:]) / y[1:])
        self.assertLess(mean_pct_diff, 1.0)

    def test_get_sample_groups(self):
        wsp_path = "examples/simple_line_example/simple_poly_and_rect.wsp"
        fcs_path = "examples/simple_line_example/data_set_simple_line_100.fcs"

        fks = Session(fcs_samples=fcs_path)
        fks.import_flowjo_workspace(wsp_path)

        groups = fks.get_sample_groups()
        groups_truth = ['default', 'All Samples', 'my_group']

        self.assertListEqual(groups, groups_truth)

        fks.add_sample_group('group2')
        groups_truth.append('group2')
        groups = fks.get_sample_groups()

        self.assertListEqual(groups, groups_truth)

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
