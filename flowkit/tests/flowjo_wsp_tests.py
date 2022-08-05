"""
Tests for FlowJo 10 workspace files
"""
import copy
import unittest
import os
from io import BytesIO
import numpy as np
from flowkit import Session, gates, transforms, extract_wsp_sample_data
from .session_tests import test_samples_8c_full_set


class FlowJoWSPTestCase(unittest.TestCase):
    def test_load_wsp_single_poly(self):
        wsp_path = "examples/data/simple_line_example/simple_poly_and_rect.wsp"
        fcs_path = "examples/data/simple_line_example/data_set_simple_line_100.fcs"

        fks = Session(fcs_samples=fcs_path)
        fks.import_flowjo_workspace(wsp_path)

        self.assertIsInstance(
            fks.get_gate(
                'my_group',
                'poly1',
                sample_id='data_set_simple_line_100.fcs'
            ),
            gates.PolygonGate
        )

        gate_names = {'rect1', 'poly1'}
        wsp_gates_tuple = fks.get_gate_ids('my_group')
        wsp_gate_names = set([g[0] for g in wsp_gates_tuple])
        self.assertSetEqual(wsp_gate_names, gate_names)

    def test_load_wsp_single_ellipse(self):
        wsp_path = "examples/data/simple_line_example/single_ellipse_51_events.wsp"
        fcs_path = "examples/data/simple_line_example/data_set_simple_line_100.fcs"

        fks = Session(fcs_samples=fcs_path)
        fks.import_flowjo_workspace(wsp_path)

        self.assertIsInstance(
            fks.get_gate(
                'All Samples',
                'ellipse1',
                sample_id='data_set_simple_line_100.fcs'
            ),
            gates.PolygonGate
        )

        fks.analyze_samples(group_name='All Samples')
        results = fks.get_gating_results('All Samples', 'data_set_simple_line_100.fcs')
        gate_count = results.get_gate_count('ellipse1')
        self.assertEqual(gate_count, 51)

    def test_load_wsp_single_quad(self):
        wsp_path = "examples/data/simple_diamond_example/simple_diamond_example_quad_gate.wsp"
        fcs_path = "examples/data/simple_diamond_example/test_data_diamond_01.fcs"

        fks = Session(fcs_samples=fcs_path)
        fks.import_flowjo_workspace(wsp_path)

        # FlowJo quadrant gates are not true quadrant gates, rather a collection of rectangle gates
        self.assertIsInstance(
            fks.get_gate(
                'All Samples',
                'Q1: channel_A- , channel_B+',
                sample_id='test_data_diamond_01.fcs'
            ),
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
        wsp_path = "examples/data/simple_diamond_example/test_data_diamond_biex_rect.wsp"
        fcs_path = "examples/data/simple_diamond_example/test_data_diamond_01.fcs"

        fks = Session(fcs_samples=fcs_path)
        fks.import_flowjo_workspace(wsp_path)

        self.assertIsInstance(
            fks.get_gate(
                'All Samples',
                'upper_right',
                sample_id='test_data_diamond_01.fcs'
            ),
            gates.RectangleGate
        )

        fks.analyze_samples(group_name='All Samples')
        results = fks.get_gating_results('All Samples', 'test_data_diamond_01.fcs')
        gate_count = results.get_gate_count('upper_right')
        self.assertEqual(gate_count, 50605)

    def test_wsp_fasinh_transform(self):
        wsp_path = "examples/data/simple_diamond_example/test_data_diamond_asinh_rect.wsp"
        fcs_path = "examples/data/simple_diamond_example/test_data_diamond_01.fcs"

        fks = Session(fcs_samples=fcs_path)
        fks.import_flowjo_workspace(wsp_path)

        self.assertIsInstance(
            fks.get_gate(
                'All Samples',
                'upper_right',
                sample_id='test_data_diamond_01.fcs'
            ),
            gates.RectangleGate
        )

        fks.analyze_samples(group_name='All Samples')
        results = fks.get_gating_results('All Samples', 'test_data_diamond_01.fcs')
        gate_count = results.get_gate_count('upper_right')
        self.assertEqual(gate_count, 50559)

    def test_wsp_fasinh_transform_v2(self):
        wsp_path = "examples/data/simple_diamond_example/test_data_diamond_asinh_rect2.wsp"
        fcs_path = "examples/data/simple_diamond_example/test_data_diamond_01.fcs"

        fks = Session(fcs_samples=fcs_path)
        fks.import_flowjo_workspace(wsp_path)

        self.assertIsInstance(
            fks.get_gate(
                'All Samples',
                'upper_right',
                sample_id='test_data_diamond_01.fcs'
            ),
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
        lut_file_path = os.path.join('examples', 'data', 'flowjo_xforms', lut_file_name)
        y, x = np.loadtxt(lut_file_path, delimiter=',', usecols=(0, 1), skiprows=1, unpack=True)

        biex_xform = transforms.WSPBiexTransform('biex', negative=neg, width=width)

        test_y = biex_xform.apply(x)

        mean_pct_diff = 100. * np.mean(np.abs(test_y[1:] - y[1:]) / y[1:])
        self.assertLess(mean_pct_diff, 0.01)

    def test_get_sample_groups(self):
        wsp_path = "examples/data/simple_line_example/simple_poly_and_rect.wsp"
        fcs_path = "examples/data/simple_line_example/data_set_simple_line_100.fcs"

        fks = Session(fcs_samples=fcs_path)
        fks.import_flowjo_workspace(wsp_path)

        groups = fks.get_sample_groups()
        groups_truth = ['default', 'All Samples', 'my_group']

        self.assertListEqual(groups, groups_truth)

        fks.add_sample_group('group2')
        groups_truth.append('group2')
        groups = fks.get_sample_groups()

        self.assertListEqual(groups, groups_truth)

    def test_parse_wsp_with_ellipse(self):
        wsp_path = "examples/data/8_color_data_set/8_color_ICS_with_ellipse.wsp"
        fcs_path = "examples/data/8_color_data_set/fcs_files/101_DEN084Y5_15_E01_008_clean.fcs"
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'
        sample_grp = 'DEN'
        gate_name = 'ellipse1'
        gate_path = ('root', 'Time', 'Singlets', 'aAmine-', 'CD3+')

        fks = Session(fcs_samples=fcs_path)
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=True)

        fks.analyze_samples(sample_grp, sample_id=sample_id)
        gate_indices = fks.get_gate_membership(sample_grp, sample_id, gate_name, gate_path=gate_path)

        self.assertIsInstance(gate_indices, np.ndarray)
        self.assertEqual(np.sum(gate_indices), 7023)

    def test_get_ambiguous_gate_objects(self):
        wsp_path = "examples/data/8_color_data_set/8_color_ICS.wsp"
        fcs_path = "examples/data/8_color_data_set/fcs_files/101_DEN084Y5_15_E01_008_clean.fcs"
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'
        sample_grp = 'DEN'
        gate_name = 'TNFa+'
        gate_path = ('root', 'Time', 'Singlets', 'aAmine-', 'CD3+', 'CD4+')

        fks = Session(fcs_samples=fcs_path)
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=True)

        fks.analyze_samples(sample_grp)
        gate_indices = fks.get_gate_membership(sample_grp, sample_id, gate_name, gate_path=gate_path)

        self.assertIsInstance(gate_indices, np.ndarray)
        self.assertEqual(np.sum(gate_indices), 21)

    def test_parse_wsp_reused_gate_with_child(self):
        wsp_path = "examples/data/8_color_data_set/reused_quad_gate_with_child.wsp"

        fks = Session(copy.deepcopy(test_samples_8c_full_set))
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=True)
        group_name = 'All Samples'
        gate_name = 'some_child_gate'

        gate_ids = fks.get_gate_ids(group_name)

        gate_id_1 = (gate_name, ('root', 'good cells', 'cd4+', 'Q2: CD107a+, IL2+'))
        gate_id_2 = (gate_name, ('root', 'good cells', 'cd8+', 'Q2: CD107a+, IL2+'))

        self.assertIn(gate_id_1, gate_ids)
        self.assertIn(gate_id_2, gate_ids)

    def test_analyze_single_sample(self):
        wsp_path = "examples/data/8_color_data_set/8_color_ICS_simple.wsp"
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'
        sample_grp = 'DEN'

        fks = Session(copy.deepcopy(test_samples_8c_full_set))
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=True)

        sample_ids = fks.get_group_sample_ids(sample_grp)
        self.assertEqual(len(sample_ids), 3)

        fks.analyze_samples(sample_grp, sample_id=sample_id)
        report = fks.get_group_report(sample_grp)

        self.assertEqual(report['sample'].nunique(), 1)

    def test_parse_wsp_sample_without_gates(self):
        wsp_path = "examples/data/8_color_data_set/8_color_ICS_sample_without_gates.wsp"
        sample_id = '101_DEN084Y5_15_E03_009_clean.fcs'
        sample_grp = 'DEN'

        fks = Session(copy.deepcopy(test_samples_8c_full_set))
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=False)

        sample_ids = fks.get_group_sample_ids(sample_grp)

        # there are technically 3 samples in the workspace 'DEN' group,
        # but one sample has no gates. FlowKit ignores these gate-less
        # samples because all samples in a FK sample group must have the
        # same gate tree. So there should be 2 samples found here.
        self.assertEqual(len(sample_ids), 2)

        fks.analyze_samples(sample_grp, sample_id=sample_id)
        results = fks.get_gating_results(sample_grp, sample_id)
        time_count = results.get_gate_count('Time')
        self.assertEqual(time_count, 257482)

    def test_extract_sample_data(self):
        wsp_path = "examples/data/8_color_data_set/8_color_ICS.wsp"
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'

        sample_data = extract_wsp_sample_data(wsp_path)

        self.assertIsInstance(sample_data, dict)
        self.assertIn(sample_id, sample_data)

        sample_id_data = sample_data[sample_id]

        self.assertIn('keywords', sample_id_data)

        sample_keywords = sample_id_data['keywords']
        sample_keyword_count = len(sample_keywords)

        self.assertGreaterEqual(sample_keyword_count, 0)

    def test_export_wsp(self):
        wsp_path = "examples/data/8_color_data_set/8_color_ICS.wsp"
        sample_grp = 'DEN'

        # use a leaf gate to test if the new WSP session is created correctly
        gate_name = 'TNFa+'
        gate_path = ('root', 'Time', 'Singlets', 'aAmine-', 'CD3+', 'CD4+')

        fks = Session(copy.deepcopy(test_samples_8c_full_set))
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=True)

        out_file = BytesIO()
        fks.export_wsp(out_file, sample_grp)
        out_file.seek(0)

        fks_out = Session(copy.deepcopy(test_samples_8c_full_set))
        fks_out.import_flowjo_workspace(out_file, ignore_missing_files=True)

        self.assertIsInstance(fks_out, Session)

        fks_gate = fks.get_gate(sample_grp, gate_name, gate_path)
        fks_out_gate = fks_out.get_gate(sample_grp, gate_name, gate_path)

        self.assertIsInstance(fks_gate, gates.RectangleGate)
        self.assertIsInstance(fks_out_gate, gates.RectangleGate)

        self.assertEqual(fks_gate.gate_name, gate_name)
        self.assertEqual(fks_out_gate.gate_name, gate_name)
