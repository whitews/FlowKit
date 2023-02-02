"""
Workspace tests
"""
import copy
import unittest
import numpy as np
import os
import pandas as pd
import warnings
from flowkit import Workspace, load_samples, Matrix, gates, transforms, extract_wsp_sample_data
# noinspection PyProtectedMember
from flowkit._models.transforms._base_transform import Transform
from flowkit.exceptions import GateReferenceError


test_samples_8c_full_set = load_samples("data/8_color_data_set/fcs_files")


class WorkspaceTestCase(unittest.TestCase):
    """Tests for Workspace class"""
    def test_workspace_summary(self):
        wsp_path = "data/8_color_data_set/8_color_ICS.wsp"
        sample_grp = 'DEN'

        wsp = Workspace(wsp_path, ignore_missing_files=True)
        wsp_summary = wsp.summary()

        self.assertIsInstance(wsp_summary, pd.DataFrame)

        group_stats = wsp_summary.loc[sample_grp]
        self.assertEqual(group_stats.max_gate_depth, 6)
        self.assertEqual(group_stats.samples, 3)
        self.assertEqual(group_stats.loaded_samples, 0)

    def test_get_comp_matrix_by_sample_id(self):
        wsp_path = "data/8_color_data_set/8_color_ICS_simple.wsp"
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'

        wsp = Workspace(
            wsp_path,
            fcs_samples=copy.deepcopy(test_samples_8c_full_set),
            ignore_missing_files=True
        )

        comp_matrix = wsp.get_comp_matrix(sample_id)

        self.assertIsInstance(comp_matrix, Matrix)

    def test_get_transforms_by_sample_id(self):
        wsp_path = "data/8_color_data_set/8_color_ICS_simple.wsp"
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'

        wsp = Workspace(
            wsp_path,
            fcs_samples=copy.deepcopy(test_samples_8c_full_set),
            ignore_missing_files=True
        )

        xforms = wsp.get_transforms(sample_id)

        self.assertEqual(len(xforms), 23)
        for cm in xforms:
            self.assertIsInstance(cm, Transform)

    def test_get_child_gate_ids(self):
        wsp_path = "data/8_color_data_set/8_color_ICS.wsp"
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'
        gate_name = 'CD3+'

        wsp = Workspace(
            wsp_path,
            fcs_samples=copy.deepcopy(test_samples_8c_full_set),
            ignore_missing_files=True
        )

        child_gate_ids = wsp.get_child_gate_ids(sample_id, gate_name)

        truth = [
            ('CD4+', ('root', 'Time', 'Singlets', 'aAmine-', 'CD3+')),
            ('CD8+', ('root', 'Time', 'Singlets', 'aAmine-', 'CD3+'))
        ]

        self.assertEqual(len(child_gate_ids), 2)
        for gate_name, gate_path in child_gate_ids:
            self.assertIn((gate_name, gate_path), truth)

    def test_find_matching_gate_paths(self):
        wsp_path = "data/8_color_data_set/8_color_ICS.wsp"
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'
        gate_name = 'IFNg+'

        wsp = Workspace(
            wsp_path,
            fcs_samples=copy.deepcopy(test_samples_8c_full_set),
            ignore_missing_files=True
        )
        gate_paths = wsp.find_matching_gate_paths(sample_id, gate_name)

        truth = [
            ('root', 'Time', 'Singlets', 'aAmine-', 'CD3+', 'CD4+'),
            ('root', 'Time', 'Singlets', 'aAmine-', 'CD3+', 'CD8+')
        ]

        self.assertEqual(len(gate_paths), 2)
        for gate_path in gate_paths:
            self.assertIn(gate_path, truth)

    def test_get_gated_events(self):
        wsp_path = "data/8_color_data_set/8_color_ICS_simple.wsp"
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'
        gate_name = 'CD3+'

        wsp = Workspace(
            wsp_path,
            fcs_samples=copy.deepcopy(test_samples_8c_full_set),
            ignore_missing_files=True
        )

        wsp.analyze_samples(sample_id=sample_id)

        df_gated_events = wsp.get_gate_events(
            sample_id,
            gate_name
        )

        self.assertIsInstance(df_gated_events, pd.DataFrame)
        self.assertEqual(len(df_gated_events), 133670)

    def test_load_wsp_single_poly(self):
        wsp_path = "data/simple_line_example/simple_poly_and_rect.wsp"
        fcs_path = "data/simple_line_example/data_set_simple_line_100.fcs"
        sample_id = 'data_set_simple_line_100.fcs'

        wsp = Workspace(wsp_path, fcs_samples=fcs_path)

        self.assertIsInstance(
            wsp.get_gate(
                sample_id,
                'poly1'
            ),
            gates.PolygonGate
        )

        gate_names = {'rect1', 'poly1'}
        wsp_gates_tuple = wsp.get_gate_ids(sample_id)
        wsp_gate_names = set([g[0] for g in wsp_gates_tuple])
        self.assertSetEqual(wsp_gate_names, gate_names)

    def test_load_wsp_single_ellipse(self):
        wsp_path = "data/simple_line_example/single_ellipse_51_events.wsp"
        fcs_path = "data/simple_line_example/data_set_simple_line_100.fcs"

        fks = Workspace(wsp_path, fcs_samples=fcs_path)

        self.assertIsInstance(
            fks.get_gate(
                sample_id='data_set_simple_line_100.fcs',
                gate_name='ellipse1'
            ),
            gates.PolygonGate
        )

        fks.analyze_samples(group_name='All Samples')
        results = fks.get_gating_results(sample_id='data_set_simple_line_100.fcs')
        gate_count = results.get_gate_count('ellipse1')
        self.assertEqual(gate_count, 51)

    def test_load_wsp_single_quad(self):
        wsp_path = "data/simple_diamond_example/simple_diamond_example_quad_gate.wsp"
        fcs_path = "data/simple_diamond_example/test_data_diamond_01.fcs"

        wsp = Workspace(wsp_path, fcs_samples=fcs_path)

        # FlowJo quadrant gates are not true quadrant gates, rather a collection of rectangle gates
        self.assertIsInstance(
            wsp.get_gate(
                sample_id='test_data_diamond_01.fcs',
                gate_name='Q1: channel_A- , channel_B+'
            ),
            gates.RectangleGate
        )

        wsp.analyze_samples(group_name='All Samples')
        results = wsp.get_gating_results('test_data_diamond_01.fcs')

        gate_count_q1 = results.get_gate_count('Q1: channel_A- , channel_B+')
        gate_count_q2 = results.get_gate_count('Q2: channel_A+ , channel_B+')
        gate_count_q3 = results.get_gate_count('Q3: channel_A+ , channel_B-')
        gate_count_q4 = results.get_gate_count('Q4: channel_A- , channel_B-')
        self.assertEqual(gate_count_q1, 49671)
        self.assertEqual(gate_count_q2, 50596)
        self.assertEqual(gate_count_q3, 50330)
        self.assertEqual(gate_count_q4, 49403)

    def test_wsp_biex_transform(self):
        wsp_path = "data/simple_diamond_example/test_data_diamond_biex_rect.wsp"
        fcs_path = "data/simple_diamond_example/test_data_diamond_01.fcs"
        sample_id = "test_data_diamond_01.fcs"

        wsp = Workspace(wsp_path, fcs_samples=fcs_path)

        self.assertIsInstance(
            wsp.get_gate(
                sample_id,
                'upper_right'
            ),
            gates.RectangleGate
        )

        wsp.analyze_samples(sample_id=sample_id)
        results = wsp.get_gating_results(sample_id=sample_id)
        gate_count = results.get_gate_count('upper_right')
        self.assertEqual(gate_count, 50605)

    def test_wsp_fasinh_transform(self):
        wsp_path = "data/simple_diamond_example/test_data_diamond_asinh_rect.wsp"
        fcs_path = "data/simple_diamond_example/test_data_diamond_01.fcs"
        sample_id = "test_data_diamond_01.fcs"

        wsp = Workspace(wsp_path, fcs_samples=fcs_path)

        self.assertIsInstance(
            wsp.get_gate(
                sample_id,
                'upper_right'
            ),
            gates.RectangleGate
        )

        wsp.analyze_samples(sample_id=sample_id)
        results = wsp.get_gating_results(sample_id=sample_id)
        gate_count = results.get_gate_count('upper_right')
        self.assertEqual(gate_count, 50559)

    def test_wsp_fasinh_transform_v2(self):
        wsp_path = "data/simple_diamond_example/test_data_diamond_asinh_rect2.wsp"
        fcs_path = "data/simple_diamond_example/test_data_diamond_01.fcs"
        sample_id = 'test_data_diamond_01.fcs'

        wsp = Workspace(wsp_path, fcs_samples=fcs_path)

        self.assertIsInstance(
            wsp.get_gate(
                sample_id,
                'upper_right',
            ),
            gates.RectangleGate
        )

        wsp.analyze_samples(sample_id=sample_id)
        results = wsp.get_gating_results(sample_id=sample_id)
        gate_count = results.get_gate_count('upper_right')
        self.assertEqual(gate_count, 50699)

    def test_wsp_biex_transform_width_interpolation(self):
        neg = 1.0
        width = -7.943282

        # this LUT exists for only the single negative value of 1.0
        lut_file_name = "tr_biex_l256_w%.6f_n%.6f_m4.418540_r262144.000029.csv" % (width, neg)
        lut_file_path = os.path.join('data', 'flowjo_xforms', lut_file_name)
        y, x = np.loadtxt(lut_file_path, delimiter=',', usecols=(0, 1), skiprows=1, unpack=True)

        biex_xform = transforms.WSPBiexTransform('biex', negative=neg, width=width)

        test_y = biex_xform.apply(x)

        mean_pct_diff = 100. * np.mean(np.abs(test_y[1:] - y[1:]) / y[1:])
        self.assertLess(mean_pct_diff, 0.01)

    def test_get_sample_groups(self):
        wsp_path = "data/simple_line_example/simple_poly_and_rect.wsp"
        fcs_path = "data/simple_line_example/data_set_simple_line_100.fcs"

        fks = Workspace(wsp_path, fcs_samples=fcs_path)

        groups = fks.get_sample_groups()
        groups_truth = ['All Samples', 'my_group']

        self.assertListEqual(groups, groups_truth)

    def test_parse_wsp_with_ellipse(self):
        wsp_path = "data/8_color_data_set/8_color_ICS_with_ellipse.wsp"
        fcs_path = "data/8_color_data_set/fcs_files/101_DEN084Y5_15_E01_008_clean.fcs"
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'
        gate_name = 'ellipse1'
        gate_path = ('root', 'Time', 'Singlets', 'aAmine-', 'CD3+')

        fks = Workspace(wsp_path, fcs_samples=fcs_path, ignore_missing_files=True)

        fks.analyze_samples(sample_id=sample_id)
        gate_indices = fks.get_gate_membership(sample_id, gate_name, gate_path=gate_path)

        self.assertIsInstance(gate_indices, np.ndarray)
        self.assertEqual(np.sum(gate_indices), 7023)

    def test_get_ambiguous_gate_objects(self):
        wsp_path = "data/8_color_data_set/8_color_ICS.wsp"
        fcs_path = "data/8_color_data_set/fcs_files/101_DEN084Y5_15_E01_008_clean.fcs"
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'
        gate_name = 'TNFa+'
        gate_path = ('root', 'Time', 'Singlets', 'aAmine-', 'CD3+', 'CD4+')

        fks = Workspace(wsp_path, fcs_samples=fcs_path, ignore_missing_files=True)

        fks.analyze_samples(sample_id=sample_id)
        gate_indices = fks.get_gate_membership(sample_id, gate_name, gate_path=gate_path)

        self.assertIsInstance(gate_indices, np.ndarray)
        self.assertEqual(np.sum(gate_indices), 21)

    def test_ambiguous_gate_raises_in_get_child_gate_ids(self):
        wsp_path = "data/8_color_data_set/8_color_ICS.wsp"
        group_name = 'DEN'
        gate_name = 'IFNg+'

        wsp = Workspace(wsp_path, ignore_missing_files=True)
        sample_ids = wsp.get_sample_ids(group_name)

        self.assertRaises(GateReferenceError, wsp.get_child_gate_ids, sample_ids[0], gate_name)

    def test_parse_wsp_reused_gate_with_child(self):
        wsp_path = "data/8_color_data_set/reused_quad_gate_with_child.wsp"

        wsp = Workspace(wsp_path, fcs_samples=copy.deepcopy(test_samples_8c_full_set), ignore_missing_files=True)
        group_name = 'All Samples'
        gate_name = 'some_child_gate'

        sample_ids = wsp.get_sample_ids(group_name)

        gate_ids = wsp.get_gate_ids(sample_id=sample_ids[0])

        gate_id_1 = (gate_name, ('root', 'good cells', 'cd4+', 'Q2: CD107a+, IL2+'))
        gate_id_2 = (gate_name, ('root', 'good cells', 'cd8+', 'Q2: CD107a+, IL2+'))

        self.assertIn(gate_id_1, gate_ids)
        self.assertIn(gate_id_2, gate_ids)

    def test_analyze_single_sample(self):
        wsp_path = "data/8_color_data_set/8_color_ICS_simple.wsp"
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'
        sample_grp = 'DEN'

        wsp = Workspace(wsp_path, fcs_samples=copy.deepcopy(test_samples_8c_full_set))

        sample_ids = wsp.get_sample_ids(group_name=sample_grp)
        self.assertEqual(len(sample_ids), 3)

        wsp.analyze_samples(sample_id=sample_id)
        report = wsp.get_analysis_report(group_name=sample_grp)

        self.assertEqual(report['sample'].nunique(), 1)

    def test_parse_wsp_sample_without_gates(self):
        wsp_path = "data/8_color_data_set/8_color_ICS_sample_without_gates.wsp"
        sample_id = '101_DEN084Y5_15_E03_009_clean.fcs'
        sample_grp = 'DEN'

        wsp = Workspace(
            wsp_path, 
            fcs_samples=copy.deepcopy(test_samples_8c_full_set), 
            ignore_missing_files=False
        )

        sample_ids = wsp.get_sample_ids(group_name=sample_grp)
        # TODO: determine whether get_sample_ids should return loaded or all samples

        # there are technically 3 samples in the workspace 'DEN' group,
        # but one sample has no gates. The Workspace class will still
        # have the reference to all 3.
        self.assertEqual(len(sample_ids), 3)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            wsp.analyze_samples(group_name=sample_grp)  # sample_id=sample_id)

        results = wsp.get_gating_results(sample_id=sample_id)
        time_count = results.get_gate_count('Time')
        self.assertEqual(time_count, 257482)

        report = wsp.get_analysis_report(group_name=sample_grp)
        self.assertEqual(2, len(report['sample'].unique()))

    def test_extract_sample_data(self):
        wsp_path = "data/8_color_data_set/8_color_ICS.wsp"
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'

        sample_data = extract_wsp_sample_data(wsp_path)

        self.assertIsInstance(sample_data, dict)
        self.assertIn(sample_id, sample_data)

        sample_id_data = sample_data[sample_id]

        self.assertIn('keywords', sample_id_data)

        sample_keywords = sample_id_data['keywords']
        sample_keyword_count = len(sample_keywords)

        self.assertGreaterEqual(sample_keyword_count, 0)
