"""
Workspace tests
"""
import copy
import unittest
import numpy as np
import os
import pandas as pd
import warnings
from flowkit import Workspace, Sample, Matrix, gates, transforms, extract_wsp_sample_data
# noinspection PyProtectedMember
from flowkit._models.transforms._base_transform import Transform
from flowkit._models.gating_results import GatingResults
from flowkit.exceptions import GateReferenceError, GateTreeError

from tests.test_config import test_samples_8c_full_set


wsp_8_color = Workspace(
    "data/8_color_data_set/8_color_ICS.wsp",
    fcs_samples=test_samples_8c_full_set
)
wsp_8_color_no_files = Workspace(
    "data/8_color_data_set/8_color_ICS.wsp",
    ignore_missing_files=True
)


class WorkspaceTestCase(unittest.TestCase):
    def setUp(self):
        """
        Setup data for WorkspaceTestCase
        :return: None
        """
        self.wsp_8_color = wsp_8_color
        self.wsp_8_color_no_files = wsp_8_color_no_files

    """Tests for Workspace class"""
    def test_workspace_summary(self):
        sample_grp = 'DEN'

        wsp = self.wsp_8_color_no_files
        wsp_summary = wsp.summary()

        self.assertIsInstance(wsp_summary, pd.DataFrame)

        group_stats = wsp_summary.loc[sample_grp]
        self.assertEqual(group_stats.max_gate_depth, 6)
        self.assertEqual(group_stats.samples, 3)
        self.assertEqual(group_stats.loaded_samples, 0)

    def test_workspace_summary_with_loaded_samples(self):
        wsp = self.wsp_8_color
        wsp_summary = wsp.summary()
        sample_grp = 'DEN'

        self.assertIsInstance(wsp_summary, pd.DataFrame)

        group_stats = wsp_summary.loc[sample_grp]
        self.assertEqual(group_stats.max_gate_depth, 6)
        self.assertEqual(group_stats.samples, 3)
        self.assertEqual(group_stats.loaded_samples, 3)

    def test_get_sample_groups(self):
        wsp_path = "data/simple_line_example/simple_poly_and_rect.wsp"
        fcs_path = "data/simple_line_example/data_set_simple_line_100.fcs"

        wsp = Workspace(wsp_path, fcs_samples=fcs_path)

        groups = wsp.get_sample_groups()
        groups_truth = ['All Samples', 'my_group']

        self.assertListEqual(groups, groups_truth)

    def test_get_sample_ids(self):
        wsp = self.wsp_8_color
        loaded_sample_ids = wsp.get_sample_ids()

        ground_truth = [
            '101_DEN084Y5_15_E01_008_clean.fcs',
            '101_DEN084Y5_15_E03_009_clean.fcs',
            '101_DEN084Y5_15_E05_010_clean.fcs'
        ]

        self.assertListEqual(loaded_sample_ids, ground_truth)

    def test_get_sample_ids_missing_sample(self):
        wsp_path = "data/8_color_data_set/8_color_ICS.wsp"
        fcs_set_missing_sample = test_samples_8c_full_set[:-1]

        wsp = Workspace(
            wsp_path,
            fcs_samples=fcs_set_missing_sample,
            ignore_missing_files=True  # need to ignore the missing sample
        )
        loaded_sample_ids = wsp.get_sample_ids()
        all_sample_ids = wsp.get_sample_ids(loaded_only=False)

        loaded_ground_truth = [
            '101_DEN084Y5_15_E01_008_clean.fcs',
            '101_DEN084Y5_15_E03_009_clean.fcs'
        ]
        all_ground_truth = [
            '101_DEN084Y5_15_E01_008_clean.fcs',
            '101_DEN084Y5_15_E03_009_clean.fcs',
            '101_DEN084Y5_15_E05_010_clean.fcs'
        ]

        self.assertListEqual(loaded_sample_ids, loaded_ground_truth)
        self.assertListEqual(all_sample_ids, all_ground_truth)

    def test_get_samples(self):
        wsp = self.wsp_8_color
        loaded_samples = wsp.get_samples()

        self.assertEqual(len(loaded_samples), 3)
        self.assertIsInstance(loaded_samples[0], Sample)

    def test_get_keywords_by_sample_id(self):
        wsp_path = "data/8_color_data_set/8_color_ICS_simple.wsp"
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'

        wsp = Workspace(
            wsp_path,
            fcs_samples=copy.deepcopy(test_samples_8c_full_set),
            ignore_missing_files=True
        )

        keywords = wsp.get_keywords(sample_id)

        self.assertIsInstance(keywords, dict)

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

        xform_lut = wsp.get_transforms(sample_id)

        self.assertEqual(len(xform_lut), 23)
        for xform in xform_lut.values():
            self.assertIsInstance(xform, Transform)

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

    def test_get_gate_events(self):
        wsp_path = "data/8_color_data_set/8_color_ICS_simple.wsp"
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'
        gate_name = 'CD3+'

        wsp = Workspace(
            wsp_path,
            fcs_samples=copy.deepcopy(test_samples_8c_full_set),
            ignore_missing_files=True
        )

        wsp.analyze_samples(sample_id=sample_id)

        # get auto-processed
        df_gate_events = wsp.get_gate_events(
            sample_id,
            gate_name
        )

        # get raw
        df_gate_events_raw = wsp.get_gate_events(
            sample_id,
            gate_name,
            source='row'
        )

        # get comp
        df_gate_events_comp = wsp.get_gate_events(
            sample_id,
            gate_name,
            source='comp'
        )

        evt_6_raw = [
            165875.51562, 136158.00000, 79839.72656,
            30412.32031, 29198.00000, 68261.58594,
            146.88000, 68.34000, 145.08000, 126.48000,
            61.38000, 1475.09998, 393.80002, 3631.10010,
            1.28000
        ]
        evt_6_comp = [
            165875.51562, 136158.00000, 79839.72656,
            30412.32031, 29198.00000, 68261.58594,
            145.23488, -3.66970, 145.08000, 126.48000,
            47.19370, 1258.81096, 263.90823, 3468.27668,
            1.28000
        ]
        evt_6_xform = [
            0.63276, 0.51940, 0.30456, 0.11601, 0.11138,
            0.26040, 0.25399, 0.22562, 0.25396, 0.25044,
            0.23534, 0.41934, 0.27620, 0.54810, 0.03635
        ]
        chan_cols = [
            'FSC-A', 'FSC-H', 'FSC-W', 'SSC-A', 'SSC-H', 'SSC-W',
            'TNFa FITC FLR-A', 'CD8 PerCP-Cy55 FLR-A', 'IL2 BV421 FLR-A',
            'Aqua Amine FLR-A', 'IFNg APC FLR-A', 'CD3 APC-H7 FLR-A',
            'CD107a PE FLR-A', 'CD4 PE-Cy7 FLR-A', 'Time'
        ]

        self.assertIsInstance(df_gate_events, pd.DataFrame)
        self.assertEqual(len(df_gate_events), 133670)

        np.testing.assert_almost_equal(df_gate_events_raw[chan_cols].loc[6].values, evt_6_raw, 5)
        np.testing.assert_almost_equal(df_gate_events_comp[chan_cols].loc[6].values, evt_6_comp, 5)
        np.testing.assert_almost_equal(df_gate_events[chan_cols].loc[6].values, evt_6_xform, 5)

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

        wsp = Workspace(wsp_path, fcs_samples=fcs_path)

        self.assertIsInstance(
            wsp.get_gate(
                sample_id='data_set_simple_line_100.fcs',
                gate_name='ellipse1'
            ),
            gates.PolygonGate
        )

        wsp.analyze_samples(group_name='All Samples')
        results = wsp.get_gating_results(sample_id='data_set_simple_line_100.fcs')
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

        biex_xform = transforms.WSPBiexTransform(negative=neg, width=width)

        test_y = biex_xform.apply(x)

        mean_pct_diff = 100. * np.mean(np.abs(test_y[1:] - y[1:]) / y[1:])
        self.assertLess(mean_pct_diff, 0.01)

    def test_parse_wsp_with_ellipse(self):
        wsp_path = "data/8_color_data_set/8_color_ICS_with_ellipse.wsp"
        fcs_path = "data/8_color_data_set/fcs_files/101_DEN084Y5_15_E01_008_clean.fcs"
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'
        gate_name = 'ellipse1'
        gate_path = ('root', 'Time', 'Singlets', 'aAmine-', 'CD3+')

        wsp = Workspace(wsp_path, fcs_samples=fcs_path, ignore_missing_files=True)

        wsp.analyze_samples(sample_id=sample_id)
        gate_indices = wsp.get_gate_membership(sample_id, gate_name, gate_path=gate_path)

        self.assertIsInstance(gate_indices, np.ndarray)
        self.assertEqual(np.sum(gate_indices), 7023)

    def test_parse_wsp_with_invalid_gate_name(self):
        wsp_path = "data/8_color_data_set/8_color_ICS_dot_gate_name.wsp"

        self.assertRaisesRegex(
            GateTreeError,
            r"Gate name '\.' is incompatible with FlowKit\. " 
            r"Gate was found in path: \/root\/Time\/Singlets\/aAmine-\/CD3\+\/CD4\+",
            Workspace,
            wsp_path,
            ignore_missing_files=True
        )

    def test_parse_wsp_with_boolean_gates(self):
        wsp_path = "data/8_color_data_set/8_color_ICS_boolean_gate_testing.wsp"
        fcs_path = "data/8_color_data_set/fcs_files/101_DEN084Y5_15_E01_008_clean.fcs"
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'

        # Boolean AND gate in CD4+ branch
        gate_name_01 = 'CD107a+ & IFNg+'
        gate_path_01 = ('root', 'Time', 'Singlets', 'aAmine-', 'CD3+', 'CD4+')

        # Boolean OR gate in CD4+ branch
        gate_name_02 = 'CD107a+ or IFNg+'
        gate_path_02 = ('root', 'Time', 'Singlets', 'aAmine-', 'CD3+', 'CD4+')

        # Boolean NOT gate in CD4+ branch
        gate_name_03 = 'IL2+-'
        gate_path_03 = ('root', 'Time', 'Singlets', 'aAmine-', 'CD3+', 'CD4+')

        # Boolean NOT gate in CD4+ branch
        gate_name_04 = 'TNFa+-'
        gate_path_04 = ('root', 'Time', 'Singlets', 'aAmine-', 'CD3+', 'CD4+')

        # Boolean NOT gate in CD8+ branch
        gate_name_05 = 'TNFa+-'
        gate_path_05 = ('root', 'Time', 'Singlets', 'aAmine-', 'CD3+', 'CD8+')

        # Child of a Boolean gate in CD8+ branch
        gate_name_06 = 'CD107a+'
        gate_path_06 = ('root', 'Time', 'Singlets', 'aAmine-', 'CD3+', 'CD8+', 'TNFa+-')

        wsp = Workspace(wsp_path, fcs_samples=fcs_path, ignore_missing_files=True)

        wsp.analyze_samples(sample_id=sample_id)
        gating_results = wsp.get_gating_results(sample_id)

        gate_count_01 = gating_results.get_gate_count(gate_name_01, gate_path_01)
        gate_count_02 = gating_results.get_gate_count(gate_name_02, gate_path_02)
        gate_count_03 = gating_results.get_gate_count(gate_name_03, gate_path_03)
        gate_count_04 = gating_results.get_gate_count(gate_name_04, gate_path_04)
        gate_count_05 = gating_results.get_gate_count(gate_name_05, gate_path_05)
        gate_count_06 = gating_results.get_gate_count(gate_name_06, gate_path_06)

        self.assertIsInstance(gating_results, GatingResults)
        self.assertEqual(gate_count_01, 0)
        self.assertEqual(gate_count_02, 72)
        self.assertEqual(gate_count_03, 82478)
        self.assertEqual(gate_count_04, 82463)
        self.assertEqual(gate_count_05, 47157)
        self.assertEqual(gate_count_06, 70)

    def test_get_ambiguous_gate_objects(self):
        wsp_path = "data/8_color_data_set/8_color_ICS.wsp"
        fcs_path = "data/8_color_data_set/fcs_files/101_DEN084Y5_15_E01_008_clean.fcs"
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'
        gate_name = 'TNFa+'
        gate_path = ('root', 'Time', 'Singlets', 'aAmine-', 'CD3+', 'CD4+')

        wsp = Workspace(wsp_path, fcs_samples=fcs_path, ignore_missing_files=True)

        wsp.analyze_samples(sample_id=sample_id)
        gate_indices = wsp.get_gate_membership(sample_id, gate_name, gate_path=gate_path)

        self.assertIsInstance(gate_indices, np.ndarray)
        self.assertEqual(np.sum(gate_indices), 21)

    def test_ambiguous_gate_raises_in_get_child_gate_ids(self):
        wsp_path = "data/8_color_data_set/8_color_ICS.wsp"
        group_name = 'DEN'
        gate_name = 'IFNg+'

        wsp = Workspace(wsp_path, ignore_missing_files=True)
        sample_ids = wsp.get_sample_ids(group_name, loaded_only=False)

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

        # there are technically 3 samples in the workspace 'DEN' group,
        # but one sample has no gates.
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
