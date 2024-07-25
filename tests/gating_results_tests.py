"""
Tests for GatingResults class
"""
import copy
import unittest
from flowkit import Workspace
from tests.test_config import test_samples_8c_full_set

wsp_path = "data/8_color_data_set/reused_quad_gate_with_child.wsp"
group_name = 'All Samples'
sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'

fks = Workspace(wsp_path, fcs_samples=copy.deepcopy(test_samples_8c_full_set), ignore_missing_files=True)
fks.analyze_samples(group_name=group_name, sample_id=sample_id)
results_8c_sample_008 = fks.get_gating_results(sample_id=sample_id)


class GatingResultsTestCase(unittest.TestCase):
    def test_get_gate_count_ambiguous_raises_value_error(self):
        gate_name = 'some_child_gate'

        self.assertRaises(ValueError, results_8c_sample_008.get_gate_count, gate_name)

    def test_get_gate_count_with_gate_path(self):
        gate_name = 'some_child_gate'
        gate_path_1 = ('root', 'good cells', 'cd4+', 'Q2: CD107a+, IL2+')

        gate_count = results_8c_sample_008.get_gate_count(gate_name=gate_name, gate_path=gate_path_1)

        self.assertEqual(gate_count, 558)

    def test_get_gate_relative_percent_with_gate_path(self):
        gate_name = 'some_child_gate'
        gate_path_2 = ('root', 'good cells', 'cd8+', 'Q2: CD107a+, IL2+')

        gate_rel_pct = results_8c_sample_008.get_gate_relative_percent(gate_name=gate_name, gate_path=gate_path_2)

        self.assertAlmostEqual(gate_rel_pct, 21.58845, 5)

    def test_get_gate_absolute_percent_with_gate_path(self):
        gate_name = 'some_child_gate'
        gate_path_2 = ('root', 'good cells', 'cd8+', 'Q2: CD107a+, IL2+')

        gate_abs_pct = results_8c_sample_008.get_gate_absolute_percent(gate_name=gate_name, gate_path=gate_path_2)

        self.assertAlmostEqual(gate_abs_pct, 0.10304, 5)
