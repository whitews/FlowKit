"""
Tests for custom gates in the GatingStrategy Class
"""
import copy
import os
import sys
import unittest
import flowkit as fk

sys.path.append(os.path.abspath('../..'))


class GatingStrategyCustomGatesTestCase(unittest.TestCase):
    def setUp(self):
        """
        Tests for custom gates in the GatingStrategy class
        :return: None
        """
        self.gs = fk.parse_gating_xml('data/8_color_data_set/8_color_ICS.xml')
        self.samples = fk.load_samples("data/8_color_data_set/fcs_files")

    def test_is_custom_gate(self):
        """Test for whether a gate is a custom gate"""
        gs = copy.deepcopy(self.gs)
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'
        time_gate_name = 'TimeGate'  # we'll make a custom gate
        cd3_gate_name = 'CD3-pos'  # has no custom gates

        time_gate = self.gs.get_gate(time_gate_name)
        time_gate_008 = copy.deepcopy(time_gate)

        time_dim_008 = time_gate_008.get_dimension('Time')
        time_dim_008.min = 0.10
        time_dim_008.max = 0.90

        gs.add_gate(time_gate_008, ('root',), sample_id=sample_id)

        not_custom_gate = gs.is_custom_gate('some_other_id', time_gate_name)
        is_custom_gate = gs.is_custom_gate(sample_id, time_gate_name)
        also_not_custom_gate = gs.is_custom_gate(sample_id, cd3_gate_name)

        self.assertTrue(is_custom_gate)
        self.assertFalse(not_custom_gate)
        self.assertFalse(also_not_custom_gate)

    def test_get_custom_gate(self):
        """Test get_gate for a custom gate"""
        gs = copy.deepcopy(self.gs)
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'
        time_gate_name = 'TimeGate'

        time_gate = self.gs.get_gate(time_gate_name)
        time_gate_008 = copy.deepcopy(time_gate)

        time_dim_008 = time_gate_008.get_dimension('Time')
        time_dim_008.min = 0.10
        time_dim_008.max = 0.90

        gs.add_gate(time_gate_008, ('root',), sample_id=sample_id)

        custom_time_gate = gs.get_gate(time_gate_name, sample_id=sample_id)
        custom_time_dim = custom_time_gate.get_dimension('Time')

        self.assertEqual(custom_time_dim.min, 0.10)
        self.assertEqual(custom_time_dim.max, 0.90)
