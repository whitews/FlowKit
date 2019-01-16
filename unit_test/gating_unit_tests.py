import unittest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath('..'))

from flowkit import Sample, GatingStrategy


class RangeGateTestCase(unittest.TestCase):
    @staticmethod
    def test_min_range_gate():
        gml_path = '/home/swhite/git/flowkit/examples/gate_ref/gml_range_gate.xml'
        fcs_path = '/home/swhite/git/flowkit/examples/gate_ref/data1.fcs'
        res_path = '/home/swhite/git/flowkit/examples/gate_ref/Results_Range1.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'Range1')

        np.testing.assert_array_equal(truth, result['Range1'])

    @staticmethod
    def test_rect1_gate():
        gml_path = '/home/swhite/git/flowkit/examples/gate_ref/gml_rect1_gate.xml'
        fcs_path = '/home/swhite/git/flowkit/examples/gate_ref/data1.fcs'
        res_path = '/home/swhite/git/flowkit/examples/gate_ref/Results_Rectangle1.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'Rectangle1')

        np.testing.assert_array_equal(truth, result['Rectangle1'])

    @staticmethod
    def test_rect2_gate():
        gml_path = '/home/swhite/git/flowkit/examples/gate_ref/gml_rect2_gate.xml'
        fcs_path = '/home/swhite/git/flowkit/examples/gate_ref/data1.fcs'
        res_path = '/home/swhite/git/flowkit/examples/gate_ref/Results_Rectangle2.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'Rectangle2')

        np.testing.assert_array_equal(truth, result['Rectangle2'])
