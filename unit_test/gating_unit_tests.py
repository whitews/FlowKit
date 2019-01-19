import unittest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath('..'))

from flowkit import Sample, GatingStrategy


class RangeGateTestCase(unittest.TestCase):
    @staticmethod
    def test_min_range_gate():
        gml_path = 'examples/gate_ref/gml_range_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/Results_Range1.txt'

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
        gml_path = 'examples/gate_ref/gml_rect1_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/Results_Rectangle1.txt'

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
        gml_path = 'examples/gate_ref/gml_rect2_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/Results_Rectangle2.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'Rectangle2')

        np.testing.assert_array_equal(truth, result['Rectangle2'])

    @staticmethod
    def test_poly1_gate():
        gml_path = 'examples/gate_ref/gml_poly1_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/Results_Polygon1.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'Polygon1')

        np.testing.assert_array_equal(truth, result['Polygon1'])

    @staticmethod
    def test_ellipse1_gate():
        gml_path = 'examples/gate_ref/gml_ellipse1_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/Results_Ellipse1.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'Ellipse1')

        np.testing.assert_array_equal(truth, result['Ellipse1'])

    @staticmethod
    def test_time_range_gate():
        gml_path = 'examples/gate_ref/gml_time_range_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/Results_Range2.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'Range2')

        np.testing.assert_array_equal(truth, result['Range2'])
