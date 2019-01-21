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
    def test_poly2_gate():
        gml_path = 'examples/gate_ref/gml_poly2_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/Results_Polygon2.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'Polygon2')

        np.testing.assert_array_equal(truth, result['Polygon2'])

    @staticmethod
    def test_poly3_non_solid_gate():
        gml_path = 'examples/gate_ref/gml_poly3ns_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/Results_Polygon3NS.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'Polygon3NS')

        np.testing.assert_array_equal(truth, result['Polygon3NS'])

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

    @staticmethod
    def test_quadrant1_gate():
        gml_path = 'examples/gate_ref/gml_quadrant1_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res1_path = 'examples/gate_ref/Results_FL2N-FL4N.txt'
        res2_path = 'examples/gate_ref/Results_FL2N-FL4P.txt'
        res3_path = 'examples/gate_ref/Results_FL2P-FL4N.txt'
        res4_path = 'examples/gate_ref/Results_FL2P-FL4P.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth1 = np.loadtxt(res1_path, dtype=np.bool)
        truth2 = np.loadtxt(res2_path, dtype=np.bool)
        truth3 = np.loadtxt(res3_path, dtype=np.bool)
        truth4 = np.loadtxt(res4_path, dtype=np.bool)

        result = gs.gate_sample(sample)

        np.testing.assert_array_equal(truth1, result['Quadrant1']['FL2N-FL4N'])
        np.testing.assert_array_equal(truth2, result['Quadrant1']['FL2N-FL4P'])
        np.testing.assert_array_equal(truth3, result['Quadrant1']['FL2P-FL4N'])
        np.testing.assert_array_equal(truth4, result['Quadrant1']['FL2P-FL4P'])
