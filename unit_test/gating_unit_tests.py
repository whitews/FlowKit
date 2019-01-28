import unittest
import sys
import os
import glob
import re
import numpy as np

sys.path.append(os.path.abspath('..'))

from flowkit import Sample, GatingStrategy


class GatingMLTestCase(unittest.TestCase):
    @staticmethod
    def test_min_range_gate():
        gml_path = 'examples/gate_ref/gml/gml_range_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_Range1.txt'

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
        gml_path = 'examples/gate_ref/gml/gml_rect1_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_Rectangle1.txt'

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
        gml_path = 'examples/gate_ref/gml/gml_rect2_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_Rectangle2.txt'

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
        gml_path = 'examples/gate_ref/gml/gml_poly1_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_Polygon1.txt'

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
        gml_path = 'examples/gate_ref/gml/gml_poly2_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_Polygon2.txt'

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
        gml_path = 'examples/gate_ref/gml/gml_poly3ns_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_Polygon3NS.txt'

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
        gml_path = 'examples/gate_ref/gml/gml_ellipse1_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_Ellipse1.txt'

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
        gml_path = 'examples/gate_ref/gml/gml_time_range_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_Range2.txt'

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
        gml_path = 'examples/gate_ref/gml/gml_quadrant1_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res1_path = 'examples/gate_ref/truth/Results_FL2N-FL4N.txt'
        res2_path = 'examples/gate_ref/truth/Results_FL2N-FL4P.txt'
        res3_path = 'examples/gate_ref/truth/Results_FL2P-FL4N.txt'
        res4_path = 'examples/gate_ref/truth/Results_FL2P-FL4P.txt'

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

    @staticmethod
    def test_quadrant2_gate():
        gml_path = 'examples/gate_ref/gml/gml_quadrant2_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res1_path = 'examples/gate_ref/truth/Results_FSCN-SSCN.txt'
        res2_path = 'examples/gate_ref/truth/Results_FSCD-SSCN-FL1N.txt'
        res3_path = 'examples/gate_ref/truth/Results_FSCP-SSCN-FL1N.txt'
        res4_path = 'examples/gate_ref/truth/Results_FSCD-FL1P.txt'
        res5_path = 'examples/gate_ref/truth/Results_FSCN-SSCP-FL1P.txt'

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
        truth5 = np.loadtxt(res5_path, dtype=np.bool)

        result = gs.gate_sample(sample)

        np.testing.assert_array_equal(truth1, result['Quadrant2']['FSCN-SSCN'])
        np.testing.assert_array_equal(truth2, result['Quadrant2']['FSCD-SSCN-FL1N'])
        np.testing.assert_array_equal(truth3, result['Quadrant2']['FSCP-SSCN-FL1N'])
        np.testing.assert_array_equal(truth4, result['Quadrant2']['FSCD-FL1P'])
        np.testing.assert_array_equal(truth5, result['Quadrant2']['FSCN-SSCP-FL1P'])

    @staticmethod
    def test_ratio_range1_gate():
        gml_path = 'examples/gate_ref/gml/gml_ratio_range1_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_RatRange1.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'RatRange1')

        np.testing.assert_array_equal(truth, result['RatRange1'])

    @staticmethod
    def test_ratio_range2_gate():
        gml_path = 'examples/gate_ref/gml/gml_ratio_range2_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_RatRange2.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'RatRange2')

        np.testing.assert_array_equal(truth, result['RatRange2'])

    @staticmethod
    def test_log_ratio_range1_gate():
        gml_path = 'examples/gate_ref/gml/gml_log_ratio_range1_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_RatRange1a.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'RatRange1a')

        np.testing.assert_array_equal(truth, result['RatRange1a'])

    @staticmethod
    def test_boolean_and1_gate():
        gml_path = 'examples/gate_ref/gml/gml_boolean_and1_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_And1.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'And1')

        np.testing.assert_array_equal(truth, result['And1'])

    @staticmethod
    def test_boolean_and2_gate():
        gml_path = 'examples/gate_ref/gml/gml_boolean_and2_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_And2.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'And2')

        np.testing.assert_array_equal(truth, result['And2'])

    @staticmethod
    def test_boolean_or1_gate():
        gml_path = 'examples/gate_ref/gml/gml_boolean_or1_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_Or1.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'Or1')

        np.testing.assert_array_equal(truth, result['Or1'])

    @staticmethod
    def test_boolean_and3_complement_gate():
        gml_path = 'examples/gate_ref/gml/gml_boolean_and3_complement_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_And3.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'And3')

        np.testing.assert_array_equal(truth, result['And3'])

    @staticmethod
    def test_boolean_not1_gate():
        gml_path = 'examples/gate_ref/gml/gml_boolean_not1_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_Not1.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'Not1')

        np.testing.assert_array_equal(truth, result['Not1'])

    @staticmethod
    def test_boolean_and4_not_gate():
        gml_path = 'examples/gate_ref/gml/gml_boolean_and4_not_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_And4.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'And4')

        np.testing.assert_array_equal(truth, result['And4'])

    @staticmethod
    def test_boolean_or2_not_gate():
        gml_path = 'examples/gate_ref/gml/gml_boolean_or2_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_Or2.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'Or2')

        np.testing.assert_array_equal(truth, result['Or2'])

    @staticmethod
    def test_matrix_poly4_gate():
        gml_path = 'examples/gate_ref/gml/gml_matrix_poly4_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_Polygon4.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'Polygon4')

        np.testing.assert_array_equal(truth, result['Polygon4'])

    @staticmethod
    def test_matrix_rect3_gate():
        gml_path = 'examples/gate_ref/gml/gml_matrix_rect3_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_Rectangle3.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'Rectangle3')

        np.testing.assert_array_equal(truth, result['Rectangle3'])

    @staticmethod
    def test_matrix_rect4_gate():
        gml_path = 'examples/gate_ref/gml/gml_matrix_rect4_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_Rectangle4.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'Rectangle4')

        np.testing.assert_array_equal(truth, result['Rectangle4'])

    @staticmethod
    def test_matrix_rect5_gate():
        gml_path = 'examples/gate_ref/gml/gml_matrix_rect5_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_Rectangle5.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'Rectangle5')

        np.testing.assert_array_equal(truth, result['Rectangle5'])

    @staticmethod
    def test_transform_asinh_range1_gate():
        gml_path = 'examples/gate_ref/gml/gml_transform_asinh_range1_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange1.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'ScaleRange1')

        np.testing.assert_array_equal(truth, result['ScaleRange1'])

    @staticmethod
    def test_transform_hyperlog_range2_gate():
        gml_path = 'examples/gate_ref/gml/gml_transform_hyperlog_range2_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange2.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'ScaleRange2')

        np.testing.assert_array_equal(truth, result['ScaleRange2'])

    @staticmethod
    def test_transform_linear_range3_gate():
        gml_path = 'examples/gate_ref/gml/gml_transform_linear_range3_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange3.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'ScaleRange3')

        np.testing.assert_array_equal(truth, result['ScaleRange3'])

    @staticmethod
    def test_transform_logicle_range4_gate():
        gml_path = 'examples/gate_ref/gml/gml_transform_logicle_range4_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange4.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'ScaleRange4')

        np.testing.assert_array_equal(truth, result['ScaleRange4'])

    @staticmethod
    def test_transform_logicle_range5_gate():
        gml_path = 'examples/gate_ref/gml/gml_transform_logicle_range5_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange5.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'ScaleRange5')

        np.testing.assert_array_equal(truth, result['ScaleRange5'])

    @staticmethod
    def test_transform_log_range6_gate():
        gml_path = 'examples/gate_ref/gml/gml_transform_log_range6_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange6.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'ScaleRange6')

        np.testing.assert_array_equal(truth, result['ScaleRange6'])

    @staticmethod
    def test_matrix_transform_asinh_range1c_gate():
        gml_path = 'examples/gate_ref/gml/gml_matrix_transform_asinh_range1c_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange1c.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'ScaleRange1c')

        np.testing.assert_array_equal(truth, result['ScaleRange1c'])

    @staticmethod
    def test_matrix_transform_hyperlog_range2c_gate():
        gml_path = 'examples/gate_ref/gml/gml_matrix_transform_hyperlog_range2c_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange2c.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'ScaleRange2c')

        np.testing.assert_array_equal(truth, result['ScaleRange2c'])

    @staticmethod
    def test_matrix_transform_linear_range3c_gate():
        gml_path = 'examples/gate_ref/gml/gml_matrix_transform_linear_range3c_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange3c.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'ScaleRange3c')

        np.testing.assert_array_equal(truth, result['ScaleRange3c'])

    @staticmethod
    def test_matrix_transform_logicle_range4c_gate():
        gml_path = 'examples/gate_ref/gml/gml_matrix_transform_logicle_range4c_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange4c.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'ScaleRange4c')

        np.testing.assert_array_equal(truth, result['ScaleRange4c'])

    @staticmethod
    def test_matrix_transform_logicle_range5c_gate():
        gml_path = 'examples/gate_ref/gml/gml_matrix_transform_logicle_range5c_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange5c.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'ScaleRange5c')

        np.testing.assert_array_equal(truth, result['ScaleRange5c'])

    @staticmethod
    def test_matrix_transform_asinh_range6c_gate():
        gml_path = 'examples/gate_ref/gml/gml_matrix_transform_asinh_range6c_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange6c.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'ScaleRange6c')

        np.testing.assert_array_equal(truth, result['ScaleRange6c'])

    @staticmethod
    def test_matrix_transform_hyperlog_range7c_gate():
        gml_path = 'examples/gate_ref/gml/gml_matrix_transform_hyperlog_range7c_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange7c.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'ScaleRange7c')

        np.testing.assert_array_equal(truth, result['ScaleRange7c'])

    @staticmethod
    def test_matrix_transform_logicle_range8c_gate():
        gml_path = 'examples/gate_ref/gml/gml_matrix_transform_logicle_range8c_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange8c.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'ScaleRange8c')

        np.testing.assert_array_equal(truth, result['ScaleRange8c'])

    @staticmethod
    def test_matrix_transform_logicle_rect1_gate():
        gml_path = 'examples/gate_ref/gml/gml_matrix_transform_logicle_rect1_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_ScaleRect1.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'ScaleRect1')

        np.testing.assert_array_equal(truth, result['ScaleRect1'])

    @staticmethod
    def test_parent_poly1_boolean_and2_gate():
        gml_path = 'examples/gate_ref/gml/gml_parent_poly1_boolean_and2_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_ParAnd2.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'ParAnd2')

        np.testing.assert_array_equal(truth, result['ParAnd2'])

    @staticmethod
    def test_parent_range1_boolean_and3_gate():
        gml_path = 'examples/gate_ref/gml/gml_parent_range1_boolean_and3_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_ParAnd3.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'ParAnd3')

        np.testing.assert_array_equal(truth, result['ParAnd3'])

    @staticmethod
    def test_parent_rect1_rect_par1_gate():
        gml_path = 'examples/gate_ref/gml/gml_parent_rect1_rect_par1_gate.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        res_path = 'examples/gate_ref/truth/Results_ScalePar1.txt'

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        truth = np.loadtxt(res_path, dtype=np.bool)

        result = gs.gate_sample(sample, 'ScalePar1')

        np.testing.assert_array_equal(truth, result['ScalePar1'])

    @staticmethod
    def test_all_gates():
        gml_path = 'examples/gate_ref/gml/gml_all_gates.xml'
        fcs_path = 'examples/gate_ref/data1.fcs'
        truth_pattern = 'examples/gate_ref/truth/Results*.txt'

        res_files = glob.glob(truth_pattern)

        truth_dict = {}

        for res_path in res_files:
            match = re.search("Results_(.+)\\.txt$", res_path)
            if match is not None:
                g_id = match.group(1)
                truth = np.loadtxt(res_path, dtype=np.bool)

                truth_dict[g_id] = truth

        gs = GatingStrategy(gml_path)
        sample = Sample(
            fcs_path,
            filter_anomalous_events=False,
            filter_negative_scatter=False
        )
        gs_results = gs.gate_sample(sample)

        for g_id, res in gs_results.items():
            if isinstance(res, dict):
                for sub_g_id, sub_res in res.items():
                    np.testing.assert_array_equal(
                        truth_dict[sub_g_id],
                        sub_res
                    )
            else:
                np.testing.assert_array_equal(
                    truth_dict[g_id],
                    res
                )
