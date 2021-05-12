import unittest
import sys
import os
import glob
import re
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('../..'))

from flowkit import Sample, GatingStrategy, Session, gates, parse_gating_xml

data1_fcs_path = 'examples/gate_ref/data1.fcs'
data1_sample = Sample(data1_fcs_path)


class GatingMLTestCase(unittest.TestCase):
    def test_parse_gating_xml(self):
        gml_path = 'examples/gate_ref/gml/gml_all_gates.xml'
        gs = parse_gating_xml(gml_path)

        self.assertIsInstance(gs, GatingStrategy)

    def test_fail_parse_gating_xml(self):
        gml_path = 'examples/simple_line_example/single_ellipse_51_events.wsp'

        self.assertRaises(ValueError, parse_gating_xml, gml_path)

    @staticmethod
    def test_min_range_gate():
        gml_path = 'examples/gate_ref/gml/gml_range_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Range1.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'Range1')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_rect1_gate():
        gml_path = 'examples/gate_ref/gml/gml_rect1_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Rectangle1.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'Rectangle1')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_rect2_gate():
        gml_path = 'examples/gate_ref/gml/gml_rect2_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Rectangle2.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'Rectangle2')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_poly1_gate():
        gml_path = 'examples/gate_ref/gml/gml_poly1_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Polygon1.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'Polygon1')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_poly2_gate():
        gml_path = 'examples/gate_ref/gml/gml_poly2_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Polygon2.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'Polygon2')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_poly3_non_solid_gate():
        gml_path = 'examples/gate_ref/gml/gml_poly3ns_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Polygon3NS.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'Polygon3NS')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_ellipse1_gate():
        gml_path = 'examples/gate_ref/gml/gml_ellipse1_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Ellipse1.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'Ellipse1')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_ellipsoid_3d_gate():
        gml_path = 'examples/gate_ref/gml/gml_ellipsoid3d_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Ellipsoid3D.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'Ellipsoid3D')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_time_range_gate():
        gml_path = 'examples/gate_ref/gml/gml_time_range_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Range2.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'Range2')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_quadrant1_gate():
        gml_path = 'examples/gate_ref/gml/gml_quadrant1_gate.xml'
        res1_path = 'examples/gate_ref/truth/Results_FL2N-FL4N.txt'
        res2_path = 'examples/gate_ref/truth/Results_FL2N-FL4P.txt'
        res3_path = 'examples/gate_ref/truth/Results_FL2P-FL4N.txt'
        res4_path = 'examples/gate_ref/truth/Results_FL2P-FL4P.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)

        truth1 = pd.read_csv(res1_path, header=None, squeeze=True, dtype='bool').values
        truth2 = pd.read_csv(res2_path, header=None, squeeze=True, dtype='bool').values
        truth3 = pd.read_csv(res3_path, header=None, squeeze=True, dtype='bool').values
        truth4 = pd.read_csv(res4_path, header=None, squeeze=True, dtype='bool').values

        s.analyze_samples(group_name)
        results = s.get_gating_results(group_name, data1_sample.original_filename)
        result1 = results.get_gate_indices('FL2N-FL4N')
        result2 = results.get_gate_indices('FL2N-FL4P')
        result3 = results.get_gate_indices('FL2P-FL4N')
        result4 = results.get_gate_indices('FL2P-FL4P')

        np.testing.assert_array_equal(truth1, result1)
        np.testing.assert_array_equal(truth2, result2)
        np.testing.assert_array_equal(truth3, result3)
        np.testing.assert_array_equal(truth4, result4)

    def test_quadrant_gate_relative_percent(self):
        gml_path = 'examples/gate_ref/gml/gml_quadrant1_gate.xml'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        result = s.get_gating_results(group_name, data1_sample.original_filename)

        total_percent = result.get_gate_relative_percent('FL2N-FL4N') + \
            result.get_gate_relative_percent('FL2N-FL4P') + \
            result.get_gate_relative_percent('FL2P-FL4N') + \
            result.get_gate_relative_percent('FL2P-FL4P')

        self.assertEqual(100.0, total_percent)

    @staticmethod
    def test_quadrant2_gate():
        gml_path = 'examples/gate_ref/gml/gml_quadrant2_gate.xml'
        res1_path = 'examples/gate_ref/truth/Results_FSCN-SSCN.txt'
        res2_path = 'examples/gate_ref/truth/Results_FSCD-SSCN-FL1N.txt'
        res3_path = 'examples/gate_ref/truth/Results_FSCP-SSCN-FL1N.txt'
        res4_path = 'examples/gate_ref/truth/Results_FSCD-FL1P.txt'
        res5_path = 'examples/gate_ref/truth/Results_FSCN-SSCP-FL1P.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth1 = pd.read_csv(res1_path, header=None, squeeze=True, dtype='bool').values
        truth2 = pd.read_csv(res2_path, header=None, squeeze=True, dtype='bool').values
        truth3 = pd.read_csv(res3_path, header=None, squeeze=True, dtype='bool').values
        truth4 = pd.read_csv(res4_path, header=None, squeeze=True, dtype='bool').values
        truth5 = pd.read_csv(res5_path, header=None, squeeze=True, dtype='bool').values

        result = s.get_gating_results(group_name, data1_sample.original_filename)

        np.testing.assert_array_equal(truth1, result.get_gate_indices('FSCN-SSCN'))
        np.testing.assert_array_equal(truth2, result.get_gate_indices('FSCD-SSCN-FL1N'))
        np.testing.assert_array_equal(truth3, result.get_gate_indices('FSCP-SSCN-FL1N'))
        np.testing.assert_array_equal(truth4, result.get_gate_indices('FSCD-FL1P'))
        np.testing.assert_array_equal(truth5, result.get_gate_indices('FSCN-SSCP-FL1P'))

    @staticmethod
    def test_ratio_range1_gate():
        gml_path = 'examples/gate_ref/gml/gml_ratio_range1_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_RatRange1.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'RatRange1')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_ratio_range2_gate():
        gml_path = 'examples/gate_ref/gml/gml_ratio_range2_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_RatRange2.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'RatRange2')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_log_ratio_range1_gate():
        gml_path = 'examples/gate_ref/gml/gml_log_ratio_range1_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_RatRange1a.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'RatRange1a')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_boolean_and1_gate():
        gml_path = 'examples/gate_ref/gml/gml_boolean_and1_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_And1.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'And1')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_boolean_and2_gate():
        gml_path = 'examples/gate_ref/gml/gml_boolean_and2_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_And2.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'And2')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_boolean_or1_gate():
        gml_path = 'examples/gate_ref/gml/gml_boolean_or1_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Or1.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'Or1')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_boolean_and3_complement_gate():
        gml_path = 'examples/gate_ref/gml/gml_boolean_and3_complement_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_And3.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'And3')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_boolean_not1_gate():
        gml_path = 'examples/gate_ref/gml/gml_boolean_not1_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Not1.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'Not1')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_boolean_and4_not_gate():
        gml_path = 'examples/gate_ref/gml/gml_boolean_and4_not_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_And4.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'And4')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_boolean_or2_complement_gate():
        gml_path = 'examples/gate_ref/gml/gml_boolean_or2_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Or2.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'Or2')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_matrix_poly4_gate():
        gml_path = 'examples/gate_ref/gml/gml_matrix_poly4_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Polygon4.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'Polygon4')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_matrix_rect3_gate():
        gml_path = 'examples/gate_ref/gml/gml_matrix_rect3_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Rectangle3.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'Rectangle3')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_matrix_rect4_gate():
        gml_path = 'examples/gate_ref/gml/gml_matrix_rect4_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Rectangle4.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'Rectangle4')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_matrix_rect5_gate():
        gml_path = 'examples/gate_ref/gml/gml_matrix_rect5_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Rectangle5.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'Rectangle5')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_transform_asinh_range1_gate():
        gml_path = 'examples/gate_ref/gml/gml_transform_asinh_range1_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange1.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'ScaleRange1')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_transform_hyperlog_range2_gate():
        gml_path = 'examples/gate_ref/gml/gml_transform_hyperlog_range2_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange2.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'ScaleRange2')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_transform_linear_range3_gate():
        gml_path = 'examples/gate_ref/gml/gml_transform_linear_range3_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange3.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'ScaleRange3')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_transform_logicle_range4_gate():
        gml_path = 'examples/gate_ref/gml/gml_transform_logicle_range4_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange4.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'ScaleRange4')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_transform_logicle_range5_gate():
        gml_path = 'examples/gate_ref/gml/gml_transform_logicle_range5_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange5.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'ScaleRange5')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_transform_log_range6_gate():
        gml_path = 'examples/gate_ref/gml/gml_transform_log_range6_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange6.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'ScaleRange6')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_matrix_transform_asinh_range1c_gate():
        gml_path = 'examples/gate_ref/gml/gml_matrix_transform_asinh_range1c_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange1c.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'ScaleRange1c')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_matrix_transform_hyperlog_range2c_gate():
        gml_path = 'examples/gate_ref/gml/gml_matrix_transform_hyperlog_range2c_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange2c.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'ScaleRange2c')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_matrix_transform_linear_range3c_gate():
        gml_path = 'examples/gate_ref/gml/gml_matrix_transform_linear_range3c_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange3c.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'ScaleRange3c')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_matrix_transform_logicle_range4c_gate():
        gml_path = 'examples/gate_ref/gml/gml_matrix_transform_logicle_range4c_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange4c.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'ScaleRange4c')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_matrix_transform_logicle_range5c_gate():
        gml_path = 'examples/gate_ref/gml/gml_matrix_transform_logicle_range5c_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange5c.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'ScaleRange5c')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_matrix_transform_asinh_range6c_gate():
        gml_path = 'examples/gate_ref/gml/gml_matrix_transform_asinh_range6c_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange6c.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'ScaleRange6c')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_matrix_transform_hyperlog_range7c_gate():
        gml_path = 'examples/gate_ref/gml/gml_matrix_transform_hyperlog_range7c_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange7c.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'ScaleRange7c')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_matrix_transform_logicle_range8c_gate():
        gml_path = 'examples/gate_ref/gml/gml_matrix_transform_logicle_range8c_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_ScaleRange8c.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'ScaleRange8c')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_matrix_transform_logicle_rect1_gate():
        gml_path = 'examples/gate_ref/gml/gml_matrix_transform_logicle_rect1_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_ScaleRect1.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'ScaleRect1')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_parent_poly1_boolean_and2_gate():
        gml_path = 'examples/gate_ref/gml/gml_parent_poly1_boolean_and2_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_ParAnd2.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'ParAnd2')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_parent_range1_boolean_and3_gate():
        gml_path = 'examples/gate_ref/gml/gml_parent_range1_boolean_and3_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_ParAnd3.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'ParAnd3')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_parent_rect1_rect_par1_gate():
        gml_path = 'examples/gate_ref/gml/gml_parent_rect1_rect_par1_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_ScalePar1.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'ScalePar1')

        np.testing.assert_array_equal(truth, result)

    @staticmethod
    def test_parent_quadrant_rect_gate():
        gml_path = 'examples/gate_ref/gml/gml_parent_quadrant_rect_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_ParQuadRect.txt'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = s.get_gate_indices(group_name, data1_sample.original_filename, 'ParRectangle1')

        np.testing.assert_array_equal(truth, result)

    def test_get_parent_rect_gate(self):
        gml_path = 'examples/gate_ref/gml/gml_parent_rect1_rect_par1_gate.xml'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        parent_id = s.get_parent_gate_id(group_name, 'ScalePar1')

        self.assertEqual(parent_id, 'ScaleRect1')

        parent_gate = s.get_gate(group_name, data1_sample.original_filename, parent_id)

        self.assertIsInstance(parent_gate, gates.RectangleGate)

    def test_get_parent_quadrant_gate(self):
        gml_path = 'examples/gate_ref/gml/gml_parent_quadrant_rect_gate.xml'

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        parent_id = s.get_parent_gate_id(group_name, 'ParRectangle1')

        self.assertEqual(parent_id, 'FL2P-FL4P')

        parent_gate = s.get_gate(group_name, data1_sample.original_filename, parent_id)

        self.assertIsInstance(parent_gate, gates.Quadrant)

    def test_gate_gating_hierarchy(self):
        gml_path = 'examples/gate_ref/gml/gml_all_gates.xml'
        gs = parse_gating_xml(gml_path)
        gs_ascii = gs.get_gate_hierarchy('ascii')
        gs_json = gs.get_gate_hierarchy('json')
        gs_dict = gs.get_gate_hierarchy('dict')

        self.assertIsInstance(gs_ascii, str)
        self.assertIsInstance(gs_json, str)
        self.assertIsInstance(gs_dict, dict)

    @staticmethod
    def test_all_gates():
        gml_path = 'examples/gate_ref/gml/gml_all_gates.xml'
        truth_pattern = 'examples/gate_ref/truth/Results*.txt'

        res_files = glob.glob(truth_pattern)

        truth_dict = {}

        for res_path in res_files:
            match = re.search("Results_(.+)\\.txt$", res_path)
            if match is not None:
                g_id = match.group(1)
                truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

                truth_dict[g_id] = truth

        s = Session()
        group_name = 'gml'
        s.add_sample_group(group_name, gating_strategy=gml_path)
        s.add_samples(data1_sample)
        s.assign_sample(data1_sample.original_filename, group_name)
        s.analyze_samples(group_name=group_name)
        results = s.get_gating_results(group_name, data1_sample.original_filename)

        for row in results.report.itertuples():
            np.testing.assert_array_equal(
                truth_dict[row[0][1]],
                results.get_gate_indices(row[0][1])
            )
