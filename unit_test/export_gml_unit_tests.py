import unittest
import sys
import os
from io import BytesIO
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('..'))

from flowkit import Sample, GatingStrategy

data1_fcs_path = 'examples/gate_ref/data1.fcs'
data1_sample = Sample(data1_fcs_path)


class ExportGMLTestCase(unittest.TestCase):
    @staticmethod
    def test_min_range_gate():
        gml_path = 'examples/gate_ref/gml/gml_range_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Range1.txt'

        gs = GatingStrategy(gml_path)

        out_file = BytesIO()

        gs.export_gml(out_file)
        out_file.seek(0)

        gs_out = GatingStrategy(out_file)
        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        result = gs_out.gate_sample(data1_sample, 'Range1')

        np.testing.assert_array_equal(truth, result.get_gate_indices('Range1'))

    @staticmethod
    def test_rect1_gate():
        gml_path = 'examples/gate_ref/gml/gml_rect1_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Rectangle1.txt'

        gs = GatingStrategy(gml_path)

        out_file = BytesIO()

        gs.export_gml(out_file)
        out_file.seek(0)

        gs_out = GatingStrategy(out_file)

        result = gs_out.gate_sample(data1_sample, 'Rectangle1')
        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        np.testing.assert_array_equal(truth, result.get_gate_indices('Rectangle1'))

    @staticmethod
    def test_rect2_gate():
        gml_path = 'examples/gate_ref/gml/gml_rect2_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Rectangle2.txt'

        gs = GatingStrategy(gml_path)

        out_file = BytesIO()

        gs.export_gml(out_file)
        out_file.seek(0)

        gs_out = GatingStrategy(out_file)

        result = gs_out.gate_sample(data1_sample, 'Rectangle2')
        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        np.testing.assert_array_equal(truth, result.get_gate_indices('Rectangle2'))

    @staticmethod
    def test_poly1_gate():
        gml_path = 'examples/gate_ref/gml/gml_poly1_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Polygon1.txt'

        gs = GatingStrategy(gml_path)

        out_file = BytesIO()

        gs.export_gml(out_file)
        out_file.seek(0)

        gs_out = GatingStrategy(out_file)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = gs_out.gate_sample(data1_sample, 'Polygon1')

        np.testing.assert_array_equal(truth, result.get_gate_indices('Polygon1'))

    @staticmethod
    def test_poly2_gate():
        gml_path = 'examples/gate_ref/gml/gml_poly2_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Polygon2.txt'

        gs = GatingStrategy(gml_path)

        out_file = BytesIO()

        gs.export_gml(out_file)
        out_file.seek(0)

        gs_out = GatingStrategy(out_file)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = gs_out.gate_sample(data1_sample, 'Polygon2')

        np.testing.assert_array_equal(truth, result.get_gate_indices('Polygon2'))

    @staticmethod
    def test_poly3_non_solid_gate():
        gml_path = 'examples/gate_ref/gml/gml_poly3ns_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Polygon3NS.txt'

        gs = GatingStrategy(gml_path)

        out_file = BytesIO()

        gs.export_gml(out_file)
        out_file.seek(0)

        gs_out = GatingStrategy(out_file)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values
        result = gs_out.gate_sample(data1_sample, 'Polygon3NS')

        np.testing.assert_array_equal(truth, result.get_gate_indices('Polygon3NS'))

    @staticmethod
    def test_ellipse1_gate():
        gml_path = 'examples/gate_ref/gml/gml_ellipse1_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Ellipse1.txt'

        gs = GatingStrategy(gml_path)

        out_file = BytesIO()

        gs.export_gml(out_file)
        out_file.seek(0)

        gs_out = GatingStrategy(out_file)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        result = gs_out.gate_sample(data1_sample, 'Ellipse1')

        np.testing.assert_array_equal(truth, result.get_gate_indices('Ellipse1'))

    @staticmethod
    def test_ellipsoid_3d_gate():
        gml_path = 'examples/gate_ref/gml/gml_ellipsoid3d_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Ellipsoid3D.txt'

        gs = GatingStrategy(gml_path)

        out_file = BytesIO()

        gs.export_gml(out_file)
        out_file.seek(0)

        gs_out = GatingStrategy(out_file)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        result = gs_out.gate_sample(data1_sample, 'Ellipsoid3D')

        np.testing.assert_array_equal(truth, result.get_gate_indices('Ellipsoid3D'))

    @staticmethod
    def test_time_range_gate():
        gml_path = 'examples/gate_ref/gml/gml_time_range_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Range2.txt'

        gs = GatingStrategy(gml_path)

        out_file = BytesIO()

        gs.export_gml(out_file)
        out_file.seek(0)

        gs_out = GatingStrategy(out_file)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        result = gs_out.gate_sample(data1_sample, 'Range2')

        np.testing.assert_array_equal(truth, result.get_gate_indices('Range2'))

    @staticmethod
    def test_quadrant1_gate():
        gml_path = 'examples/gate_ref/gml/gml_quadrant1_gate.xml'
        res1_path = 'examples/gate_ref/truth/Results_FL2N-FL4N.txt'
        res2_path = 'examples/gate_ref/truth/Results_FL2N-FL4P.txt'
        res3_path = 'examples/gate_ref/truth/Results_FL2P-FL4N.txt'
        res4_path = 'examples/gate_ref/truth/Results_FL2P-FL4P.txt'

        gs = GatingStrategy(gml_path)

        out_file = BytesIO()

        gs.export_gml(out_file)
        out_file.seek(0)

        gs_out = GatingStrategy(out_file)

        truth1 = pd.read_csv(res1_path, header=None, squeeze=True, dtype='bool').values
        truth2 = pd.read_csv(res2_path, header=None, squeeze=True, dtype='bool').values
        truth3 = pd.read_csv(res3_path, header=None, squeeze=True, dtype='bool').values
        truth4 = pd.read_csv(res4_path, header=None, squeeze=True, dtype='bool').values

        result = gs_out.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth1, result.get_gate_indices('FL2N-FL4N'))
        np.testing.assert_array_equal(truth2, result.get_gate_indices('FL2N-FL4P'))
        np.testing.assert_array_equal(truth3, result.get_gate_indices('FL2P-FL4N'))
        np.testing.assert_array_equal(truth4, result.get_gate_indices('FL2P-FL4P'))

    @staticmethod
    def test_quadrant2_gate():
        gml_path = 'examples/gate_ref/gml/gml_quadrant2_gate.xml'
        res1_path = 'examples/gate_ref/truth/Results_FSCN-SSCN.txt'
        res2_path = 'examples/gate_ref/truth/Results_FSCD-SSCN-FL1N.txt'
        res3_path = 'examples/gate_ref/truth/Results_FSCP-SSCN-FL1N.txt'
        res4_path = 'examples/gate_ref/truth/Results_FSCD-FL1P.txt'
        res5_path = 'examples/gate_ref/truth/Results_FSCN-SSCP-FL1P.txt'

        gs = GatingStrategy(gml_path)

        out_file = BytesIO()

        gs.export_gml(out_file)
        out_file.seek(0)

        gs_out = GatingStrategy(out_file)

        truth1 = pd.read_csv(res1_path, header=None, squeeze=True, dtype='bool').values
        truth2 = pd.read_csv(res2_path, header=None, squeeze=True, dtype='bool').values
        truth3 = pd.read_csv(res3_path, header=None, squeeze=True, dtype='bool').values
        truth4 = pd.read_csv(res4_path, header=None, squeeze=True, dtype='bool').values
        truth5 = pd.read_csv(res5_path, header=None, squeeze=True, dtype='bool').values

        result = gs_out.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth1, result.get_gate_indices('FSCN-SSCN'))
        np.testing.assert_array_equal(truth2, result.get_gate_indices('FSCD-SSCN-FL1N'))
        np.testing.assert_array_equal(truth3, result.get_gate_indices('FSCP-SSCN-FL1N'))
        np.testing.assert_array_equal(truth4, result.get_gate_indices('FSCD-FL1P'))
        np.testing.assert_array_equal(truth5, result.get_gate_indices('FSCN-SSCP-FL1P'))

    @staticmethod
    def test_ratio_range1_gate():
        gml_path = 'examples/gate_ref/gml/gml_ratio_range1_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_RatRange1.txt'

        gs = GatingStrategy(gml_path)

        out_file = BytesIO()

        gs.export_gml(out_file)
        out_file.seek(0)

        gs_out = GatingStrategy(out_file)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        result = gs_out.gate_sample(data1_sample, 'RatRange1')

        np.testing.assert_array_equal(truth, result.get_gate_indices('RatRange1'))

    @staticmethod
    def test_ratio_range2_gate():
        gml_path = 'examples/gate_ref/gml/gml_ratio_range2_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_RatRange2.txt'

        gs = GatingStrategy(gml_path)

        out_file = BytesIO()

        gs.export_gml(out_file)
        out_file.seek(0)

        gs_out = GatingStrategy(out_file)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        result = gs_out.gate_sample(data1_sample, 'RatRange2')

        np.testing.assert_array_equal(truth, result.get_gate_indices('RatRange2'))

    @staticmethod
    def test_log_ratio_range1_gate():
        gml_path = 'examples/gate_ref/gml/gml_log_ratio_range1_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_RatRange1a.txt'

        gs = GatingStrategy(gml_path)

        out_file = BytesIO()

        gs.export_gml(out_file)
        out_file.seek(0)

        gs_out = GatingStrategy(out_file)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        result = gs_out.gate_sample(data1_sample, 'RatRange1a')

        np.testing.assert_array_equal(truth, result.get_gate_indices('RatRange1a'))

    @staticmethod
    def test_boolean_and1_gate():
        gml_path = 'examples/gate_ref/gml/gml_boolean_and1_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_And1.txt'

        gs = GatingStrategy(gml_path)

        out_file = BytesIO()

        gs.export_gml(out_file)
        out_file.seek(0)

        gs_out = GatingStrategy(out_file)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        result = gs_out.gate_sample(data1_sample, 'And1')

        np.testing.assert_array_equal(truth, result.get_gate_indices('And1'))

    @staticmethod
    def test_boolean_and2_gate():
        gml_path = 'examples/gate_ref/gml/gml_boolean_and2_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_And2.txt'

        gs = GatingStrategy(gml_path)

        out_file = BytesIO()

        gs.export_gml(out_file)
        out_file.seek(0)

        gs_out = GatingStrategy(out_file)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        result = gs_out.gate_sample(data1_sample, 'And2')

        np.testing.assert_array_equal(truth, result.get_gate_indices('And2'))

    @staticmethod
    def test_boolean_or1_gate():
        gml_path = 'examples/gate_ref/gml/gml_boolean_or1_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Or1.txt'

        gs = GatingStrategy(gml_path)

        out_file = BytesIO()

        gs.export_gml(out_file)
        out_file.seek(0)

        gs_out = GatingStrategy(out_file)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        result = gs_out.gate_sample(data1_sample, 'Or1')

        np.testing.assert_array_equal(truth, result.get_gate_indices('Or1'))

    @staticmethod
    def test_boolean_and3_complement_gate():
        gml_path = 'examples/gate_ref/gml/gml_boolean_and3_complement_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_And3.txt'

        gs = GatingStrategy(gml_path)

        out_file = BytesIO()

        gs.export_gml(out_file)
        out_file.seek(0)

        gs_out = GatingStrategy(out_file)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        result = gs_out.gate_sample(data1_sample, 'And3')

        np.testing.assert_array_equal(truth, result.get_gate_indices('And3'))

    @staticmethod
    def test_boolean_not1_gate():
        gml_path = 'examples/gate_ref/gml/gml_boolean_not1_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Not1.txt'

        gs = GatingStrategy(gml_path)

        out_file = BytesIO()

        gs.export_gml(out_file)
        out_file.seek(0)

        gs_out = GatingStrategy(out_file)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        result = gs_out.gate_sample(data1_sample, 'Not1')

        np.testing.assert_array_equal(truth, result.get_gate_indices('Not1'))

    @staticmethod
    def test_boolean_and4_not_gate():
        gml_path = 'examples/gate_ref/gml/gml_boolean_and4_not_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_And4.txt'

        gs = GatingStrategy(gml_path)

        out_file = BytesIO()

        gs.export_gml(out_file)
        out_file.seek(0)

        gs_out = GatingStrategy(out_file)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        result = gs_out.gate_sample(data1_sample, 'And4')

        np.testing.assert_array_equal(truth, result.get_gate_indices('And4'))

    @staticmethod
    def test_boolean_or2_complement_gate():
        gml_path = 'examples/gate_ref/gml/gml_boolean_or2_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Or2.txt'

        gs = GatingStrategy(gml_path)

        out_file = BytesIO()

        gs.export_gml(out_file)
        out_file.seek(0)

        gs_out = GatingStrategy(out_file)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        result = gs_out.gate_sample(data1_sample, 'Or2')

        np.testing.assert_array_equal(truth, result.get_gate_indices('Or2'))
