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
