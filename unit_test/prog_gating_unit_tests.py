import unittest
import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('..'))

from flowkit import Sample, GatingStrategy, Dimension, Vertex, gates

data1_fcs_path = 'examples/gate_ref/data1.fcs'
data1_sample = Sample(data1_fcs_path)


class GatingTestCase(unittest.TestCase):
    @staticmethod
    def test_add_min_range_gate():
        res_path = 'examples/gate_ref/truth/Results_Range1.txt'

        gs = GatingStrategy()

        dim1 = Dimension("FSC-H", compensation_ref="uncompensated", range_min=100)
        dims = [dim1]

        rect_gate = gates.RectangleGate("Range1", None, dims, gs)
        gs.add_gate(rect_gate)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        result = gs.gate_sample(data1_sample, 'Range1')

        np.testing.assert_array_equal(truth, result.get_gate_indices('Range1'))

    @staticmethod
    def test_add_rect1_gate():
        gs = GatingStrategy()

        dim1 = Dimension("SSC-H", compensation_ref="uncompensated", range_min=20, range_max=80)
        dim2 = Dimension("FL1-H", compensation_ref="uncompensated", range_min=70, range_max=200)
        dims = [dim1, dim2]

        rect_gate = gates.RectangleGate("Rectangle1", None, dims, gs)
        gs.add_gate(rect_gate)

        res_path = 'examples/gate_ref/truth/Results_Rectangle1.txt'
        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        result = gs.gate_sample(data1_sample, 'Rectangle1')

        np.testing.assert_array_equal(truth, result.get_gate_indices('Rectangle1'))

    @staticmethod
    def test_add_rect2_gate():
        gs = GatingStrategy()

        dim1 = Dimension("SSC-H", compensation_ref="FCS", range_min=20, range_max=80)
        dim2 = Dimension("FL1-H", compensation_ref="FCS", range_min=70, range_max=200)
        dims = [dim1, dim2]

        rect_gate = gates.RectangleGate("Rectangle2", None, dims, gs)
        gs.add_gate(rect_gate)

        res_path = 'examples/gate_ref/truth/Results_Rectangle2.txt'
        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        result = gs.gate_sample(data1_sample, 'Rectangle2')

        np.testing.assert_array_equal(truth, result.get_gate_indices('Rectangle2'))

    @staticmethod
    def test_add_poly1_gate():
        gs = GatingStrategy()

        dim1 = Dimension("FL2-H", compensation_ref="FCS")
        dim2 = Dimension("FL3-H", compensation_ref="FCS")
        dims = [dim1, dim2]

        vertices = [
            Vertex([5, 5]),
            Vertex([500, 5]),
            Vertex([500, 500])
        ]

        poly_gate = gates.PolygonGate("Polygon1", None, dims, vertices, gs)
        gs.add_gate(poly_gate)

        res_path = 'examples/gate_ref/truth/Results_Polygon1.txt'
        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        result = gs.gate_sample(data1_sample, 'Polygon1')

        np.testing.assert_array_equal(truth, result.get_gate_indices('Polygon1'))

    @staticmethod
    def test_add_poly2_gate():
        gs = GatingStrategy()

        dim1 = Dimension("FL1-H", compensation_ref="FCS")
        dim2 = Dimension("FL4-H", compensation_ref="FCS")
        dims = [dim1, dim2]

        vertices = [
            Vertex([20, 10]),
            Vertex([120, 10]),
            Vertex([120, 160]),
            Vertex([20, 160])
        ]

        poly_gate = gates.PolygonGate("Polygon2", None, dims, vertices, gs)
        gs.add_gate(poly_gate)

        res_path = 'examples/gate_ref/truth/Results_Polygon2.txt'
        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        result = gs.gate_sample(data1_sample, 'Polygon2')

        np.testing.assert_array_equal(truth, result.get_gate_indices('Polygon2'))

    @staticmethod
    def test_add_poly3_non_solid_gate():
        gs = GatingStrategy()

        dim1 = Dimension("SSC-H", compensation_ref="uncompensated")
        dim2 = Dimension("FL3-H", compensation_ref="FCS")
        dims = [dim1, dim2]

        vertices = [
            Vertex([10, 10]),
            Vertex([500, 10]),
            Vertex([500, 390]),
            Vertex([100, 390]),
            Vertex([100, 180]),
            Vertex([200, 180]),
            Vertex([200, 300]),
            Vertex([10, 300])
        ]

        poly_gate = gates.PolygonGate("Polygon3NS", None, dims, vertices, gs)
        gs.add_gate(poly_gate)

        res_path = 'examples/gate_ref/truth/Results_Polygon3NS.txt'
        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        result = gs.gate_sample(data1_sample, 'Polygon3NS')

        np.testing.assert_array_equal(truth, result.get_gate_indices('Polygon3NS'))

    @staticmethod
    def test_add_ellipse1_gate():
        gs = GatingStrategy()

        dim1 = Dimension("FL3-H", compensation_ref="uncompensated")
        dim2 = Dimension("FL4-H", compensation_ref="uncompensated")
        dims = [dim1, dim2]

        coords = [12.99701, 16.22941]
        cov_mat = [[62.5, 37.5], [37.5, 62.5]]
        dist_square = 1

        poly_gate = gates.EllipsoidGate("Ellipse1", None, dims, coords, cov_mat, dist_square, gs)
        gs.add_gate(poly_gate)

        res_path = 'examples/gate_ref/truth/Results_Ellipse1.txt'
        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        result = gs.gate_sample(data1_sample, 'Ellipse1')

        np.testing.assert_array_equal(truth, result.get_gate_indices('Ellipse1'))

    @staticmethod
    def test_add_ellipsoid_3d_gate():
        gs = GatingStrategy()

        dim1 = Dimension("FL3-H", compensation_ref="FCS")
        dim2 = Dimension("FL4-H", compensation_ref="FCS")
        dim3 = Dimension("FL1-H", compensation_ref="FCS")
        dims = [dim1, dim2, dim3]

        coords = [40.3, 30.6, 20.8]
        cov_mat = [[2.5, 7.5, 17.5], [7.5, 7.0, 13.5], [15.5, 13.5, 4.3]]
        dist_square = 1

        poly_gate = gates.EllipsoidGate("Ellipsoid3D", None, dims, coords, cov_mat, dist_square, gs)
        gs.add_gate(poly_gate)

        res_path = 'examples/gate_ref/truth/Results_Ellipsoid3D.txt'
        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        result = gs.gate_sample(data1_sample, 'Ellipsoid3D')

        np.testing.assert_array_equal(truth, result.get_gate_indices('Ellipsoid3D'))

    @staticmethod
    def test_add_time_range_gate():
        res_path = 'examples/gate_ref/truth/Results_Range2.txt'

        gs = GatingStrategy()

        dim1 = Dimension("Time", compensation_ref="uncompensated", range_min=20, range_max=80)
        dims = [dim1]

        rect_gate = gates.RectangleGate("Range2", None, dims, gs)
        gs.add_gate(rect_gate)

        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        result = gs.gate_sample(data1_sample, 'Range2')

        np.testing.assert_array_equal(truth, result.get_gate_indices('Range2'))