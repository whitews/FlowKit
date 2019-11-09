import unittest
import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('..'))

from flowkit import Sample, GatingStrategy, Dimension, QuadrantDivider, Vertex, gates

data1_fcs_path = 'examples/gate_ref/data1.fcs'
data1_sample = Sample(data1_fcs_path)

quadrants_q1 = {
            'FL2P-FL4P': [
                {
                    'divider': 'FL2',
                    'dimension': 'FL2-H',
                    'location': 15.0,
                    'min': 12.14748,
                    'max': None
                },
                {
                    'divider': 'FL4',
                    'dimension': 'FL4-H',
                    'location': 15.0,
                    'min': 14.22417,
                    'max': None
                }
            ],
            'FL2N-FL4P': [
                {
                    'divider': 'FL2',
                    'dimension': 'FL2-H',
                    'location': 5.0,
                    'min': None,
                    'max': 12.14748
                },
                {
                    'divider': 'FL4',
                    'dimension': 'FL4-H',
                    'location': 15.0,
                    'min': 14.22417,
                    'max': None
                }
            ],
            'FL2N-FL4N': [
                {
                    'divider': 'FL2',
                    'dimension': 'FL2-H',
                    'location': 5.0,
                    'min': None,
                    'max': 12.14748
                },
                {
                    'divider': 'FL4',
                    'dimension': 'FL4-H',
                    'location': 5.0,
                    'min': None,
                    'max': 14.22417
                }
            ],
            'FL2P-FL4N': [
                {
                    'divider': 'FL2',
                    'dimension': 'FL2-H',
                    'location': 15.0,
                    'min': 12.14748,
                    'max': None
                },
                {
                    'divider': 'FL4',
                    'dimension': 'FL4-H',
                    'location': 5.0,
                    'min': None,
                    'max': 14.22417
                }
            ]
        }

quadrants_q2 = {
    'FSCD-FL1P': [
        {
            'dimension': 'FSC-H',
            'divider': 'FSC',
            'location': 30.0,
            'max': 70.02725,
            'min': 28.0654},
        {
            'dimension': 'FL1-H',
            'divider': 'FL1',
            'location': 10.0,
            'max': None,
            'min': 6.43567
        }
    ],
    'FSCD-SSCN-FL1N': [
        {
            'dimension': 'FSC-H',
            'divider': 'FSC',
            'location': 30.0,
            'max': 70.02725,
            'min': 28.0654
        },
        {
            'dimension': 'SSC-H',
            'divider': 'SSC',
            'location': 10.0,
            'max': 17.75,
            'min': None
        },
        {
            'dimension': 'FL1-H',
            'divider': 'FL1',
            'location': 5.0,
            'max': 6.43567,
            'min': None
        }
    ],
    'FSCN-SSCN': [
        {
            'dimension': 'FSC-H',
            'divider': 'FSC',
            'location': 10.0,
            'max': 28.0654,
            'min': None
        },
        {
            'dimension': 'SSC-H',
            'divider': 'SSC',
            'location': 10.0,
            'max': 17.75,
            'min': None
        }
    ],
    'FSCN-SSCP-FL1P': [
        {
            'dimension': 'FSC-H',
            'divider': 'FSC',
            'location': 10.0,
            'max': 28.0654,
            'min': None
        },
        {
            'dimension': 'SSC-H',
            'divider': 'SSC',
            'location': 20.0,
            'max': None,
            'min': 17.75
        },
        {
            'dimension': 'FL1-H',
            'divider': 'FL1',
            'location': 15.0,
            'max': None,
            'min': 6.43567
        }
    ],
    'FSCP-SSCN-FL1N': [
        {
            'dimension': 'FSC-H',
            'divider': 'FSC',
            'location': 80.0,
            'max': None,
            'min': 70.02725
        },
        {
            'dimension': 'SSC-H',
            'divider': 'SSC',
            'location': 10.0,
            'max': 17.75,
            'min': None
        },
        {
            'dimension': 'FL1-H',
            'divider': 'FL1',
            'location': 5.0,
            'max': 6.43567,
            'min': None
        }
    ]
}


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

    @staticmethod
    def test_add_quadrant1_gate():
        res1_path = 'examples/gate_ref/truth/Results_FL2N-FL4N.txt'
        res2_path = 'examples/gate_ref/truth/Results_FL2N-FL4P.txt'
        res3_path = 'examples/gate_ref/truth/Results_FL2P-FL4N.txt'
        res4_path = 'examples/gate_ref/truth/Results_FL2P-FL4P.txt'

        gs = GatingStrategy()

        div1 = QuadrantDivider("FL2", "FL2-H", "FCS", [12.14748])
        div2 = QuadrantDivider("FL4", "FL4-H", "FCS", [14.22417])

        divs = [div1, div2]

        quad_gate = gates.QuadrantGate("Quadrant1", None, divs, quadrants_q1, gs)
        gs.add_gate(quad_gate)

        truth1 = pd.read_csv(res1_path, header=None, squeeze=True, dtype='bool').values
        truth2 = pd.read_csv(res2_path, header=None, squeeze=True, dtype='bool').values
        truth3 = pd.read_csv(res3_path, header=None, squeeze=True, dtype='bool').values
        truth4 = pd.read_csv(res4_path, header=None, squeeze=True, dtype='bool').values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth1, result.get_gate_indices('FL2N-FL4N'))
        np.testing.assert_array_equal(truth2, result.get_gate_indices('FL2N-FL4P'))
        np.testing.assert_array_equal(truth3, result.get_gate_indices('FL2P-FL4N'))
        np.testing.assert_array_equal(truth4, result.get_gate_indices('FL2P-FL4P'))

    def test_add_quadrant_gate_relative_percent(self):
        gs = GatingStrategy()

        div1 = QuadrantDivider("FL2", "FL2-H", "FCS", [12.14748])
        div2 = QuadrantDivider("FL4", "FL4-H", "FCS", [14.22417])

        divs = [div1, div2]

        quad_gate = gates.QuadrantGate("Quadrant1", None, divs, quadrants_q1, gs)
        gs.add_gate(quad_gate)

        result = gs.gate_sample(data1_sample)

        total_percent = result.get_gate_relative_percent('FL2N-FL4N') + \
            result.get_gate_relative_percent('FL2N-FL4P') + \
            result.get_gate_relative_percent('FL2P-FL4N') + \
            result.get_gate_relative_percent('FL2P-FL4P')

        self.assertEqual(100.0, total_percent)

    @staticmethod
    def test_add_quadrant2_gate():
        res1_path = 'examples/gate_ref/truth/Results_FSCN-SSCN.txt'
        res2_path = 'examples/gate_ref/truth/Results_FSCD-SSCN-FL1N.txt'
        res3_path = 'examples/gate_ref/truth/Results_FSCP-SSCN-FL1N.txt'
        res4_path = 'examples/gate_ref/truth/Results_FSCD-FL1P.txt'
        res5_path = 'examples/gate_ref/truth/Results_FSCN-SSCP-FL1P.txt'

        truth1 = pd.read_csv(res1_path, header=None, squeeze=True, dtype='bool').values
        truth2 = pd.read_csv(res2_path, header=None, squeeze=True, dtype='bool').values
        truth3 = pd.read_csv(res3_path, header=None, squeeze=True, dtype='bool').values
        truth4 = pd.read_csv(res4_path, header=None, squeeze=True, dtype='bool').values
        truth5 = pd.read_csv(res5_path, header=None, squeeze=True, dtype='bool').values

        gs = GatingStrategy()

        div1 = QuadrantDivider("FSC", "FSC-H", "uncompensated", [28.0654, 70.02725])
        div2 = QuadrantDivider("SSC", "SSC-H", "uncompensated", [17.75])
        div3 = QuadrantDivider("FL1", "FL1-H", "uncompensated", [6.43567])

        divs = [div1, div2, div3]

        quad_gate = gates.QuadrantGate("Quadrant2", None, divs, quadrants_q2, gs)
        gs.add_gate(quad_gate)

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth1, result.get_gate_indices('FSCN-SSCN'))
        np.testing.assert_array_equal(truth2, result.get_gate_indices('FSCD-SSCN-FL1N'))
        np.testing.assert_array_equal(truth3, result.get_gate_indices('FSCP-SSCN-FL1N'))
        np.testing.assert_array_equal(truth4, result.get_gate_indices('FSCD-FL1P'))
        np.testing.assert_array_equal(truth5, result.get_gate_indices('FSCN-SSCP-FL1P'))
