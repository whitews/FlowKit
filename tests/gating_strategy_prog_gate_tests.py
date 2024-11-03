"""
Tests for programmatically adding gates to a GatingStrategy
"""

import copy
import unittest
import numpy as np
import pandas as pd

import flowkit as fk

from tests.test_config import (
    data1_sample,
    range1_gate,
    ell1_coords,
    ell1_cov_mat,
    ell1_dist_square,
    asinh_xform_10000_4_1,
    linear_xform_10000_500,
    logicle_xform_10000__1__4__0_5,
    logicle_xform_10000__0_5__4_5__0,
    poly1_vertices,
    poly1_gate,
    ellipse1_gate,
    range2_gate,
    quad1_gate,
    hyperlog_xform_10000__1__4_5__0,
    comp_matrix_01,
)


class GatingTestCase(unittest.TestCase):
    @staticmethod
    def test_add_min_range_gate():
        res_path = "data/gate_ref/truth/Results_Range1.txt"

        gs = fk.GatingStrategy()

        gs.add_gate(range1_gate, ("root",))

        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("Range1"))

    @staticmethod
    def test_add_rect1_gate():
        gs = fk.GatingStrategy()

        dim1 = fk.Dimension(
            "SSC-H", compensation_ref="uncompensated", range_min=20, range_max=80
        )
        dim2 = fk.Dimension(
            "FL1-H", compensation_ref="uncompensated", range_min=70, range_max=200
        )
        dims = [dim1, dim2]

        rect_gate = fk.gates.RectangleGate("Rectangle1", dims)
        gs.add_gate(rect_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_Rectangle1.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("Rectangle1"))

    @staticmethod
    def test_add_rect2_gate():
        gs = fk.GatingStrategy()

        dim1 = fk.Dimension("SSC-H", compensation_ref="FCS", range_min=20, range_max=80)
        dim2 = fk.Dimension(
            "FL1-H", compensation_ref="FCS", range_min=70, range_max=200
        )
        dims = [dim1, dim2]

        rect_gate = fk.gates.RectangleGate("Rectangle2", dims)
        gs.add_gate(rect_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_Rectangle2.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("Rectangle2"))

    @staticmethod
    def test_add_poly1_gate():
        gs = fk.GatingStrategy()

        gs.add_gate(poly1_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_Polygon1.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("Polygon1"))

    @staticmethod
    def test_add_poly2_gate():
        gs = fk.GatingStrategy()

        dim1 = fk.Dimension("FL1-H", compensation_ref="FCS")
        dim2 = fk.Dimension("FL4-H", compensation_ref="FCS")
        dims = [dim1, dim2]

        vertices = [[20, 10], [120, 10], [120, 160], [20, 160]]

        poly_gate = fk.gates.PolygonGate("Polygon2", dims, vertices)
        gs.add_gate(poly_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_Polygon2.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("Polygon2"))

    @staticmethod
    def test_add_poly3_non_solid_gate():
        gs = fk.GatingStrategy()

        dim1 = fk.Dimension("SSC-H", compensation_ref="uncompensated")
        dim2 = fk.Dimension("FL3-H", compensation_ref="FCS")
        dims = [dim1, dim2]

        vertices = [
            [10, 10],
            [500, 10],
            [500, 390],
            [100, 390],
            [100, 180],
            [200, 180],
            [200, 300],
            [10, 300],
        ]

        poly_gate = fk.gates.PolygonGate("Polygon3NS", dims, vertices)
        gs.add_gate(poly_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_Polygon3NS.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("Polygon3NS"))

    @staticmethod
    def test_add_ellipse1_gate():
        gs = fk.GatingStrategy()

        gs.add_gate(ellipse1_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_Ellipse1.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("Ellipse1"))

    @staticmethod
    def test_add_ellipsoid_3d_gate():
        gs = fk.GatingStrategy()

        dim1 = fk.Dimension("FL3-H", compensation_ref="FCS")
        dim2 = fk.Dimension("FL4-H", compensation_ref="FCS")
        dim3 = fk.Dimension("FL1-H", compensation_ref="FCS")
        dims = [dim1, dim2, dim3]

        coords = [40.3, 30.6, 20.8]
        cov_mat = [[2.5, 7.5, 17.5], [7.5, 7.0, 13.5], [15.5, 13.5, 4.3]]
        dist_square = 1

        poly_gate = fk.gates.EllipsoidGate(
            "Ellipsoid3D", dims, coords, cov_mat, dist_square
        )
        gs.add_gate(poly_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_Ellipsoid3D.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("Ellipsoid3D"))

    @staticmethod
    def test_add_time_range_gate():
        res_path = "data/gate_ref/truth/Results_Range2.txt"

        gs = fk.GatingStrategy()

        gs.add_gate(range2_gate, ("root",))

        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("Range2"))

    @staticmethod
    def test_add_quadrant1_gate():
        res1_path = "data/gate_ref/truth/Results_FL2N-FL4N.txt"
        res2_path = "data/gate_ref/truth/Results_FL2N-FL4P.txt"
        res3_path = "data/gate_ref/truth/Results_FL2P-FL4N.txt"
        res4_path = "data/gate_ref/truth/Results_FL2P-FL4P.txt"

        gs = fk.GatingStrategy()

        gs.add_gate(quad1_gate, ("root",))

        truth1 = pd.read_csv(res1_path, header=None, dtype="bool").squeeze().values
        truth2 = pd.read_csv(res2_path, header=None, dtype="bool").squeeze().values
        truth3 = pd.read_csv(res3_path, header=None, dtype="bool").squeeze().values
        truth4 = pd.read_csv(res4_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth1, result.get_gate_membership("FL2N-FL4N"))
        np.testing.assert_array_equal(truth2, result.get_gate_membership("FL2N-FL4P"))
        np.testing.assert_array_equal(truth3, result.get_gate_membership("FL2P-FL4N"))
        np.testing.assert_array_equal(truth4, result.get_gate_membership("FL2P-FL4P"))

    def test_add_quadrant_gate_relative_percent(self):
        gs = fk.GatingStrategy()

        gs.add_gate(quad1_gate, ("root",))

        result = gs.gate_sample(data1_sample)

        total_percent = (
            result.get_gate_relative_percent("FL2N-FL4N")
            + result.get_gate_relative_percent("FL2N-FL4P")
            + result.get_gate_relative_percent("FL2P-FL4N")
            + result.get_gate_relative_percent("FL2P-FL4P")
        )

        self.assertEqual(100.0, total_percent)

    @staticmethod
    def test_add_quadrant2_gate():
        res1_path = "data/gate_ref/truth/Results_FSCN-SSCN.txt"
        res2_path = "data/gate_ref/truth/Results_FSCD-SSCN-FL1N.txt"
        res3_path = "data/gate_ref/truth/Results_FSCP-SSCN-FL1N.txt"
        res4_path = "data/gate_ref/truth/Results_FSCD-FL1P.txt"
        res5_path = "data/gate_ref/truth/Results_FSCN-SSCP-FL1P.txt"

        truth1 = pd.read_csv(res1_path, header=None, dtype="bool").squeeze().values
        truth2 = pd.read_csv(res2_path, header=None, dtype="bool").squeeze().values
        truth3 = pd.read_csv(res3_path, header=None, dtype="bool").squeeze().values
        truth4 = pd.read_csv(res4_path, header=None, dtype="bool").squeeze().values
        truth5 = pd.read_csv(res5_path, header=None, dtype="bool").squeeze().values

        gs = fk.GatingStrategy()

        div1 = fk.QuadrantDivider("FSC", "FSC-H", "uncompensated", [28.0654, 70.02725])
        div2 = fk.QuadrantDivider("SSC", "SSC-H", "uncompensated", [17.75])
        div3 = fk.QuadrantDivider("FL1", "FL1-H", "uncompensated", [6.43567])

        divs = [div1, div2, div3]

        q2_quad_1 = fk.gates.Quadrant(
            quadrant_id="FSCD-FL1P",
            divider_refs=["FSC", "FL1"],
            divider_ranges=[(28.0654, 70.02725), (6.43567, None)],
        )
        q2_quad_2 = fk.gates.Quadrant(
            quadrant_id="FSCD-SSCN-FL1N",
            divider_refs=["FSC", "SSC", "FL1"],
            divider_ranges=[(28.0654, 70.02725), (None, 17.75), (None, 6.43567)],
        )
        q2_quad_3 = fk.gates.Quadrant(
            quadrant_id="FSCN-SSCN",
            divider_refs=["FSC", "SSC"],
            divider_ranges=[(None, 28.0654), (None, 17.75)],
        )
        q2_quad_4 = fk.gates.Quadrant(
            quadrant_id="FSCN-SSCP-FL1P",
            divider_refs=["FSC", "SSC", "FL1"],
            divider_ranges=[(None, 28.0654), (17.75, None), (6.43567, None)],
        )
        q2_quad_5 = fk.gates.Quadrant(
            quadrant_id="FSCP-SSCN-FL1N",
            divider_refs=["FSC", "SSC", "FL1"],
            divider_ranges=[(70.02725, None), (None, 17.75), (None, 6.43567)],
        )

        quadrants_q2 = [q2_quad_1, q2_quad_2, q2_quad_3, q2_quad_4, q2_quad_5]

        quad_gate = fk.gates.QuadrantGate("Quadrant2", divs, quadrants_q2)
        gs.add_gate(quad_gate, ("root",))

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth1, result.get_gate_membership("FSCN-SSCN"))
        np.testing.assert_array_equal(
            truth2, result.get_gate_membership("FSCD-SSCN-FL1N")
        )
        np.testing.assert_array_equal(
            truth3, result.get_gate_membership("FSCP-SSCN-FL1N")
        )
        np.testing.assert_array_equal(truth4, result.get_gate_membership("FSCD-FL1P"))
        np.testing.assert_array_equal(
            truth5, result.get_gate_membership("FSCN-SSCP-FL1P")
        )

    @staticmethod
    def test_add_ratio_range1_gate():
        gs = fk.GatingStrategy()

        rat_xform_fl2h_fl2a = fk.transforms.RatioTransform(
            ["FL2-H", "FL2-A"], param_a=1, param_b=0, param_c=-1
        )
        gs.add_transform("FL2Rat1", rat_xform_fl2h_fl2a)

        dim_rat1 = fk.RatioDimension(
            "FL2Rat1", compensation_ref="uncompensated", range_min=3, range_max=16.4
        )
        dims = [dim_rat1]

        rect_gate = fk.gates.RectangleGate("RatRange1", dims)
        gs.add_gate(rect_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_RatRange1.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("RatRange1"))

    @staticmethod
    def test_add_ratio_range2_gate():
        gs = fk.GatingStrategy()

        rat_xform_fl2h_fl2a = fk.transforms.RatioTransform(
            ["FL2-H", "FL2-A"], param_a=2.7, param_b=-100, param_c=-300
        )
        gs.add_transform("FL2Rat2", rat_xform_fl2h_fl2a)

        dim_rat2 = fk.RatioDimension(
            "FL2Rat2", compensation_ref="uncompensated", range_min=0.95, range_max=1.05
        )
        dims = [dim_rat2]

        rect_gate = fk.gates.RectangleGate("RatRange2", dims)
        gs.add_gate(rect_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_RatRange2.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("RatRange2"))

    @staticmethod
    def test_add_log_ratio_range1_gate():
        gs = fk.GatingStrategy()

        rat_xform_fl2h_fl2a = fk.transforms.RatioTransform(
            ["FL2-H", "FL2-A"], param_a=1, param_b=0, param_c=-1
        )
        gs.add_transform("FL2Rat1", rat_xform_fl2h_fl2a)

        log_rat_xform = fk.transforms.LogTransform(param_t=100, param_m=2)
        gs.add_transform("MyRatLog", log_rat_xform)

        dim_rat1 = fk.RatioDimension(
            "FL2Rat1",
            compensation_ref="uncompensated",
            transformation_ref="MyRatLog",
            range_min=0.40625,
            range_max=0.6601562,
        )
        dims = [dim_rat1]

        rect_gate = fk.gates.RectangleGate("RatRange1a", dims)
        gs.add_gate(rect_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_RatRange1a.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("RatRange1a"))

    @staticmethod
    def test_add_boolean_and1_gate():
        gs = fk.GatingStrategy()

        gs.add_gate(poly1_gate, ("root",))
        gs.add_gate(range2_gate, ("root",))

        gate_refs = [
            {"ref": "Polygon1", "path": ("root",), "complement": False},
            {"ref": "Range2", "path": ("root",), "complement": False},
        ]

        bool_gate = fk.gates.BooleanGate("And1", "and", gate_refs)
        gs.add_gate(bool_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_And1.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("And1"))

    @staticmethod
    def test_add_boolean_and2_gate():
        gs = fk.GatingStrategy()

        gs.add_gate(range1_gate, ("root",))
        gs.add_gate(poly1_gate, ("root",))
        gs.add_gate(ellipse1_gate, ("root",))

        gate_refs = [
            {"ref": "Range1", "path": ("root",), "complement": False},
            {"ref": "Ellipse1", "path": ("root",), "complement": False},
            {"ref": "Polygon1", "path": ("root",), "complement": False},
        ]

        bool_gate = fk.gates.BooleanGate("And2", "and", gate_refs)
        gs.add_gate(bool_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_And2.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("And2"))

    @staticmethod
    def test_add_boolean_or1_gate():
        gs = fk.GatingStrategy()

        gs.add_gate(range1_gate, ("root",))
        gs.add_gate(poly1_gate, ("root",))
        gs.add_gate(ellipse1_gate, ("root",))

        gate_refs = [
            {"ref": "Range1", "path": ("root",), "complement": False},
            {"ref": "Ellipse1", "path": ("root",), "complement": False},
            {"ref": "Polygon1", "path": ("root",), "complement": False},
        ]

        bool_gate = fk.gates.BooleanGate("Or1", "or", gate_refs)
        gs.add_gate(bool_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_Or1.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("Or1"))

    @staticmethod
    def test_add_boolean_and3_complement_gate():
        gs = fk.GatingStrategy()

        gs.add_gate(range1_gate, ("root",))
        gs.add_gate(poly1_gate, ("root",))
        gs.add_gate(ellipse1_gate, ("root",))

        gate_refs = [
            {"ref": "Range1", "path": ("root",), "complement": False},
            {"ref": "Ellipse1", "path": ("root",), "complement": True},
            {"ref": "Polygon1", "path": ("root",), "complement": False},
        ]

        bool_gate = fk.gates.BooleanGate("And3", "and", gate_refs)
        gs.add_gate(bool_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_And3.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("And3"))

    @staticmethod
    def test_add_boolean_not1_gate():
        gs = fk.GatingStrategy()

        gs.add_gate(ellipse1_gate, ("root",))

        gate_refs = [{"ref": "Ellipse1", "path": ("root",), "complement": False}]

        bool_gate = fk.gates.BooleanGate("Not1", "not", gate_refs)
        gs.add_gate(bool_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_Not1.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("Not1"))

    @staticmethod
    def test_add_boolean_and4_not_gate():
        gs = fk.GatingStrategy()

        gs.add_gate(range1_gate, ("root",))
        gs.add_gate(poly1_gate, ("root",))
        gs.add_gate(ellipse1_gate, ("root",))

        gate1_refs = [{"ref": "Ellipse1", "path": ("root",), "complement": False}]

        bool1_gate = fk.gates.BooleanGate("Not1", "not", gate1_refs)
        gs.add_gate(bool1_gate, ("root",))

        gate2_refs = [
            {"ref": "Range1", "path": ("root",), "complement": False},
            {"ref": "Not1", "path": ("root",), "complement": False},
            {"ref": "Polygon1", "path": ("root",), "complement": False},
        ]

        bool2_gate = fk.gates.BooleanGate("And4", "and", gate2_refs)
        gs.add_gate(bool2_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_And4.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("And4"))

    @staticmethod
    def test_add_boolean_or2_complement_gate():
        gs = fk.GatingStrategy()

        dim1 = fk.Dimension("SSC-H", compensation_ref="FCS", range_min=20, range_max=80)
        dim2 = fk.Dimension(
            "FL1-H", compensation_ref="FCS", range_min=70, range_max=200
        )
        rect_dims = [dim1, dim2]

        rect_gate = fk.gates.RectangleGate("Rectangle2", rect_dims)
        gs.add_gate(rect_gate, ("root",))
        gs.add_gate(quad1_gate, ("root",))

        gate1_refs = [
            {"ref": "Rectangle2", "path": ("root",), "complement": False},
            {"ref": "FL2N-FL4N", "path": ("root", "Quadrant1"), "complement": True},
        ]

        bool1_gate = fk.gates.BooleanGate("Or2", "or", gate1_refs)
        gs.add_gate(bool1_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_Or2.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("Or2"))

    @staticmethod
    def test_add_matrix_poly4_gate():
        gs = fk.GatingStrategy()

        gs.add_comp_matrix("MySpill", comp_matrix_01)

        dim1 = fk.Dimension("PE", compensation_ref="MySpill")
        dim2 = fk.Dimension("PerCP", compensation_ref="MySpill")
        dims = [dim1, dim2]

        poly_gate = fk.gates.PolygonGate("Polygon4", dims, poly1_vertices)
        gs.add_gate(poly_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_Polygon4.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("Polygon4"))

    @staticmethod
    def test_add_matrix_rect3_gate():
        gs = fk.GatingStrategy()

        gs.add_comp_matrix("MySpill", comp_matrix_01)

        dim1 = fk.Dimension(
            "FITC", compensation_ref="MySpill", range_min=5, range_max=70
        )
        dim2 = fk.Dimension(
            "PE", compensation_ref="MySpill", range_min=9, range_max=208
        )
        dims = [dim1, dim2]

        rect_gate = fk.gates.RectangleGate("Rectangle3", dims)
        gs.add_gate(rect_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_Rectangle3.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("Rectangle3"))

    @staticmethod
    def test_add_matrix_rect4_gate():
        gs = fk.GatingStrategy()

        gs.add_comp_matrix("MySpill", comp_matrix_01)

        dim1 = fk.Dimension(
            "PerCP", compensation_ref="MySpill", range_min=7, range_max=90
        )
        dim2 = fk.Dimension(
            "FSC-H", compensation_ref="uncompensated", range_min=10, range_max=133
        )
        dims = [dim1, dim2]

        rect_gate = fk.gates.RectangleGate("Rectangle4", dims)
        gs.add_gate(rect_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_Rectangle4.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("Rectangle4"))

    @staticmethod
    def test_add_matrix_rect5_gate():
        gs = fk.GatingStrategy()

        gs.add_comp_matrix("MySpill", comp_matrix_01)

        dim1 = fk.Dimension(
            "PerCP", compensation_ref="MySpill", range_min=7, range_max=90
        )
        dim2 = fk.Dimension("FSC-H", compensation_ref="uncompensated", range_min=10)
        dims = [dim1, dim2]

        rect_gate = fk.gates.RectangleGate("Rectangle5", dims)
        gs.add_gate(rect_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_Rectangle5.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("Rectangle5"))

    @staticmethod
    def test_add_transform_asinh_range1_gate():
        gs = fk.GatingStrategy()

        gs.add_transform("AsinH_10000_4_1", asinh_xform_10000_4_1)

        dim1 = fk.Dimension(
            "FL1-H", "uncompensated", "AsinH_10000_4_1", range_min=0.37, range_max=0.63
        )
        dims = [dim1]

        rect_gate = fk.gates.RectangleGate("ScaleRange1", dims)
        gs.add_gate(rect_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_ScaleRange1.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("ScaleRange1"))

    @staticmethod
    def test_add_transform_hyperlog_range2_gate():
        gs = fk.GatingStrategy()

        gs.add_transform("Hyperlog_10000_1_4.5_0", hyperlog_xform_10000__1__4_5__0)

        dim1 = fk.Dimension(
            "FL1-H",
            "uncompensated",
            "Hyperlog_10000_1_4.5_0",
            range_min=0.37,
            range_max=0.63,
        )
        dims = [dim1]

        rect_gate = fk.gates.RectangleGate("ScaleRange2", dims)
        gs.add_gate(rect_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_ScaleRange2.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("ScaleRange2"))

    @staticmethod
    def test_add_transform_linear_range3_gate():
        gs = fk.GatingStrategy()

        gs.add_transform("Linear_10000_500", linear_xform_10000_500)

        dim1 = fk.Dimension(
            "FL1-H",
            "uncompensated",
            "Linear_10000_500",
            range_min=0.049,
            range_max=0.055,
        )
        dims = [dim1]

        rect_gate = fk.gates.RectangleGate("ScaleRange3", dims)
        gs.add_gate(rect_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_ScaleRange3.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("ScaleRange3"))

    @staticmethod
    def test_add_transform_logicle_range4_gate():
        gs = fk.GatingStrategy()

        gs.add_transform("Logicle_10000_0.5_4.5_0", logicle_xform_10000__0_5__4_5__0)

        dim1 = fk.Dimension(
            "FL1-H",
            "uncompensated",
            "Logicle_10000_0.5_4.5_0",
            range_min=0.37,
            range_max=0.63,
        )
        dims = [dim1]

        rect_gate = fk.gates.RectangleGate("ScaleRange4", dims)
        gs.add_gate(rect_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_ScaleRange4.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("ScaleRange4"))

    @staticmethod
    def test_add_transform_logicle_range5_gate():
        gs = fk.GatingStrategy()

        gs.add_transform("Logicle_10000_1_4_0.5", logicle_xform_10000__1__4__0_5)

        dim1 = fk.Dimension(
            "FL1-H",
            "uncompensated",
            "Logicle_10000_1_4_0.5",
            range_min=0.37,
            range_max=0.63,
        )
        dims = [dim1]

        rect_gate = fk.gates.RectangleGate("ScaleRange5", dims)
        gs.add_gate(rect_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_ScaleRange5.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("ScaleRange5"))

    @staticmethod
    def test_add_transform_log_range6_gate():
        gs = fk.GatingStrategy()

        xform = fk.transforms.LogTransform(param_t=10000, param_m=5)
        gs.add_transform("Logarithmic_10000_5", xform)

        dim1 = fk.Dimension(
            "FL1-H",
            "uncompensated",
            "Logarithmic_10000_5",
            range_min=0.37,
            range_max=0.63,
        )
        dims = [dim1]

        rect_gate = fk.gates.RectangleGate("ScaleRange6", dims)
        gs.add_gate(rect_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_ScaleRange6.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("ScaleRange6"))

    @staticmethod
    def test_add_matrix_transform_asinh_range1c_gate():
        gs = fk.GatingStrategy()

        gs.add_comp_matrix("MySpill", comp_matrix_01)

        gs.add_transform("AsinH_10000_4_1", asinh_xform_10000_4_1)

        dim1 = fk.Dimension(
            "FITC", "MySpill", "AsinH_10000_4_1", range_min=0.37, range_max=0.63
        )
        dims = [dim1]

        rect_gate = fk.gates.RectangleGate("ScaleRange1c", dims)
        gs.add_gate(rect_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_ScaleRange1c.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("ScaleRange1c"))

    @staticmethod
    def test_add_matrix_transform_hyperlog_range2c_gate():
        gs = fk.GatingStrategy()

        gs.add_comp_matrix("MySpill", comp_matrix_01)

        gs.add_transform("Hyperlog_10000_1_4.5_0", hyperlog_xform_10000__1__4_5__0)

        dim1 = fk.Dimension(
            "FITC", "MySpill", "Hyperlog_10000_1_4.5_0", range_min=0.37, range_max=0.63
        )
        dims = [dim1]

        rect_gate = fk.gates.RectangleGate("ScaleRange2c", dims)
        gs.add_gate(rect_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_ScaleRange2c.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("ScaleRange2c"))

    @staticmethod
    def test_add_matrix_transform_linear_range3c_gate():
        gs = fk.GatingStrategy()

        gs.add_comp_matrix("MySpill", comp_matrix_01)

        gs.add_transform("Linear_10000_500", linear_xform_10000_500)

        dim1 = fk.Dimension(
            "FITC", "MySpill", "Linear_10000_500", range_min=0.049, range_max=0.055
        )
        dims = [dim1]

        rect_gate = fk.gates.RectangleGate("ScaleRange3c", dims)
        gs.add_gate(rect_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_ScaleRange3c.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("ScaleRange3c"))

    @staticmethod
    def test_add_matrix_transform_logicle_range4c_gate():
        gs = fk.GatingStrategy()

        gs.add_comp_matrix("MySpill", comp_matrix_01)

        gs.add_transform("Logicle_10000_0.5_4.5_0", logicle_xform_10000__0_5__4_5__0)

        dim1 = fk.Dimension(
            "FITC", "MySpill", "Logicle_10000_0.5_4.5_0", range_min=0.37, range_max=0.63
        )
        dims = [dim1]

        rect_gate = fk.gates.RectangleGate("ScaleRange4c", dims)
        gs.add_gate(rect_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_ScaleRange4c.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("ScaleRange4c"))

    @staticmethod
    def test_add_matrix_transform_logicle_range5c_gate():
        gs = fk.GatingStrategy()

        gs.add_comp_matrix("MySpill", comp_matrix_01)

        gs.add_transform("Logicle_10000_1_4_0.5", logicle_xform_10000__1__4__0_5)

        dim1 = fk.Dimension(
            "FITC", "MySpill", "Logicle_10000_1_4_0.5", range_min=0.37, range_max=0.63
        )
        dims = [dim1]

        rect_gate = fk.gates.RectangleGate("ScaleRange5c", dims)
        gs.add_gate(rect_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_ScaleRange5c.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("ScaleRange5c"))

    @staticmethod
    def test_add_matrix_transform_asinh_range6c_gate():
        gs = fk.GatingStrategy()

        gs.add_comp_matrix("MySpill", comp_matrix_01)

        gs.add_transform("AsinH_10000_4_1", asinh_xform_10000_4_1)

        dim1 = fk.Dimension(
            "PE", "MySpill", "AsinH_10000_4_1", range_min=0.09, range_max=0.36
        )
        dims = [dim1]

        rect_gate = fk.gates.RectangleGate("ScaleRange6c", dims)
        gs.add_gate(rect_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_ScaleRange6c.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("ScaleRange6c"))

    @staticmethod
    def test_add_matrix_transform_hyperlog_range7c_gate():
        gs = fk.GatingStrategy()

        gs.add_comp_matrix("MySpill", comp_matrix_01)

        gs.add_transform("Hyperlog_10000_1_4.5_0", hyperlog_xform_10000__1__4_5__0)

        dim1 = fk.Dimension(
            "PE", "MySpill", "Hyperlog_10000_1_4.5_0", range_min=0.09, range_max=0.36
        )
        dims = [dim1]

        rect_gate = fk.gates.RectangleGate("ScaleRange7c", dims)
        gs.add_gate(rect_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_ScaleRange7c.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("ScaleRange7c"))

    @staticmethod
    def test_add_matrix_transform_logicle_range8c_gate():
        gs = fk.GatingStrategy()

        gs.add_comp_matrix("MySpill", comp_matrix_01)

        gs.add_transform("Logicle_10000_1_4_0.5", logicle_xform_10000__1__4__0_5)

        dim1 = fk.Dimension(
            "PE", "MySpill", "Logicle_10000_1_4_0.5", range_min=0.09, range_max=0.36
        )
        dims = [dim1]

        rect_gate = fk.gates.RectangleGate("ScaleRange8c", dims)
        gs.add_gate(rect_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_ScaleRange8c.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("ScaleRange8c"))

    @staticmethod
    def test_add_matrix_transform_logicle_rect1_gate():
        gs = fk.GatingStrategy()

        gs.add_comp_matrix("MySpill", comp_matrix_01)

        gs.add_transform("Logicle_10000_0.5_4.5_0", logicle_xform_10000__0_5__4_5__0)

        dim1 = fk.Dimension(
            "PE", "MySpill", "Logicle_10000_0.5_4.5_0", range_min=0.31, range_max=0.69
        )
        dim2 = fk.Dimension(
            "PerCP",
            "MySpill",
            "Logicle_10000_0.5_4.5_0",
            range_min=0.27,
            range_max=0.73,
        )
        dims = [dim1, dim2]

        rect_gate = fk.gates.RectangleGate("ScaleRect1", dims)
        gs.add_gate(rect_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_ScaleRect1.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("ScaleRect1"))

    @staticmethod
    def test_add_parent_poly1_boolean_and2_gate():
        gs = fk.GatingStrategy()

        gs.add_gate(poly1_gate, ("root",))

        dim3 = fk.Dimension("FL3-H", compensation_ref="FCS")
        dim4 = fk.Dimension("FL4-H", compensation_ref="FCS")
        dims2 = [dim3, dim4]

        ellipse_gate = fk.gates.EllipsoidGate(
            "Ellipse1", dims2, ell1_coords, ell1_cov_mat, ell1_dist_square
        )
        gs.add_gate(ellipse_gate, ("root",))
        gs.add_gate(range1_gate, ("root",))

        gate_refs = [
            {"ref": "Range1", "path": ("root",), "complement": False},
            {"ref": "Ellipse1", "path": ("root",), "complement": False},
        ]

        bool_gate = fk.gates.BooleanGate("ParAnd2", "and", gate_refs)
        gs.add_gate(bool_gate, ("root", "Polygon1"))

        res_path = "data/gate_ref/truth/Results_ParAnd2.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("ParAnd2"))

    @staticmethod
    def test_add_parent_range1_boolean_and3_gate():
        gs = fk.GatingStrategy()

        gs.add_gate(poly1_gate, ("root",))
        gs.add_gate(ellipse1_gate, ("root",))
        gs.add_gate(range1_gate, ("root",))

        gate_refs = [
            {"ref": "Polygon1", "path": ("root",), "complement": False},
            {"ref": "Ellipse1", "path": ("root",), "complement": True},
        ]

        bool_gate = fk.gates.BooleanGate("ParAnd3", "and", gate_refs)
        gs.add_gate(bool_gate, ("root", "Range1"))

        res_path = "data/gate_ref/truth/Results_ParAnd3.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("ParAnd3"))

    @staticmethod
    def test_add_parent_rect1_rect_par1_gate():
        gs = fk.GatingStrategy()

        gs.add_comp_matrix("MySpill", comp_matrix_01)

        gs.add_transform("Logicle_10000_0.5_4.5_0", logicle_xform_10000__0_5__4_5__0)
        gs.add_transform("Hyperlog_10000_1_4.5_0", hyperlog_xform_10000__1__4_5__0)

        gs.add_gate(poly1_gate, ("root",))

        dim1 = fk.Dimension(
            "PE", "MySpill", "Logicle_10000_0.5_4.5_0", range_min=0.31, range_max=0.69
        )
        dim2 = fk.Dimension(
            "PerCP",
            "MySpill",
            "Logicle_10000_0.5_4.5_0",
            range_min=0.27,
            range_max=0.73,
        )
        dims1 = [dim1, dim2]

        rect_gate1 = fk.gates.RectangleGate("ScaleRect1", dims1)
        gs.add_gate(rect_gate1, ("root",))

        dim3 = fk.Dimension(
            "FITC", "MySpill", "Hyperlog_10000_1_4.5_0", range_min=0.12, range_max=0.43
        )
        dims2 = [dim3]

        rect_gate2 = fk.gates.RectangleGate("ScalePar1", dims2)
        gs.add_gate(rect_gate2, ("root", "ScaleRect1"))

        res_path = "data/gate_ref/truth/Results_ScalePar1.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(truth, result.get_gate_membership("ScalePar1"))

    @staticmethod
    def test_add_parent_quadrant_rect_gate():
        gs = fk.GatingStrategy()
        gs.add_gate(quad1_gate, ("root",))

        dim1 = fk.Dimension("FL2-H", "uncompensated", None, range_min=6, range_max=14.4)
        dim2 = fk.Dimension("FL4-H", "uncompensated", None, range_min=7, range_max=16)
        dims1 = [dim1, dim2]

        rect_gate1 = fk.gates.RectangleGate("ParRectangle1", dims1)
        gs.add_gate(rect_gate1, ("root", "Quadrant1", "FL2P-FL4P"))

        res_path = "data/gate_ref/truth/Results_ParQuadRect.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        result = gs.gate_sample(data1_sample)

        np.testing.assert_array_equal(
            truth, result.get_gate_membership("ParRectangle1")
        )

    def test_quad_gate_with_parent_gate(self):
        gs = fk.GatingStrategy()

        dim1 = fk.Dimension(
            "SSC-H", compensation_ref="uncompensated", range_min=20, range_max=80
        )
        dim2 = fk.Dimension(
            "FL1-H", compensation_ref="uncompensated", range_min=70, range_max=200
        )
        dims = [dim1, dim2]

        rect_gate = fk.gates.RectangleGate("Rectangle1", dims)
        gs.add_gate(rect_gate, ("root",))

        quad_gate_copy = copy.deepcopy(quad1_gate)

        gs.add_gate(quad_gate_copy, ("root", "Rectangle1"))

        result = gs.gate_sample(data1_sample)

        # just ensure the above succeeded and made a GatingResults object w/ a DataFrame report
        self.assertIsInstance(result.report, pd.DataFrame)
