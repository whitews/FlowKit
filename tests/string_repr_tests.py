"""
Unit tests for string representations
"""

import unittest
import flowkit as fk

from tests.test_config import (
    data1_sample,
    comp_matrix_01,
    poly1_gate,
    range1_gate,
    ellipse1_gate,
    quad1_gate,
    logicle_xform_10000__0_5__4_5__0,
    hyperlog_xform_10000__1__4_5__0,
)


class StringReprTestCase(unittest.TestCase):
    """Tests related to string representations of FlowKit classes"""

    def test_dim_repr(self):
        poly1_dim1 = fk.Dimension("FL2-H", compensation_ref="FCS")
        dim_string = "Dimension(id: FL2-H)"

        self.assertEqual(repr(poly1_dim1), dim_string)

    def test_ratio_dim_repr(self):
        dim_rat1 = fk.RatioDimension(
            "FL2Rat1", compensation_ref="uncompensated", range_min=3, range_max=16.4
        )
        dim_string = "RatioDimension(ratio_reference: FL2Rat1)"

        self.assertEqual(repr(dim_rat1), dim_string)

    def test_quad_div_repr(self):
        quad1_div1 = fk.QuadrantDivider("FL2", "FL2-H", "FCS", [12.14748])
        quad_div_string = "QuadrantDivider(id: FL2, dim_ref: FL2-H)"

        self.assertEqual(repr(quad1_div1), quad_div_string)

    def test_matrix_repr(self):
        comp_mat = comp_matrix_01
        comp_mat_string = "Matrix(dims: 3)"

        self.assertEqual(repr(comp_mat), comp_mat_string)

    def test_linear_transform_repr(self):
        xform = fk.transforms.LinearTransform(param_t=10000.0, param_a=0.0)
        xform_string = "LinearTransform(t: 10000.0, a: 0.0)"

        self.assertEqual(repr(xform), xform_string)

    def test_log_transform_repr(self):
        xform = fk.transforms.LogTransform(param_t=10000.0, param_m=4.5)
        xform_string = "LogTransform(t: 10000.0, m: 4.5)"

        self.assertEqual(repr(xform), xform_string)

    def test_ratio_transform_repr(self):
        ratio_dims = ["FL1-H", "FL2-H"]
        xform = fk.transforms.RatioTransform(
            ratio_dims, param_a=1.0, param_b=0.0, param_c=0.0
        )
        xform_string = "RatioTransform(FL1-H / FL2-H, a: 1.0, b: 0.0, c: 0.0)"

        self.assertEqual(repr(xform), xform_string)

    def test_hyperlog_transform_repr(self):
        xform = fk.transforms.HyperlogTransform(
            param_t=10000.0, param_w=0.5, param_m=4.5, param_a=0.0
        )
        xform_string = "HyperlogTransform(t: 10000.0, w: 0.5, m: 4.5, a: 0.0)"

        self.assertEqual(repr(xform), xform_string)

    def test_logicle_transform_repr(self):
        xform = fk.transforms.LogicleTransform(
            param_t=10000.0, param_w=0.5, param_m=4.5, param_a=0.0
        )
        xform_string = "LogicleTransform(t: 10000.0, w: 0.5, m: 4.5, a: 0.0)"

        self.assertEqual(repr(xform), xform_string)

    def test_asinh_transform_repr(self):
        xform = fk.transforms.AsinhTransform(param_t=10000.0, param_m=4.5, param_a=0.0)
        xform_string = "AsinhTransform(t: 10000.0, m: 4.5, a: 0.0)"

        self.assertEqual(repr(xform), xform_string)

    def test_wsp_log_xform_repr(self):
        xform = fk.transforms.WSPLogTransform(offset=0.5, decades=4.5)
        xform_string = "WSPLogTransform(offset: 0.5, decades: 4.5)"

        self.assertEqual(repr(xform), xform_string)

    def test_sample_repr(self):
        fcs_file_path = "data/gate_ref/data1.fcs"
        sample = fk.Sample(fcs_path_or_data=fcs_file_path)
        sample_string = "Sample(v2.0, B07, 8 channels, 13367 events)"

        self.assertEqual(repr(sample), sample_string)

    def test_rect_gate_repr(self):
        gate = range1_gate

        repr_string = "RectangleGate(Range1, dims: 1)"

        self.assertEqual(repr(gate), repr_string)

    def test_poly_gate_repr(self):
        gate = poly1_gate

        repr_string = "PolygonGate(Polygon1, vertices: 3)"

        self.assertEqual(repr(gate), repr_string)

    def test_ellipsoid_gate_repr(self):
        gate = ellipse1_gate

        repr_string = "EllipsoidGate(Ellipse1, coords: [12.99701, 16.22941])"

        self.assertEqual(repr(gate), repr_string)

    def test_quad_gate_repr(self):
        gate = quad1_gate

        repr_string = "QuadrantGate(Quadrant1, quadrants: 4)"

        self.assertEqual(repr(gate), repr_string)

    def test_bool_gate_repr(self):
        gate_refs = [
            {"ref": "Polygon1", "path": ("root",), "complement": False},
            {"ref": "Range2", "path": ("root",), "complement": False},
        ]

        gate = fk.gates.BooleanGate("And1", "and", gate_refs)

        repr_string = "BooleanGate(And1, type: and)"

        self.assertEqual(repr(gate), repr_string)

    def test_gating_strategy_repr(self):
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

        gs_string = "GatingStrategy(3 gates, 2 transforms, 1 compensations)"

        self.assertEqual(repr(gs), gs_string)

    def test_session_repr(self):
        session = fk.Session(fcs_samples=data1_sample)

        session_string = "Session(1 samples)"

        self.assertEqual(repr(session), session_string)

    def test_workspace_repr(self):
        wsp_path = "data/simple_line_example/simple_poly_and_rect.wsp"
        fcs_path = "data/simple_line_example/data_set_simple_line_100.fcs"

        wsp = fk.Workspace(wsp_path, fcs_samples=fcs_path)

        wsp_string = "Workspace(1 samples loaded, 2 sample groups)"

        self.assertEqual(repr(wsp), wsp_string)
