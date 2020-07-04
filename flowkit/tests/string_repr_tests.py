"""
Unit tests for string representations
"""
import unittest
import flowkit as fk
from .prog_gating_unit_tests import comp_matrix_01, logicle_xform1, hyperlog_xform1, poly1_gate

data1_fcs_path = 'examples/gate_ref/data1.fcs'
data1_sample = fk.Sample(data1_fcs_path)


class StringReprTestCase(unittest.TestCase):
    """Tests related to string representations of FlowKit classes"""
    def test_vert_repr(self):
        vert = fk.Vertex([500, 5])
        vert_string = "Vertex([500, 5])"

        self.assertEqual(repr(vert), vert_string)

    def test_dim_repr(self):
        poly1_dim1 = fk.Dimension('FL2-H', compensation_ref='FCS')
        dim_string = "Dimension(label: FL2-H)"

        self.assertEqual(repr(poly1_dim1), dim_string)

    def test_ratio_dim_repr(self):
        dim_rat1 = fk.RatioDimension(
            'FL2Rat1',
            compensation_ref='uncompensated',
            range_min=3,
            range_max=16.4
        )
        dim_string = "RatioDimension(ratio_reference: FL2Rat1)"

        self.assertEqual(repr(dim_rat1), dim_string)

    def test_quad_div_repr(self):
        quad1_div1 = fk.QuadrantDivider('FL2', 'FL2-H', 'FCS', [12.14748])
        quad_div_string = "QuadrantDivider(id: FL2, dim_ref: FL2-H)"

        self.assertEqual(repr(quad1_div1), quad_div_string)

    def test_wsp_log_xform_repr(self):
        xform = fk.transforms.WSPLogTransform('wsp_log', offset=0.5, decades=4.5)
        xform_string = "WSPLogTransform(wsp_log, offset: 0.5, decades: 4.5)"

        self.assertEqual(repr(xform), xform_string)

    def test_sample_repr(self):
        fcs_file_path = "examples/gate_ref/data1.fcs"
        sample = fk.Sample(fcs_path_or_data=fcs_file_path)
        sample_string = "Sample(v2.0, B07, 8 channels, 13367 events)"

        self.assertEqual(repr(sample), sample_string)

    def test_gating_strategy_repr(self):
        gs = fk.GatingStrategy()

        gs.add_comp_matrix(comp_matrix_01)

        gs.add_transform(logicle_xform1)
        gs.add_transform(hyperlog_xform1)

        gs.add_gate(poly1_gate)

        dim1 = fk.Dimension('PE', 'MySpill', 'Logicle_10000_0.5_4.5_0', range_min=0.31, range_max=0.69)
        dim2 = fk.Dimension('PerCP', 'MySpill', 'Logicle_10000_0.5_4.5_0', range_min=0.27, range_max=0.73)
        dims1 = [dim1, dim2]

        rect_gate1 = fk.gates.RectangleGate('ScaleRect1', None, dims1)
        gs.add_gate(rect_gate1)

        dim3 = fk.Dimension('FITC', 'MySpill', 'Hyperlog_10000_1_4.5_0', range_min=0.12, range_max=0.43)
        dims2 = [dim3]

        rect_gate2 = fk.gates.RectangleGate('ScalePar1', 'ScaleRect1', dims2)
        gs.add_gate(rect_gate2)

        gs_string = "GatingStrategy(3 gates, 2 transforms, 1 compensations)"

        self.assertEqual(repr(gs), gs_string)
