"""
Tests for GatingStrategy Class
"""
import unittest
import flowkit as fk
from tests.test_config import (
    data1_sample,
    poly1_gate,
    hyperlog_xform_10000__1__4_5__0,
    logicle_xform_10000__0_5__4_5__0,
    sample_with_spill,
    comp_matrix_01
)


class GatingStrategyTestCase(unittest.TestCase):
    def test_add_gate_non_gate_class(self):
        gs = fk.GatingStrategy()
        self.assertRaises(TypeError, gs.add_gate, "not a gate class")

    def test_add_duplicate_gate_id_raises(self):
        gs = fk.GatingStrategy()
        gs.add_gate(poly1_gate, ('root',))

        self.assertRaises(fk.exceptions.GateTreeError, gs.add_gate, poly1_gate, ('root',))

    def test_get_gate_raises_ValueError(self):
        gs = fk.GatingStrategy()
        gs.add_gate(poly1_gate, ('root',))

        self.assertRaises(fk.exceptions.GateReferenceError, gs.get_gate, 'nonexistent-gate')

    def test_get_parent_gate_is_none(self):
        gs = fk.GatingStrategy()
        gs.add_gate(poly1_gate, ('root',))

        parent_gate_id = gs.get_parent_gate_id('Polygon1')

        self.assertIsNone(parent_gate_id)

    def test_add_transform_non_transform_class(self):
        gs = fk.GatingStrategy()
        self.assertRaises(TypeError, gs.add_transform, transform_id="not a transform class", transform=dict())

    def test_add_duplicate_transform_id(self):
        gs = fk.GatingStrategy()
        gs.add_transform('Logicle_10000_0.5_4.5_0', logicle_xform_10000__0_5__4_5__0)

        self.assertRaises(KeyError, gs.add_transform, 'Logicle_10000_0.5_4.5_0', logicle_xform_10000__0_5__4_5__0)

    def test_add_matrix_non_matrix_class(self):
        gs = fk.GatingStrategy()
        self.assertRaises(TypeError, gs.add_comp_matrix, matrix_id="not a matrix class", matrix=dict())

    def test_add_duplicate_matrix_id(self):
        gs = fk.GatingStrategy()
        gs.add_comp_matrix('MySpill', comp_matrix_01)

        self.assertRaises(KeyError, gs.add_comp_matrix, 'MySpill', comp_matrix_01)

    def test_add_invalid_matrix_id(self):
        gs = fk.GatingStrategy()

        # Both 'uncompensated' and 'FCS' are reserved matrix IDs in GatingML 2.0
        self.assertRaises(ValueError, gs.add_comp_matrix, 'uncompensated', comp_matrix_01)
        self.assertRaises(ValueError, gs.add_comp_matrix, 'FCS', comp_matrix_01)

    def test_fcs_defined_spill(self):
        gs = fk.GatingStrategy()

        asinh_xform = fk.transforms.AsinhTransform(param_t=262144, param_m=4.0, param_a=0.0)
        gs.add_transform('asinh', asinh_xform)

        dim_cd3 = fk.Dimension(
            'CD3 APC-H7 FLR-A',
            range_max=-0.4,
            transformation_ref='asinh',
            compensation_ref='FCS'
        )

        cd3_rect_gate = fk.gates.RectangleGate('cd3_low', dimensions=[dim_cd3])
        gs.add_gate(cd3_rect_gate, gate_path=('root',))

        gating_result = gs.gate_sample(sample_with_spill)
        cd3_low_count = gating_result.get_gate_count('cd3_low')

        self.assertEqual(cd3_low_count, 71)

    def test_get_max_depth(self):
        gml_path = 'data/gate_ref/gml/gml_all_gates.xml'
        gs = fk.parse_gating_xml(gml_path)
        gs_depth = gs.get_max_depth()

        self.assertEqual(gs_depth, 2)

    def test_absolute_percent(self):
        gs = fk.GatingStrategy()

        gs.add_comp_matrix('MySpill', comp_matrix_01)

        gs.add_transform('Logicle_10000_0.5_4.5_0', logicle_xform_10000__0_5__4_5__0)
        gs.add_transform('Hyperlog_10000_1_4.5_0', hyperlog_xform_10000__1__4_5__0)

        gs.add_gate(poly1_gate, ('root',))

        dim1 = fk.Dimension('PE', 'MySpill', 'Logicle_10000_0.5_4.5_0', range_min=0.31, range_max=0.69)
        dim2 = fk.Dimension('PerCP', 'MySpill', 'Logicle_10000_0.5_4.5_0', range_min=0.27, range_max=0.73)
        dims1 = [dim1, dim2]

        rect_gate1 = fk.gates.RectangleGate('ScaleRect1', dims1)
        gs.add_gate(rect_gate1, ('root',))

        dim3 = fk.Dimension('FITC', 'MySpill', 'Hyperlog_10000_1_4.5_0', range_min=0.12, range_max=0.43)
        dims2 = [dim3]

        rect_gate2 = fk.gates.RectangleGate('ScalePar1', dims2)
        gs.add_gate(rect_gate2, ('root', 'ScaleRect1'))

        result = gs.gate_sample(data1_sample)
        parent_gate_name, parent_gate_path = gs.get_parent_gate_id(rect_gate2.gate_name)
        parent_gate = gs.get_gate(parent_gate_name, parent_gate_path)
        parent_gate_count = result.get_gate_count(parent_gate.gate_name)
        gate_count = result.get_gate_count(rect_gate2.gate_name)
        gate_abs_pct = result.get_gate_absolute_percent(rect_gate2.gate_name)
        gate_rel_pct = result.get_gate_relative_percent(rect_gate2.gate_name)

        true_count = 558
        true_abs_pct = (558 / data1_sample.event_count) * 100
        true_rel_pct = (558 / float(parent_gate_count)) * 100

        self.assertEqual(true_count, gate_count)
        self.assertEqual(true_abs_pct, gate_abs_pct)
        self.assertEqual(true_rel_pct, gate_rel_pct)

    def test_clear_cache(self):
        gs = fk.GatingStrategy()

        gs.add_comp_matrix('MySpill', comp_matrix_01)

        gs.add_transform('Logicle_10000_0.5_4.5_0', logicle_xform_10000__0_5__4_5__0)
        gs.add_transform('Hyperlog_10000_1_4.5_0', hyperlog_xform_10000__1__4_5__0)

        gs.add_gate(poly1_gate, ('root',))

        dim1 = fk.Dimension('PE', 'MySpill', 'Logicle_10000_0.5_4.5_0', range_min=0.31, range_max=0.69)
        dim2 = fk.Dimension('PerCP', 'MySpill', 'Logicle_10000_0.5_4.5_0', range_min=0.27, range_max=0.73)
        dims1 = [dim1, dim2]

        rect_gate1 = fk.gates.RectangleGate('ScaleRect1', dims1)
        gs.add_gate(rect_gate1, ('root',))

        dim3 = fk.Dimension('FITC', 'MySpill', 'Hyperlog_10000_1_4.5_0', range_min=0.12, range_max=0.43)
        dims2 = [dim3]

        rect_gate2 = fk.gates.RectangleGate('ScalePar1', dims2)
        gs.add_gate(rect_gate2, ('root', 'ScaleRect1'))

        _ = gs.gate_sample(data1_sample, cache_events=True)

        pre_proc_events = gs._cached_preprocessed_events

        truth_key_set = {
            ('MySpill', None, None),
            ('MySpill', 'Logicle_10000_0.5_4.5_0', 3),
            ('MySpill', 'Logicle_10000_0.5_4.5_0', 4),
            ('MySpill', 'Hyperlog_10000_1_4.5_0', 2)
        }

        self.assertSetEqual(set(pre_proc_events['B07'].keys()), truth_key_set)

        gs.clear_cache()
        pre_proc_events = gs._cached_preprocessed_events

        self.assertEqual(pre_proc_events, {})

    def test_cache_preprocessed_events(self):
        gs = fk.GatingStrategy()

        gs.add_comp_matrix('MySpill', comp_matrix_01)

        gs.add_transform('Logicle_10000_0.5_4.5_0', logicle_xform_10000__0_5__4_5__0)
        gs.add_transform('Hyperlog_10000_1_4.5_0', hyperlog_xform_10000__1__4_5__0)

        gs.add_gate(poly1_gate, ('root',))

        dim1 = fk.Dimension('PE', 'MySpill', 'Logicle_10000_0.5_4.5_0', range_min=0.31, range_max=0.69)
        dim2 = fk.Dimension('PerCP', 'MySpill', 'Logicle_10000_0.5_4.5_0', range_min=0.27, range_max=0.73)
        dims1 = [dim1, dim2]

        rect_gate1 = fk.gates.RectangleGate('ScaleRect1', dims1)
        gs.add_gate(rect_gate1, ('root',))

        dim3 = fk.Dimension('FITC', 'MySpill', 'Hyperlog_10000_1_4.5_0', range_min=0.12, range_max=0.43)
        dims2 = [dim3]

        rect_gate2 = fk.gates.RectangleGate('ScalePar1', dims2)
        gs.add_gate(rect_gate2, ('root', 'ScaleRect1'))

        _ = gs.gate_sample(data1_sample, cache_events=True)

        pre_proc_events = gs._cached_preprocessed_events

        truth_key_set = {
            ('MySpill', None, None),
            ('MySpill', 'Logicle_10000_0.5_4.5_0', 3),
            ('MySpill', 'Logicle_10000_0.5_4.5_0', 4),
            ('MySpill', 'Hyperlog_10000_1_4.5_0', 2)
        }

        self.assertSetEqual(set(pre_proc_events['B07'].keys()), truth_key_set)
