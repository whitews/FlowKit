import unittest
import numpy as np
import flowkit as fk


data1_fcs_path = 'examples/gate_ref/data1.fcs'
data1_sample = fk.Sample(data1_fcs_path)

poly1_vertices = [
    fk.Vertex([5, 5]),
    fk.Vertex([500, 5]),
    fk.Vertex([500, 500])
]
poly1_dim1 = fk.Dimension('FL2-H', compensation_ref='FCS')
poly1_dim2 = fk.Dimension('FL3-H', compensation_ref='FCS')
poly1_dims1 = [poly1_dim1, poly1_dim2]
poly1_gate = fk.gates.PolygonGate('Polygon1', None, poly1_dims1, poly1_vertices)

hyperlog_xform1 = fk.transforms.HyperlogTransform(
    'Hyperlog_10000_1_4.5_0',
    param_t=10000,
    param_w=1,
    param_m=4.5,
    param_a=0
)

logicle_xform1 = fk.transforms.LogicleTransform(
    'Logicle_10000_0.5_4.5_0',
    param_t=10000,
    param_w=0.5,
    param_m=4.5,
    param_a=0
)

spill01_fluoros = ['FITC', 'PE', 'PerCP']
spill01_detectors = ['FL1-H', 'FL2-H', 'FL3-H']
spill01_data = np.array(
    [
        [1, 0.02, 0.06],
        [0.11, 1, 0.07],
        [0.09, 0.01, 1]
    ]
)
comp_matrix_01 = fk.Matrix('MySpill', spill01_data, spill01_detectors, spill01_fluoros)


class GatingStrategyTestCase(unittest.TestCase):
    def test_add_gate_non_gate_class(self):
        gs = fk.GatingStrategy()
        self.assertRaises(ValueError, gs.add_gate, "not a gate class")

    def test_add_duplicate_gate_id(self):
        gs = fk.GatingStrategy()
        gs.add_gate(poly1_gate)

        self.assertRaises(KeyError, gs.add_gate, poly1_gate)

    def test_add_transform_non_transform_class(self):
        gs = fk.GatingStrategy()
        self.assertRaises(ValueError, gs.add_transform, "not a transform class")

    def test_add_duplicate_transform_id(self):
        gs = fk.GatingStrategy()
        gs.add_transform(logicle_xform1)

        self.assertRaises(KeyError, gs.add_transform, logicle_xform1)

    def test_add_matrix_non_matrix_class(self):
        gs = fk.GatingStrategy()
        self.assertRaises(ValueError, gs.add_comp_matrix, "not a matrix class")

    def test_add_duplicate_matrix_id(self):
        gs = fk.GatingStrategy()
        gs.add_comp_matrix(comp_matrix_01)

        self.assertRaises(KeyError, gs.add_comp_matrix, comp_matrix_01)

    def test_absolute_percent(self):
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

        result = gs.gate_sample(data1_sample)
        parent_gate = gs.get_parent_gate(rect_gate2.id)
        parent_gate_count = result.get_gate_count(parent_gate.id)
        gate_count = result.get_gate_count(rect_gate2.id)
        gate_abs_pct = result.get_gate_absolute_percent(rect_gate2.id)
        gate_rel_pct = result.get_gate_relative_percent(rect_gate2.id)

        true_count = 558
        true_abs_pct = (558 / data1_sample.event_count) * 100
        true_rel_pct = (558 / float(parent_gate_count)) * 100

        self.assertEqual(true_count, gate_count)
        self.assertEqual(true_abs_pct, gate_abs_pct)
        self.assertEqual(true_rel_pct, gate_rel_pct)


class GatingStrategyReusedGatesTestCase(unittest.TestCase):
    def setUp(self):
        """
        This TestCase tests more complex GatingStrategy use cases, particularly
        the re-use of a gate in 2 different branches where the parent of each
        gate is also re-used. For example:

        root
        ╰── Gate_A
            ├── Gate_B
            │   ╰── ReusedParent
            │       ╰── ReusedChild
            ╰── Gate_C
                ╰── ReusedParent
                    ╰── ReusedChild

        :return: None
        """
        self.gs = fk.GatingStrategy()

        time_dim = fk.Dimension('Gate_A', range_min=0.1, range_max=0.9)
        dim_fsc_w = fk.Dimension('FSC-W')
        dim_fsc_h = fk.Dimension('FSC-H')
        dim_ssc_a = fk.Dimension('SSC-A')
        dim_amine_a = fk.Dimension('Aqua Amine FLR-A')
        dim_cd3_a = fk.Dimension('CD3 APC-H7 FLR-A')

        gate_a = fk.gates.RectangleGate('Gate_A', None, [time_dim])
        self.gs.add_gate(gate_a)

        gate_b_vertices = [
            fk.Vertex([0.328125, 0.1640625]),
            fk.Vertex([0.296875, 0.1484375]),
            fk.Vertex([0.30859375, 0.8515625]),
            fk.Vertex([0.34765625, 0.3984375]),
            fk.Vertex([0.3359375, 0.1875])
        ]
        gate_b = fk.gates.PolygonGate(
            'Gate_B', parent_id=gate_a.id, dimensions=[dim_fsc_w, dim_fsc_h], vertices=gate_b_vertices
        )
        self.gs.add_gate(gate_b)

        gate_c_vertices = [
            fk.Vertex([0.328125, 0.1640625]),
            fk.Vertex([0.296875, 0.1484375]),
            fk.Vertex([0.30859375, 0.8515625]),
            fk.Vertex([0.34765625, 0.3984375]),
            fk.Vertex([0.3359375, 0.1875])
        ]
        gate_c = fk.gates.PolygonGate(
            'Gate_C', parent_id=gate_a.id, dimensions=[dim_fsc_h, dim_fsc_w], vertices=gate_c_vertices
        )
        self.gs.add_gate(gate_c)

        reused_parent_vertices = [
            fk.Vertex([0.2629268137285685, 0.0625]),
            fk.Vertex([0.24318837264468562, 0.03515625]),
            fk.Vertex([0.21573453285608676, 0.0390625]),
            fk.Vertex([0.29042797365869377, 0.24609375]),
            fk.Vertex([0.29042797365869377, 0.1484375])
        ]

        reused_parent_gate_1 = fk.gates.PolygonGate(
            'ReusedParent', gate_b.id, [dim_amine_a, dim_ssc_a], reused_parent_vertices
        )
        reused_parent_gate_2 = fk.gates.PolygonGate(
            'ReusedParent', gate_c.id, [dim_amine_a, dim_ssc_a], reused_parent_vertices
        )
        self.gs.add_gate(reused_parent_gate_1)
        self.gs.add_gate(reused_parent_gate_2)

        reused_child_vertices = [
            fk.Vertex([0.28415161867527605, 0.11328125]),
            fk.Vertex([0.3132637699981912, 0.203125]),
            fk.Vertex([0.6896802981119161, 0.05078125]),
            fk.Vertex([0.5692952580886116, 0.01953125]),
            fk.Vertex([0.3192472844795108, 0.01953125])
        ]

        reused_child_gate = fk.gates.PolygonGate(
            'ReusedChild', 'ReusedParent', [dim_cd3_a, dim_ssc_a], reused_child_vertices
        )

        gate_path_1 = ['root', 'Gate_A', 'Gate_B', 'ReusedParent']
        gate_path_2 = ['root', 'Gate_A', 'Gate_C', 'ReusedParent']
        self.gs.add_gate(reused_child_gate, gate_path=gate_path_1)
        self.gs.add_gate(reused_child_gate, gate_path=gate_path_2)

        self.all_gate_ids = [
            ('Gate_A', ['root']),
            ('Gate_B', ['root', 'Gate_A']),
            ('ReusedParent', ['root', 'Gate_A', 'Gate_B']),
            ('ReusedChild', ['root', 'Gate_A', 'Gate_B', 'ReusedParent']),
            ('Gate_C', ['root', 'Gate_A']),
            ('ReusedParent', ['root', 'Gate_A', 'Gate_C']),
            ('ReusedChild', ['root', 'Gate_A', 'Gate_C', 'ReusedParent'])
        ]

    def test_gate_reuse_with_reused_parent(self):
        self.assertListEqual(self.all_gate_ids, self.gs.get_gate_ids())

    def test_get_gate(self):
        # test getting all individual gates
        for gate_item in self.all_gate_ids:
            gate = self.gs.get_gate(gate_item[0], gate_item[1])
            self.assertEqual(gate.id, gate_item[0])

    def test_get_child_gates(self):
        parent_gate_id = 'Gate_A'
        parent_gate_path = ['root']
        child_gate_ids = ['Gate_B', 'Gate_C']
        child_gates = self.gs.get_child_gates(parent_gate_id, parent_gate_path)

        retrieved_gate_ids = []
        for child_gate in child_gates:
            self.assertIsInstance(child_gate, fk.gates._gates.Gate)
            retrieved_gate_ids.append(child_gate.id)

        self.assertListEqual(child_gate_ids, sorted(retrieved_gate_ids))

    def test_get_gate_fails_without_path(self):
        self.assertRaises(ValueError, self.gs.get_gate, 'ReusedParent')
