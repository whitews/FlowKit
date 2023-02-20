"""
Tests for re-used gates in the GatingStrategy Class
"""
import unittest
import flowkit as fk


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

        time_dim = fk.Dimension('Time', range_min=0.1, range_max=0.9)
        dim_fsc_w = fk.Dimension('FSC-W')
        dim_fsc_h = fk.Dimension('FSC-H')
        dim_ssc_a = fk.Dimension('SSC-A')
        dim_amine_a = fk.Dimension('Aqua Amine FLR-A')
        dim_cd3_a = fk.Dimension('CD3 APC-H7 FLR-A')

        gate_a = fk.gates.RectangleGate('Gate_A', [time_dim])
        self.gs.add_gate(gate_a, ('root',))

        gate_b_vertices = [
            [0.328125, 0.1640625],
            [0.296875, 0.1484375],
            [0.30859375, 0.8515625],
            [0.34765625, 0.3984375],
            [0.3359375, 0.1875]
        ]
        gate_b = fk.gates.PolygonGate(
            'Gate_B', dimensions=[dim_fsc_w, dim_fsc_h], vertices=gate_b_vertices
        )
        self.gs.add_gate(gate_b, ('root', 'Gate_A'))

        gate_c_vertices = [
            [0.328125, 0.1640625],
            [0.296875, 0.1484375],
            [0.30859375, 0.8515625],
            [0.34765625, 0.3984375],
            [0.3359375, 0.1875]
        ]
        gate_c = fk.gates.PolygonGate(
            'Gate_C', dimensions=[dim_fsc_h, dim_fsc_w], vertices=gate_c_vertices
        )
        self.gs.add_gate(gate_c, ('root', 'Gate_A'))

        reused_parent_vertices = [
            [0.2629268137285685, 0.0625],
            [0.24318837264468562, 0.03515625],
            [0.21573453285608676, 0.0390625],
            [0.29042797365869377, 0.24609375],
            [0.29042797365869377, 0.1484375]
        ]

        reused_parent_gate_1 = fk.gates.PolygonGate(
            'ReusedParent', [dim_amine_a, dim_ssc_a], reused_parent_vertices
        )
        reused_parent_gate_2 = fk.gates.PolygonGate(
            'ReusedParent', [dim_amine_a, dim_ssc_a], reused_parent_vertices
        )
        self.gs.add_gate(reused_parent_gate_1, ('root', 'Gate_A', 'Gate_B'))
        self.gs.add_gate(reused_parent_gate_2, ('root', 'Gate_A', 'Gate_C'))

        reused_child_vertices = [
            [0.28415161867527605, 0.11328125],
            [0.3132637699981912, 0.203125],
            [0.6896802981119161, 0.05078125],
            [0.5692952580886116, 0.01953125],
            [0.3192472844795108, 0.01953125]
        ]

        reused_child_gate = fk.gates.PolygonGate(
            'ReusedChild', [dim_cd3_a, dim_ssc_a], reused_child_vertices
        )

        gate_path_1 = ('root', 'Gate_A', 'Gate_B', 'ReusedParent')
        gate_path_2 = ('root', 'Gate_A', 'Gate_C', 'ReusedParent')
        self.gs.add_gate(reused_child_gate, gate_path=gate_path_1)
        self.gs.add_gate(reused_child_gate, gate_path=gate_path_2)

        self.all_gate_ids = [
            ('Gate_A', ('root',)),
            ('Gate_B', ('root', 'Gate_A')),
            ('ReusedParent', ('root', 'Gate_A', 'Gate_B')),
            ('ReusedChild', ('root', 'Gate_A', 'Gate_B', 'ReusedParent')),
            ('Gate_C', ('root', 'Gate_A')),
            ('ReusedParent', ('root', 'Gate_A', 'Gate_C')),
            ('ReusedChild', ('root', 'Gate_A', 'Gate_C', 'ReusedParent'))
        ]

    def test_gate_reuse_with_reused_parent(self):
        self.assertListEqual(self.all_gate_ids, self.gs.get_gate_ids())

    def test_get_gate(self):
        # test getting all individual gates
        for gate_item in self.all_gate_ids:
            gate = self.gs.get_gate(gate_item[0], gate_item[1])
            self.assertEqual(gate.gate_name, gate_item[0])

    def test_get_child_gate_ids(self):
        parent_gate_name = 'Gate_A'
        parent_gate_path = ['root']
        child_gate_names = ['Gate_B', 'Gate_C']
        child_gate_ids = self.gs.get_child_gate_ids(parent_gate_name, parent_gate_path)

        retrieved_gate_names = []
        for gate_name, gate_path in child_gate_ids:
            retrieved_gate_names.append(gate_name)

        self.assertListEqual(child_gate_names, sorted(retrieved_gate_names))

    def test_get_gate_fails_without_path(self):
        self.assertRaises(fk.exceptions.GateReferenceError, self.gs.get_gate, 'ReusedParent')
