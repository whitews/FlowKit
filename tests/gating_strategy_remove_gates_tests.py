"""
Test for removing gates from a GatingStrategy
"""
import unittest
import copy
from .session_tests import test_samples_8c_full_set_dict
from .matrix_tests import csv_8c_comp_file_path, detectors_8c
import flowkit as fk


class GatingStrategyRemoveGatesTestCase(unittest.TestCase):
    def setUp(self):
        """
        This TestCase tests removing gates from a GatingStrategy, covering
        the various scenarios of the types of gates and their relationships.
        A custom gate hierarchy was created to test these scenarios, and
        looks like:

        root
        ╰── Time-range
            ╰── Singlets-poly
                ╰── Live-poly
                    ╰── CD3-pos-range
                        ├── CD4-pos-poly
                        ├── CD8-pos-poly
                        ├── CD4-CD8-dbl-pos-bool
                        ├── Q-CD4-CD8
                        │   ├── CD4P-CD8P
                        │   ├── CD4N-CD8P
                        │   │   ╰── CD107a-pos-range
                        │   ├── CD4N-CD8N
                        │   ╰── CD4P-CD8N
                        ╰── CD4-or-CD8-pos-bool
                            ╰── CD107a-pos-range

        :return: None
        """
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'
        sample = copy.deepcopy(test_samples_8c_full_set_dict[sample_id])
        comp = fk.Matrix('spill', csv_8c_comp_file_path, detectors_8c)

        session = fk.Session()
        session.add_samples(sample)
        session.add_comp_matrix(comp)

        # define transforms we'll be using
        # linear transform for our time dimension
        lin_xform = fk.transforms.LinearTransform('time-lin', param_t=69, param_a=0)
        sc_lin_xform = fk.transforms.LinearTransform('scatter-lin', param_t=262144, param_a=0)
        flr_xform = fk.transforms.LogicleTransform(
            'flr-logicle', param_t=262144, param_a=0, param_w=1, param_m=4.418539922
        )

        session.add_transform(lin_xform)
        session.add_transform(sc_lin_xform)
        session.add_transform(flr_xform)

        # time dimension with ranges
        time_dim = fk.Dimension('Time', transformation_ref='time-lin', range_min=0.1, range_max=0.9)

        # scatter dims
        dim_ssc_w = fk.Dimension('SSC-W', transformation_ref="scatter-lin")
        dim_ssc_h = fk.Dimension('SSC-H', transformation_ref="scatter-lin")
        dim_ssc_a = fk.Dimension('SSC-A', transformation_ref="scatter-lin")
        dim_fsc_w = fk.Dimension('FSC-W', transformation_ref="scatter-lin")
        dim_fsc_h = fk.Dimension('FSC-H', transformation_ref="scatter-lin")
        dim_fsc_a = fk.Dimension('FSC-A', transformation_ref="scatter-lin")

        # fluoro_dims
        dim_amine_a = fk.Dimension('Aqua Amine FLR-A', compensation_ref='spill', transformation_ref='flr-logicle')
        dim_cd3 = fk.Dimension('CD3 APC-H7 FLR-A', compensation_ref='spill', transformation_ref='flr-logicle')
        dim_cd4 = fk.Dimension('CD4 PE-Cy7 FLR-A', compensation_ref='spill', transformation_ref='flr-logicle')
        dim_cd8 = fk.Dimension('CD8 PerCP-Cy55 FLR-A', compensation_ref='spill', transformation_ref='flr-logicle')
        dim_cd107a = fk.Dimension('CD107a PE FLR-A', compensation_ref='spill', transformation_ref='flr-logicle')

        # Start with time gate
        gate_time = fk.gates.RectangleGate('Time-range', [time_dim])
        session.add_gate(gate_time, ('root',))

        gate_singlets_poly_vertices = [
            [0.328125, 0.2],
            [0.28, 0.25],
            [0.29, 0.83],
            [0.34765625, 0.3984375],
            [0.3359375, 0.2]
        ]
        gate_singlets_poly = fk.gates.PolygonGate(
            'Singlets-poly',
            dimensions=[dim_fsc_w, dim_fsc_h],
            vertices=gate_singlets_poly_vertices
        )
        session.add_gate(gate_singlets_poly, ('root', 'Time-range'))

        gate_live_poly_vertices = [
            [0.2629268137285685, 0.0625],
            [0.24318837264468562, 0.03515625],
            [0.21573453285608676, 0.0390625],
            [0.202, 0.216],
            [0.228, 0.288],
            [0.280, 0.319],
            [0.29042797365869377, 0.24609375],
            [0.29042797365869377, 0.1484375],
        ]
        gate_live_poly = fk.gates.PolygonGate(
            'Live-poly', dimensions=[dim_amine_a, dim_ssc_a], vertices=gate_live_poly_vertices
        )
        session.add_gate(gate_live_poly, ('root', 'Time-range', 'Singlets-poly'))

        dim_cd3_pos = copy.deepcopy(dim_cd3)
        dim_cd3_pos.min = 0.282
        dim_cd3_pos.max = None

        gate_cd3_pos_range = fk.gates.RectangleGate('CD3-pos-range', dimensions=[dim_cd3_pos])
        session.add_gate(
            gate_cd3_pos_range, ('root', 'Time-range', 'Singlets-poly', 'Live-poly')
        )

        gate_path_cd3_pos = ('root', 'Time-range', 'Singlets-poly', 'Live-poly', 'CD3-pos-range')

        gate_cd4_pos_vertices = [[0.25, 0.38], [0.65, 0.5], [0.65, 0.8], [0.25, 0.8]]

        gate_cd4_pos = fk.gates.PolygonGate(
            'CD4-pos-poly',
            dimensions=[dim_cd3, dim_cd4],
            vertices=gate_cd4_pos_vertices
        )
        session.add_gate(gate_cd4_pos, gate_path_cd3_pos)

        gate_cd8_pos_vertices = [[0.2, 0.38], [0.7, 0.38], [0.7, 0.9], [0.2, 0.9]]

        gate_cd8_pos = fk.gates.PolygonGate(
            'CD8-pos-poly',
            dimensions=[dim_cd3, dim_cd8],
            vertices=gate_cd8_pos_vertices
        )
        session.add_gate(gate_cd8_pos, gate_path_cd3_pos)

        cd4_pos_gate_paths = session.find_matching_gate_paths(gate_cd4_pos.gate_name)
        cd8_pos_gate_paths = session.find_matching_gate_paths(gate_cd8_pos.gate_name)

        gate_cd4_cd8_dbl_pos_refs = [
            {
                'ref': gate_cd4_pos.gate_name,
                'path': cd4_pos_gate_paths[0],
                'complement': False
            },
            {
                'ref': gate_cd8_pos.gate_name,
                'path': cd8_pos_gate_paths[0],
                'complement': False
            }
        ]

        gate_cd4_cd8_dbl_pos = fk.gates.BooleanGate(
            'CD4-CD8-dbl-pos-bool',
            'and',
            gate_cd4_cd8_dbl_pos_refs
        )
        session.add_gate(gate_cd4_cd8_dbl_pos, gate_path_cd3_pos)

        quad1_div1 = fk.QuadrantDivider('div-cd4', dim_cd4.id, 'spill', [0.4], transformation_ref='flr-logicle')
        quad1_div2 = fk.QuadrantDivider('div-cd8', dim_cd8.id, 'spill', [0.4], transformation_ref='flr-logicle')
        quad1_divs = [quad1_div1, quad1_div2]

        quad_1 = fk.gates.Quadrant(
            quadrant_id='CD4P-CD8P',
            divider_refs=['div-cd4', 'div-cd8'],
            divider_ranges=[(0.4, None), (0.4, None)]
        )
        quad_2 = fk.gates.Quadrant(
            quadrant_id='CD4N-CD8P',
            divider_refs=['div-cd4', 'div-cd8'],
            divider_ranges=[(None, 0.4), (0.4, None)]
        )
        quad_3 = fk.gates.Quadrant(
            quadrant_id='CD4N-CD8N',
            divider_refs=['div-cd4', 'div-cd8'],
            divider_ranges=[(None, 0.4), (None, 0.4)]
        )
        quad_4 = fk.gates.Quadrant(
            quadrant_id='CD4P-CD8N',
            divider_refs=['div-cd4', 'div-cd8'],
            divider_ranges=[(0.4, None), (None, 0.4)]
        )
        quadrants_q1 = [quad_1, quad_2, quad_3, quad_4]

        quad1_gate = fk.gates.QuadrantGate('Q-CD4-CD8', quad1_divs, quadrants_q1)
        session.add_gate(quad1_gate, gate_path_cd3_pos)

        # the next bool gate will be CD4+ OR CD8+ from the quadrants
        cd4_pos_q_gate_paths = session.find_matching_gate_paths('CD4P-CD8N')
        cd8_pos_q_gate_paths = session.find_matching_gate_paths('CD4N-CD8P')

        gate_cd4_or_cd8_pos_refs = [
            {
                'ref': 'CD4P-CD8N',
                'path': cd4_pos_q_gate_paths[0],
                'complement': False
            },
            {
                'ref': 'CD4N-CD8P',
                'path': cd8_pos_q_gate_paths[0],
                'complement': False
            }
        ]

        gate_cd4_or_cd8_pos = fk.gates.BooleanGate(
            'CD4-or-CD8-pos-bool',
            'or',
            gate_cd4_or_cd8_pos_refs
        )
        session.add_gate(gate_cd4_or_cd8_pos, gate_path_cd3_pos)
        gate_path_cd4_or_cd8_pos = tuple(list(gate_path_cd3_pos) + [gate_cd4_or_cd8_pos.gate_name])

        dim_cd107a_pos = copy.deepcopy(dim_cd107a)
        dim_cd107a_pos.min = 0.4
        dim_cd107a_pos.max = None

        dim_cd107a_pos_range = fk.gates.RectangleGate(
            'CD107a-pos-range',
            dimensions=[dim_cd107a_pos]
        )
        # this goes under the CD4N-CD8P Quadrant
        cd8_pos_q_gate_path_full = list(cd8_pos_q_gate_paths[0])
        cd8_pos_q_gate_path_full.append('CD4N-CD8P')
        cd8_pos_q_gate_path_full = tuple(cd8_pos_q_gate_path_full)
        session.add_gate(dim_cd107a_pos_range, cd8_pos_q_gate_path_full)

        dim_cd107a_pos_range2 = fk.gates.RectangleGate(
            'CD107a-pos-range',
            dimensions=[dim_cd107a_pos]
        )
        session.add_gate(dim_cd107a_pos_range2, gate_path_cd4_or_cd8_pos)

        self.gating_strategy = session.gating_strategy

    def test_remove_quadrant_fails(self):
        gs = copy.deepcopy(self.gating_strategy)
        gate_name_to_remove = 'CD4P-CD8N'

        self.assertRaises(fk.exceptions.QuadrantReferenceError, gs.remove_gate, gate_name_to_remove)

    def test_remove_gate_with_bool_dep_fails(self):
        gs = copy.deepcopy(self.gating_strategy)
        gate_name_to_remove = 'Q-CD4-CD8'

        self.assertRaises(fk.exceptions.GateTreeError, gs.remove_gate, gate_name_to_remove, keep_children=True)

    def test_remove_bool_dep(self):
        gs = copy.deepcopy(self.gating_strategy)
        gate_name_to_remove = 'CD4-or-CD8-pos-bool'

        gs.remove_gate(gate_name_to_remove)

        self.assertRaises(fk.exceptions.GateReferenceError, gs.get_gate, gate_name_to_remove)

    def test_remove_gate_keep_children(self):
        # reminder of gate tree relevant to test:
        # root
        # ╰── Time-range
        #     ╰── Singlets-poly
        #         ╰── Live-poly
        #             ╰── CD3-pos-range

        gate_name_to_remove = 'Live-poly'

        parent_gate_name = 'Singlets-poly'

        # new child gate IDs will omit 'Live-poly'
        ground_truth_new_child_gate_ids = [
            (
                'CD3-pos-range',
                ('root', 'Time-range', 'Singlets-poly')
            )
        ]

        gs = copy.deepcopy(self.gating_strategy)
        gs.remove_gate(gate_name_to_remove, keep_children=True)
        new_child_gate_ids = gs.get_child_gate_ids(parent_gate_name)

        self.assertEqual(new_child_gate_ids, ground_truth_new_child_gate_ids)
