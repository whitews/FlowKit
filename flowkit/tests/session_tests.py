"""
Session Tests
"""
import copy
import unittest
import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('../..'))

from flowkit import Session, Sample, Matrix, Dimension, gates, transforms, load_samples
# noinspection PyProtectedMember
from flowkit._models.transforms._base_transform import Transform
# noinspection PyProtectedMember
from flowkit._models.gates._base_gate import Gate
from flowkit.exceptions import GateReferenceError
from .gating_strategy_prog_gate_tests import data1_sample, poly1_gate, poly1_vertices, comp_matrix_01, asinh_xform1

fcs_file_paths = [
    "data/100715.fcs",
    "data/109567.fcs",
    "data/113548.fcs"
]
test_samples_base_set = load_samples(fcs_file_paths)
test_samples_8c_full_set = load_samples("data/8_color_data_set/fcs_files")
test_samples_8c_full_set_dict = {s.original_filename: s for s in test_samples_8c_full_set}


class SessionTestCase(unittest.TestCase):
    """Tests for Session class"""
    def test_load_samples_from_list_of_paths(self):
        fks = Session(fcs_samples=fcs_file_paths)

        self.assertEqual(len(fks.sample_lut.keys()), 3)
        self.assertIsInstance(fks.get_sample('100715.fcs'), Sample)

        sample_ids = ["100715.fcs", "109567.fcs", "113548.fcs"]
        self.assertListEqual(fks.get_sample_ids(), sample_ids)

    def test_load_samples_from_list_of_samples(self):
        samples = copy.deepcopy(test_samples_base_set)
        fks = Session(fcs_samples=samples)

        self.assertEqual(len(fks.sample_lut.keys()), 3)
        self.assertIsInstance(fks.get_sample('100715.fcs'), Sample)

    def test_get_comp_matrix(self):
        fks = Session()
        sample_group = 'default'
        fks.add_sample_group(sample_group)
        fks.add_samples(data1_sample, sample_group)
        fks.add_comp_matrix(comp_matrix_01, sample_group)
        comp_mat = fks.get_comp_matrix(sample_group, 'B07', 'MySpill')

        self.assertIsInstance(comp_mat, Matrix)

    def test_get_sample_comp_matrices(self):
        wsp_path = "data/8_color_data_set/8_color_ICS_simple.wsp"
        sample_grp = 'DEN'
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'

        fks = Session(copy.deepcopy(test_samples_8c_full_set))
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=True)

        comp_matrices = fks.get_sample_comp_matrices(sample_grp, sample_id)

        self.assertEqual(len(comp_matrices), 1)
        for cm in comp_matrices:
            self.assertIsInstance(cm, Matrix)

    def test_get_group_comp_matrices(self):
        wsp_path = "data/8_color_data_set/8_color_ICS_simple.wsp"
        sample_grp = 'DEN'

        fks = Session(copy.deepcopy(test_samples_8c_full_set))
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=True)

        comp_matrices = fks.get_group_comp_matrices(sample_grp)

        self.assertEqual(len(comp_matrices), 1)
        for cm in comp_matrices:
            self.assertIsInstance(cm, Matrix)

    def test_get_transform(self):
        fks = Session()
        sample_group = 'default'
        fks.add_sample_group(sample_group)
        fks.add_samples(data1_sample, sample_group)
        fks.add_transform(asinh_xform1, sample_group)
        comp_mat = fks.get_transform(sample_group, 'AsinH_10000_4_1')

        self.assertIsInstance(comp_mat, transforms.AsinhTransform)

    def test_get_group_transforms(self):
        wsp_path = "data/8_color_data_set/8_color_ICS_simple.wsp"
        sample_grp = 'DEN'

        fks = Session(copy.deepcopy(test_samples_8c_full_set))
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=True)

        xforms = fks.get_group_transforms(sample_grp)

        self.assertEqual(len(xforms), 23)
        for cm in xforms:
            self.assertIsInstance(cm, Transform)

    def test_get_sample_transforms(self):
        wsp_path = "data/8_color_data_set/8_color_ICS_simple.wsp"
        sample_grp = 'DEN'
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'

        fks = Session(copy.deepcopy(test_samples_8c_full_set))
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=True)

        xforms = fks.get_sample_transforms(sample_grp, sample_id)

        self.assertEqual(len(xforms), 23)
        for cm in xforms:
            self.assertIsInstance(cm, Transform)

    def test_get_sample_gates(self):
        wsp_path = "data/8_color_data_set/8_color_ICS_simple.wsp"
        sample_grp = 'DEN'
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'

        fks = Session(copy.deepcopy(test_samples_8c_full_set))
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=True)

        sample_gates = fks.get_sample_gates(sample_grp, sample_id)

        self.assertEqual(len(sample_gates), 4)
        for cm in sample_gates:
            self.assertIsInstance(cm, Gate)

    def test_get_sample_gate_events(self):
        wsp_path = "data/8_color_data_set/8_color_ICS_simple.wsp"
        sample_grp = 'DEN'
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'
        gate_name = 'CD3+'

        fks = Session(copy.deepcopy(test_samples_8c_full_set))
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=True)

        fks.analyze_samples(sample_grp, sample_id)

        sample_comp = fks.get_sample_comp_matrices(sample_grp, sample_id)[0]
        sample_xform = transforms.LogicleTransform(
            'my_logicle',
            param_t=262144.0,
            param_w=1.0,
            param_m=4.418539922,
            param_a=0.0
        )

        df_gated_events = fks.get_gate_events(
            sample_grp,
            sample_id,
            gate_name,
            matrix=sample_comp,
            transform=sample_xform
        )

        self.assertIsInstance(df_gated_events, pd.DataFrame)
        self.assertEqual(len(df_gated_events), 133670)

    def test_get_wsp_gated_events(self):
        wsp_path = "data/8_color_data_set/8_color_ICS_simple.wsp"
        sample_grp = 'DEN'
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'
        gate_name = 'CD3+'

        fks = Session(copy.deepcopy(test_samples_8c_full_set))
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=True)

        fks.analyze_samples(sample_grp, sample_id)

        df_gated_events = fks.get_wsp_gated_events(
            sample_grp,
            [sample_id],
            gate_name
        )

        self.assertIsInstance(df_gated_events, list)
        self.assertEqual(len(df_gated_events[0]), 133670)

    def test_get_child_gate_ids(self):
        wsp_path = "data/8_color_data_set/8_color_ICS.wsp"
        sample_grp = 'DEN'
        gate_name = 'CD3+'

        fks = Session()
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=True)

        child_gate_ids = fks.get_child_gate_ids(sample_grp, gate_name)

        truth = [
            ('CD4+', ('root', 'Time', 'Singlets', 'aAmine-', 'CD3+')),
            ('CD8+', ('root', 'Time', 'Singlets', 'aAmine-', 'CD3+'))
        ]

        self.assertEqual(len(child_gate_ids), 2)
        for gate_name, gate_path in child_gate_ids:
            self.assertIn((gate_name, gate_path), truth)

    def test_ambiguous_gate_raises_in_get_child_gate_ids(self):
        wsp_path = "data/8_color_data_set/8_color_ICS.wsp"
        sample_grp = 'DEN'
        gate_name = 'IFNg+'

        fks = Session()
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=True)

        self.assertRaises(GateReferenceError, fks.get_child_gate_ids, sample_grp, gate_name)

    def test_find_matching_gate_paths(self):
        wsp_path = "data/8_color_data_set/8_color_ICS.wsp"
        sample_grp = 'DEN'
        gate_name = 'IFNg+'

        fks = Session()
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=True)

        gate_paths = fks.find_matching_gate_paths(sample_grp, gate_name)

        truth = [
            ('root', 'Time', 'Singlets', 'aAmine-', 'CD3+', 'CD4+'),
            ('root', 'Time', 'Singlets', 'aAmine-', 'CD3+', 'CD8+')
        ]

        self.assertEqual(len(gate_paths), 2)
        for gate_path in gate_paths:
            self.assertIn(gate_path, truth)

    @staticmethod
    def test_add_poly1_gate():
        fks = Session()
        sample_group = 'default'
        fks.add_sample_group(sample_group)
        fks.add_samples(data1_sample, sample_group)
        fks.add_gate(poly1_gate, ('root',), sample_group)
        fks.analyze_samples(sample_group)
        result = fks.get_gating_results(sample_group, data1_sample.original_filename)

        res_path = 'data/gate_ref/truth/Results_Polygon1.txt'
        truth = pd.read_csv(res_path, header=None, dtype='bool').squeeze().values

        np.testing.assert_array_equal(truth, result.get_gate_membership('Polygon1'))

    def test_get_gate_from_template(self):
        fks = Session()
        sample_group = 'default'
        fks.add_sample_group(sample_group)
        fks.add_samples(data1_sample, sample_group)
        fks.add_gate(poly1_gate, ('root',), sample_group)

        template_gate = fks.get_gate(sample_group, 'Polygon1')

        self.assertEqual(template_gate.gate_name, 'Polygon1')

    @staticmethod
    def test_add_matrix_poly4_gate():
        fks = Session()
        sample_group = 'default'
        fks.add_sample_group(sample_group)
        fks.add_samples(data1_sample, sample_group)

        fks.add_comp_matrix(comp_matrix_01, sample_group)

        dim1 = Dimension('PE', compensation_ref='MySpill')
        dim2 = Dimension('PerCP', compensation_ref='MySpill')
        dims = [dim1, dim2]

        poly_gate = gates.PolygonGate('Polygon4', dims, poly1_vertices)
        fks.add_gate(poly_gate, ('root',), sample_group)

        res_path = 'data/gate_ref/truth/Results_Polygon4.txt'
        truth = pd.read_csv(res_path, header=None, dtype='bool').squeeze().values

        fks.analyze_samples(sample_group)
        result = fks.get_gating_results(sample_group, data1_sample.original_filename)

        np.testing.assert_array_equal(truth, result.get_gate_membership('Polygon4'))

    @staticmethod
    def test_add_transform_asinh_range1_gate():
        fks = Session()
        sample_group = 'default'
        fks.add_sample_group(sample_group)
        fks.add_samples(data1_sample, sample_group)
        fks.add_transform(asinh_xform1, sample_group)

        dim1 = Dimension('FL1-H', 'uncompensated', 'AsinH_10000_4_1', range_min=0.37, range_max=0.63)
        dims = [dim1]

        rect_gate = gates.RectangleGate('ScaleRange1', dims)
        fks.add_gate(rect_gate, ('root',), sample_group)

        res_path = 'data/gate_ref/truth/Results_ScaleRange1.txt'
        truth = pd.read_csv(res_path, header=None, dtype='bool').squeeze().values

        fks.analyze_samples(sample_group)
        result = fks.get_gating_results(sample_group, data1_sample.original_filename)

        np.testing.assert_array_equal(truth, result.get_gate_membership('ScaleRange1'))

    def test_add_samples_with_group(self):
        sample_ids = ["100715.fcs", "109567.fcs", "113548.fcs"]

        s = Session()

        group_name = 'gml'
        s.add_sample_group(group_name)

        s.add_samples(copy.deepcopy(test_samples_base_set), group_name)

        s_sample_ids = sorted(s.get_group_sample_ids(group_name))

        self.assertListEqual(sample_ids, s_sample_ids)

    def test_new_group_membership_is_empty(self):
        s = Session()

        group_name = 'new_group'
        s.add_sample_group(group_name)

        s.add_samples(copy.deepcopy(test_samples_base_set))  # load without assigning a group

        s_sample_ids = sorted(s.get_group_sample_ids(group_name))

        self.assertListEqual(s_sample_ids, [])

    def test_session_summary(self):
        wsp_path = "data/8_color_data_set/8_color_ICS.wsp"
        sample_grp = 'DEN'

        fks = Session()
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=True)

        fks_summary = fks.summary()

        self.assertIsInstance(fks_summary, pd.DataFrame)

        group_stats = fks_summary.loc[sample_grp]
        self.assertEqual(group_stats.max_gate_depth, 6)
        self.assertEqual(group_stats.samples, 3)
        self.assertEqual(group_stats.loaded_samples, 0)

    def test_analyze_samples_multiproc(self):
        wsp_path = "data/8_color_data_set/8_color_ICS_simple.wsp"
        sample_grp = 'DEN'
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'
        gate_name = 'CD3+'

        fks = Session(copy.deepcopy(test_samples_8c_full_set))
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=True)

        fks.analyze_samples(sample_grp)

        gate_membership = fks.get_gate_membership(sample_grp, sample_id, gate_name)

        self.assertEqual(gate_membership.sum(), 133670)
