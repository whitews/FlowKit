"""
Session Tests
"""
import unittest
import sys
import os
from io import BytesIO
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('../..'))

from flowkit import Session, Sample, Matrix, Dimension, gates, transforms
# noinspection PyProtectedMember
from flowkit._models.transforms._base_transform import Transform
# noinspection PyProtectedMember
from flowkit._models.gates._base_gate import Gate
from .gating_strategy_prog_gate_tests import data1_sample, poly1_gate, poly1_vertices, comp_matrix_01, asinh_xform1

fcs_file_paths = [
    "examples/100715.fcs",
    "examples/109567.fcs",
    "examples/113548.fcs"
]


class SessionTestCase(unittest.TestCase):
    """Tests for Session class"""
    def test_load_samples_from_list_of_paths(self):
        fks = Session(fcs_samples=fcs_file_paths)

        self.assertEqual(len(fks.sample_lut.keys()), 3)
        self.assertIsInstance(fks.get_sample('100715.fcs'), Sample)

        sample_ids = ["100715.fcs", "109567.fcs", "113548.fcs"]
        self.assertListEqual(fks.get_sample_ids(), sample_ids)

    def test_load_samples_from_list_of_samples(self):
        samples = [Sample(file_path) for file_path in fcs_file_paths]
        fks = Session(fcs_samples=samples)

        self.assertEqual(len(fks.sample_lut.keys()), 3)
        self.assertIsInstance(fks.get_sample('100715.fcs'), Sample)

    def test_get_comp_matrix(self):
        fks = Session(fcs_samples=data1_sample)
        fks.add_comp_matrix(comp_matrix_01)
        comp_mat = fks.get_comp_matrix('default', 'B07', 'MySpill')

        self.assertIsInstance(comp_mat, Matrix)

    def test_get_sample_comp_matrices(self):
        wsp_path = "examples/8_color_data_set/8_color_ICS_simple.wsp"
        fcs_path = "examples/8_color_data_set/fcs_files"
        sample_grp = 'DEN'
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'

        fks = Session(fcs_samples=fcs_path)
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=True)

        comp_matrices = fks.get_sample_comp_matrices(sample_grp, sample_id)

        self.assertEqual(len(comp_matrices), 1)
        for cm in comp_matrices:
            self.assertIsInstance(cm, Matrix)

    def test_get_group_comp_matrices(self):
        wsp_path = "examples/8_color_data_set/8_color_ICS_simple.wsp"
        fcs_path = "examples/8_color_data_set/fcs_files"
        sample_grp = 'DEN'

        fks = Session(fcs_samples=fcs_path)
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=True)

        comp_matrices = fks.get_group_comp_matrices(sample_grp)

        self.assertEqual(len(comp_matrices), 3)
        for cm in comp_matrices:
            self.assertIsInstance(cm, Matrix)

    def test_get_transform(self):
        fks = Session(fcs_samples=data1_sample)
        fks.add_transform(asinh_xform1)
        comp_mat = fks.get_transform('default', 'AsinH_10000_4_1')

        self.assertIsInstance(comp_mat, transforms.AsinhTransform)

    def test_get_group_transforms(self):
        wsp_path = "examples/8_color_data_set/8_color_ICS_simple.wsp"
        fcs_path = "examples/8_color_data_set/fcs_files"
        sample_grp = 'DEN'

        fks = Session(fcs_samples=fcs_path)
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=True)

        xforms = fks.get_group_transforms(sample_grp)

        self.assertEqual(len(xforms), 23)
        for cm in xforms:
            self.assertIsInstance(cm, Transform)

    def test_get_sample_transforms(self):
        wsp_path = "examples/8_color_data_set/8_color_ICS_simple.wsp"
        fcs_path = "examples/8_color_data_set/fcs_files"
        sample_grp = 'DEN'
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'

        fks = Session(fcs_samples=fcs_path)
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=True)

        xforms = fks.get_sample_transforms(sample_grp, sample_id)

        self.assertEqual(len(xforms), 23)
        for cm in xforms:
            self.assertIsInstance(cm, Transform)

    def test_get_sample_gates(self):
        wsp_path = "examples/8_color_data_set/8_color_ICS_simple.wsp"
        fcs_path = "examples/8_color_data_set/fcs_files"
        sample_grp = 'DEN'
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'

        fks = Session(fcs_samples=fcs_path)
        fks.import_flowjo_workspace(wsp_path, ignore_missing_files=True)

        sample_gates = fks.get_sample_gates(sample_grp, sample_id)

        self.assertEqual(len(sample_gates), 4)
        for cm in sample_gates:
            self.assertIsInstance(cm, Gate)

    def test_get_sample_gate_events(self):
        wsp_path = "examples/8_color_data_set/8_color_ICS_simple.wsp"
        fcs_path = "examples/8_color_data_set/fcs_files"
        sample_grp = 'DEN'
        sample_id = '101_DEN084Y5_15_E01_008_clean.fcs'
        gate_id = 'CD3+'

        fks = Session(fcs_samples=fcs_path)
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
            gate_id,
            matrix=sample_comp,
            transform=sample_xform
        )

        self.assertIsInstance(df_gated_events, pd.DataFrame)
        self.assertEqual(len(df_gated_events), 133670)

    @staticmethod
    def test_add_poly1_gate():
        fks = Session(fcs_samples=data1_sample)
        fks.add_gate(poly1_gate)
        fks.analyze_samples()
        result = fks.get_gating_results('default', data1_sample.original_filename)

        res_path = 'examples/gate_ref/truth/Results_Polygon1.txt'
        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        np.testing.assert_array_equal(truth, result.get_gate_indices('Polygon1'))

    @staticmethod
    def test_add_matrix_poly4_gate():
        fks = Session(fcs_samples=data1_sample)

        fks.add_comp_matrix(comp_matrix_01)

        dim1 = Dimension('PE', compensation_ref='MySpill')
        dim2 = Dimension('PerCP', compensation_ref='MySpill')
        dims = [dim1, dim2]

        poly_gate = gates.PolygonGate('Polygon4', None, dims, poly1_vertices)
        fks.add_gate(poly_gate)

        res_path = 'examples/gate_ref/truth/Results_Polygon4.txt'
        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        fks.analyze_samples()
        result = fks.get_gating_results('default', data1_sample.original_filename)

        np.testing.assert_array_equal(truth, result.get_gate_indices('Polygon4'))

    @staticmethod
    def test_add_transform_asinh_range1_gate():
        fks = Session(fcs_samples=data1_sample)
        fks.add_transform(asinh_xform1)

        dim1 = Dimension('FL1-H', 'uncompensated', 'AsinH_10000_4_1', range_min=0.37, range_max=0.63)
        dims = [dim1]

        rect_gate = gates.RectangleGate('ScaleRange1', None, dims)
        fks.add_gate(rect_gate)

        res_path = 'examples/gate_ref/truth/Results_ScaleRange1.txt'
        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        fks.analyze_samples()
        result = fks.get_gating_results('default', data1_sample.original_filename)

        np.testing.assert_array_equal(truth, result.get_gate_indices('ScaleRange1'))

    def test_wsp_export_simple_poly50(self):
        wsp_path = "examples/simple_line_example/simple_poly_and_rect_v2_poly50.wsp"
        fcs_path = "examples/simple_line_example/data_set_simple_line_100.fcs"
        sample_group = 'my_group'
        sample_id = 'data_set_simple_line_100.fcs'

        fks = Session(fcs_path)
        fks.import_flowjo_workspace(wsp_path)

        with BytesIO() as fh_out:
            fks.export_wsp(
                fh_out,
                sample_group
            )
            fh_out.seek(0)

            fks2 = Session(fcs_path)
            fks2.import_flowjo_workspace(fh_out)

        fks.analyze_samples(sample_group)
        fks_results = fks.get_gating_results(sample_group, sample_id)

        fks2.analyze_samples(sample_group)
        fks2_results = fks2.get_gating_results(sample_group, sample_id)

        gate_refs = fks.get_gate_ids(sample_group)

        self.assertEqual(len(gate_refs), 2)

        fks_rect1_count = fks_results.get_gate_count('rect1')
        fks2_rect1_count = fks2_results.get_gate_count('rect1')
        fks_poly1_count = fks_results.get_gate_count('poly1')
        fks2_poly1_count = fks2_results.get_gate_count('poly1')

        self.assertEqual(fks_rect1_count, 0)
        self.assertEqual(fks2_rect1_count, 0)
        self.assertEqual(fks_poly1_count, 50)
        self.assertEqual(fks2_poly1_count, 50)
