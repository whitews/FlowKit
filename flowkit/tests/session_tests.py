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
from .gating_strategy_prog_gate_tests import data1_sample, poly1_gate, poly1_vertices, comp_matrix_01, asinh_xform1

fcs_file_paths = [
    "data/100715.fcs",
    "data/109567.fcs",
    "data/113548.fcs"
]
test_samples_base_set = load_samples(fcs_file_paths)
test_samples_8c_full_set = load_samples("data/8_color_data_set/fcs_files")
test_samples_8c_full_set_dict = {s.id: s for s in test_samples_8c_full_set}


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
        fks.add_samples(data1_sample)
        fks.add_comp_matrix(comp_matrix_01)
        comp_mat = fks.get_comp_matrix('MySpill')

        self.assertIsInstance(comp_mat, Matrix)

    def test_get_transform(self):
        fks = Session()
        fks.add_transform(asinh_xform1)
        comp_mat = fks.get_transform('AsinH_10000_4_1')

        self.assertIsInstance(comp_mat, transforms.AsinhTransform)

    @staticmethod
    def test_add_poly1_gate():
        fks = Session()
        fks.add_samples(data1_sample)
        fks.add_gate(poly1_gate, ('root',))
        fks.analyze_samples()
        result = fks.get_gating_results(data1_sample.id)

        res_path = 'data/gate_ref/truth/Results_Polygon1.txt'
        truth = pd.read_csv(res_path, header=None, dtype='bool').squeeze().values

        np.testing.assert_array_equal(truth, result.get_gate_membership('Polygon1'))

    def test_get_gate_from_template(self):
        fks = Session()
        fks.add_samples(data1_sample)
        fks.add_gate(poly1_gate, ('root',))

        template_gate = fks.get_gate('Polygon1')

        self.assertEqual(template_gate.gate_name, 'Polygon1')

    @staticmethod
    def test_add_matrix_poly4_gate():
        fks = Session()
        fks.add_samples(data1_sample)

        fks.add_comp_matrix(comp_matrix_01)

        dim1 = Dimension('PE', compensation_ref='MySpill')
        dim2 = Dimension('PerCP', compensation_ref='MySpill')
        dims = [dim1, dim2]

        poly_gate = gates.PolygonGate('Polygon4', dims, poly1_vertices)
        fks.add_gate(poly_gate, ('root',))

        res_path = 'data/gate_ref/truth/Results_Polygon4.txt'
        truth = pd.read_csv(res_path, header=None, dtype='bool').squeeze().values

        fks.analyze_samples()
        result = fks.get_gating_results(data1_sample.id)

        np.testing.assert_array_equal(truth, result.get_gate_membership('Polygon4'))

    @staticmethod
    def test_add_transform_asinh_range1_gate():
        fks = Session()
        fks.add_samples(data1_sample)
        fks.add_transform(asinh_xform1)

        dim1 = Dimension('FL1-H', 'uncompensated', 'AsinH_10000_4_1', range_min=0.37, range_max=0.63)
        dims = [dim1]

        rect_gate = gates.RectangleGate('ScaleRange1', dims)
        fks.add_gate(rect_gate, ('root',))

        res_path = 'data/gate_ref/truth/Results_ScaleRange1.txt'
        truth = pd.read_csv(res_path, header=None, dtype='bool').squeeze().values

        fks.analyze_samples()
        result = fks.get_gating_results(data1_sample.id)

        np.testing.assert_array_equal(truth, result.get_gate_membership('ScaleRange1'))

    def test_get_gate_hierarchy(self):
        gml_path = 'data/gate_ref/gml/gml_parent_poly1_boolean_and2_gate.xml'
        fcs_path = 'data/gate_ref/data1.fcs'

        session = Session(gating_strategy=gml_path, fcs_samples=fcs_path)

        hierarchy_ascii = session.get_gate_hierarchy()

        hierarchy_truth = """root
├── Range1
├── Polygon1
│   ╰── ParAnd2
╰── Ellipse1"""

        self.assertEqual(hierarchy_ascii, hierarchy_truth)

    def test_get_sample_gates(self):
        gml_path = 'data/gate_ref/gml/gml_parent_poly1_boolean_and2_gate.xml'
        fcs_path = 'data/gate_ref/data1.fcs'
        sample_id = 'B07'

        session = Session(gating_strategy=gml_path, fcs_samples=fcs_path)

        sample_gates = session.get_sample_gates(sample_id)
        sample_gate_names = {g.gate_name for g in sample_gates}

        truth_gate_names = {
            'Range1', 'Polygon1', 'ParAnd2', 'Ellipse1'
        }

        self.assertSetEqual(sample_gate_names, truth_gate_names)

    def test_get_child_gate_ids(self):
        gml_path = 'data/gate_ref/gml/gml_parent_poly1_boolean_and2_gate.xml'
        fcs_path = 'data/gate_ref/data1.fcs'
        parent_gate_id = 'Polygon1'

        session = Session(gating_strategy=gml_path, fcs_samples=fcs_path)

        child_gate_ids = session.get_child_gate_ids(parent_gate_id)

        truth_gate_ids = [('ParAnd2', ('root', 'Polygon1'))]

        self.assertListEqual(child_gate_ids, truth_gate_ids)
