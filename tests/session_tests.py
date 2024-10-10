"""
Session Tests
"""

import copy
import unittest
import numpy as np
import pandas as pd
import warnings

from flowkit import (
    Session,
    Sample,
    Matrix,
    Dimension,
    gates,
    transforms,
    generate_transforms,
)
from flowkit.exceptions import GateTreeError
from tests.test_config import (
    fcs_file_paths,
    test_samples_base_set,
    data1_sample,
    poly1_gate,
    poly1_vertices,
    comp_matrix_01,
    asinh_xform_10000_4_1,
)


class SessionTestCase(unittest.TestCase):
    """Tests for Session class"""

    def test_create_session_raises(self):
        # verify non-valid args raise ValueError
        self.assertRaises(ValueError, Session, {})

    def test_load_samples_from_list_of_paths(self):
        fks = Session(fcs_samples=fcs_file_paths)

        self.assertEqual(len(fks.sample_lut.keys()), 3)
        self.assertIsInstance(fks.get_sample("100715.fcs"), Sample)

        sample_ids = ["100715.fcs", "109567.fcs", "113548.fcs"]
        self.assertListEqual(fks.get_sample_ids(), sample_ids)

    def test_load_samples_from_list_of_samples(self):
        samples = copy.deepcopy(test_samples_base_set)
        fks = Session(fcs_samples=samples)

        self.assertEqual(len(fks.sample_lut.keys()), 3)
        self.assertIsInstance(fks.get_sample("100715.fcs"), Sample)

    def test_add_samples_sample_already_exists(self):
        samples = copy.deepcopy(test_samples_base_set)
        fks = Session(fcs_samples=samples)

        self.assertEqual(len(fks.get_sample_ids()), 3)

        # add sample that was already added above
        # ignore warning about already existing sample
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fks.add_samples(samples[0])

        # verify sample count didn't change
        self.assertEqual(len(fks.get_sample_ids()), 3)

    def test_get_comp_matrix(self):
        fks = Session()
        fks.add_comp_matrix("MySpill", comp_matrix_01)
        comp_mat = fks.get_comp_matrix("MySpill")

        self.assertIsInstance(comp_mat, Matrix)

    def test_get_comp_matrices(self):
        fks = Session()
        fks.add_comp_matrix("MySpill", comp_matrix_01)
        comp_matrix_02 = copy.deepcopy(comp_matrix_01)
        fks.add_comp_matrix("MySpill2", comp_matrix_02)

        matrix_lut = {"MySpill": comp_matrix_01, "MySpill2": comp_matrix_02}

        session_matrix_lut = fks.get_comp_matrices()

        self.assertEqual(matrix_lut, session_matrix_lut)

    def test_get_transform(self):
        fks = Session()
        fks.add_transform("AsinH_10000_4_1", asinh_xform_10000_4_1)
        xform = fks.get_transform("AsinH_10000_4_1")

        self.assertIsInstance(xform, transforms.AsinhTransform)

    def test_get_transforms(self):
        xform_lut = generate_transforms(data1_sample)

        session = Session()

        for xform_id, xform in xform_lut.items():
            session.add_transform(xform_id, xform)

        session_xform_lut = session.get_transforms()

        self.assertDictEqual(xform_lut, session_xform_lut)

    @staticmethod
    def test_add_poly1_gate():
        fks = Session()
        fks.add_samples(data1_sample)
        fks.add_gate(poly1_gate, ("root",))
        fks.analyze_samples()
        result = fks.get_gating_results(data1_sample.id)

        res_path = "data/gate_ref/truth/Results_Polygon1.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        np.testing.assert_array_equal(truth, result.get_gate_membership("Polygon1"))

    def test_get_gate_from_template(self):
        fks = Session()
        fks.add_samples(data1_sample)
        fks.add_gate(poly1_gate, ("root",))

        template_gate = fks.get_gate("Polygon1")

        self.assertEqual(template_gate.gate_name, "Polygon1")

    @staticmethod
    def test_add_matrix_poly4_gate():
        fks = Session()
        fks.add_samples(data1_sample)

        fks.add_comp_matrix("MySpill", comp_matrix_01)

        dim1 = Dimension("PE", compensation_ref="MySpill")
        dim2 = Dimension("PerCP", compensation_ref="MySpill")
        dims = [dim1, dim2]

        poly_gate = gates.PolygonGate("Polygon4", dims, poly1_vertices)
        fks.add_gate(poly_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_Polygon4.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        fks.analyze_samples()
        result = fks.get_gating_results(data1_sample.id)

        np.testing.assert_array_equal(truth, result.get_gate_membership("Polygon4"))

    @staticmethod
    def test_add_transform_asinh_range1_gate():
        fks = Session()
        fks.add_samples(data1_sample)
        fks.add_transform("AsinH_10000_4_1", asinh_xform_10000_4_1)

        dim1 = Dimension(
            "FL1-H", "uncompensated", "AsinH_10000_4_1", range_min=0.37, range_max=0.63
        )
        dims = [dim1]

        rect_gate = gates.RectangleGate("ScaleRange1", dims)
        fks.add_gate(rect_gate, ("root",))

        res_path = "data/gate_ref/truth/Results_ScaleRange1.txt"
        truth = pd.read_csv(res_path, header=None, dtype="bool").squeeze().values

        fks.analyze_samples()
        result = fks.get_gating_results(data1_sample.id)

        np.testing.assert_array_equal(truth, result.get_gate_membership("ScaleRange1"))

    def test_get_gate_hierarchy(self):
        gml_path = "data/gate_ref/gml/gml_parent_poly1_boolean_and2_gate.xml"
        fcs_path = "data/gate_ref/data1.fcs"

        session = Session(gating_strategy=gml_path, fcs_samples=fcs_path)

        hierarchy_ascii = session.get_gate_hierarchy()

        hierarchy_truth = """root
├── Range1
├── Polygon1
│   ╰── ParAnd2
╰── Ellipse1"""

        self.assertEqual(hierarchy_ascii, hierarchy_truth)

    def test_get_sample_gates(self):
        gml_path = "data/gate_ref/gml/gml_parent_poly1_boolean_and2_gate.xml"
        fcs_path = "data/gate_ref/data1.fcs"
        sample_id = "B07"

        session = Session(gating_strategy=gml_path, fcs_samples=fcs_path)

        sample_gates = session.get_sample_gates(sample_id)
        sample_gate_names = {g.gate_name for g in sample_gates}

        truth_gate_names = {"Range1", "Polygon1", "ParAnd2", "Ellipse1"}

        self.assertSetEqual(sample_gate_names, truth_gate_names)

    def test_get_child_gate_ids(self):
        gml_path = "data/gate_ref/gml/gml_parent_poly1_boolean_and2_gate.xml"
        fcs_path = "data/gate_ref/data1.fcs"
        parent_gate_id = "Polygon1"

        session = Session(gating_strategy=gml_path, fcs_samples=fcs_path)

        child_gate_ids = session.get_child_gate_ids(parent_gate_id)

        truth_gate_ids = [("ParAnd2", ("root", "Polygon1"))]

        self.assertListEqual(child_gate_ids, truth_gate_ids)

    def test_get_analysis_report(self):
        gml_path = "data/gate_ref/gml/gml_parent_poly1_boolean_and2_gate.xml"
        fcs_path = "data/gate_ref/data1.fcs"

        session = Session(gating_strategy=gml_path, fcs_samples=fcs_path)
        session.analyze_samples()

        session_report = session.get_analysis_report()

        self.assertIsInstance(session_report, pd.DataFrame)
        self.assertEqual(session_report.shape, (4, 10))

    def test_get_gating_results_raises(self):
        gml_path = "data/gate_ref/gml/gml_parent_poly1_boolean_and2_gate.xml"
        fcs_path = "data/gate_ref/data1.fcs"
        sample_id = "B07"

        session = Session(gating_strategy=gml_path, fcs_samples=fcs_path)

        # purposely try to get results prior to calling analyze_samples()
        self.assertRaises(KeyError, session.get_gating_results, sample_id)

    def test_get_gate_events(self):
        gml_path = "data/gate_ref/gml/gml_parent_poly1_boolean_and2_gate.xml"
        fcs_path = "data/gate_ref/data1.fcs"
        sample_id = "B07"

        session = Session(gating_strategy=gml_path, fcs_samples=fcs_path)
        session.analyze_samples()

        gate_events = session.get_gate_events(sample_id, gate_name="ParAnd2")

        self.assertIsInstance(gate_events, pd.DataFrame)
        self.assertEqual(len(gate_events), 12)

    def test_rename_gate(self):
        gml_path = "data/gate_ref/gml/gml_parent_poly1_boolean_and2_gate.xml"
        fcs_path = "data/gate_ref/data1.fcs"
        sample_id = "B07"

        session = Session(gating_strategy=gml_path, fcs_samples=fcs_path)

        # rename Boolean gate 'ParAnd2'
        gate_to_rename = 'ParAnd2'
        new_gate_name = 'BoolAnd2'

        session.rename_gate(gate_to_rename, new_gate_name)

        # verify they are now absent from the gate tree
        sample_gates = session.get_sample_gates(sample_id)
        sample_gate_names = {g.gate_name for g in sample_gates}

        truth_gate_names = {"Range1", "Polygon1", "BoolAnd2", "Ellipse1"}

        self.assertSetEqual(sample_gate_names, truth_gate_names)

    def test_remove_gate(self):
        gml_path = "data/gate_ref/gml/gml_parent_poly1_boolean_and2_gate.xml"
        fcs_path = "data/gate_ref/data1.fcs"
        sample_id = "B07"

        session = Session(gating_strategy=gml_path, fcs_samples=fcs_path)

        # try to remove 'Polygon1' gate
        # will fail b/c of Boolean gate 'ParAnd2' references it
        gate_to_remove = 'Polygon1'
        self.assertRaises(GateTreeError, session.remove_gate, gate_to_remove)

        # so remove Boolean gate 'ParAnd2' first
        session.remove_gate('ParAnd2')

        # and then remove 'Polygon1'
        session.remove_gate(gate_to_remove)

        # verify they are now absent from the gate tree
        sample_gates = session.get_sample_gates(sample_id)
        sample_gate_names = {g.gate_name for g in sample_gates}

        truth_gate_names = {"Range1", "Ellipse1"}

        self.assertSetEqual(sample_gate_names, truth_gate_names)