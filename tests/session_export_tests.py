"""
Session Export Tests
"""
import copy
import unittest
from io import BytesIO
from flowkit import Session, Workspace, gates
from tests.test_config import test_samples_8c_full_set, quad1_gate, data1_sample


class SessionExportTestCase(unittest.TestCase):
    def test_wsp_export_simple_poly50(self):
        wsp_path = "data/simple_line_example/simple_poly_and_rect_v2_poly50.wsp"
        fcs_path = "data/simple_line_example/data_set_simple_line_100.fcs"
        sample_group = 'my_group'
        sample_id = 'data_set_simple_line_100.fcs'

        wsp = Workspace(wsp_path, fcs_samples=fcs_path)
        gs = wsp.get_gating_strategy(sample_id)
        session = Session(gating_strategy=gs, fcs_samples=fcs_path)

        with BytesIO() as fh_out:
            session.export_wsp(
                fh_out,
                sample_group
            )
            fh_out.seek(0)

            wsp2 = Workspace(fh_out, fcs_samples=fcs_path)

        wsp.analyze_samples(sample_id=sample_id)
        wsp_results = wsp.get_gating_results(sample_id)

        wsp2.analyze_samples(sample_id=sample_id)
        wsp2_results = wsp2.get_gating_results(sample_id)

        gate_refs = wsp.get_gate_ids(sample_id=sample_id)

        self.assertEqual(len(gate_refs), 2)

        wsp_rect1_count = wsp_results.get_gate_count('rect1')
        wsp2_rect1_count = wsp2_results.get_gate_count('rect1')
        wsp_poly1_count = wsp_results.get_gate_count('poly1')
        wsp2_poly1_count = wsp2_results.get_gate_count('poly1')

        self.assertEqual(wsp_rect1_count, 0)
        self.assertEqual(wsp2_rect1_count, 0)
        self.assertEqual(wsp_poly1_count, 50)
        self.assertEqual(wsp2_poly1_count, 50)

    def test_wsp_export_diamond_biex(self):
        wsp_path = "data/simple_diamond_example/test_data_diamond_biex_rect.wsp"
        fcs_path = "data/simple_diamond_example/test_data_diamond_01.fcs"
        sample_group = 'my_group'
        sample_id = 'test_data_diamond_01.fcs'

        wsp = Workspace(wsp_path, fcs_samples=fcs_path)
        gs = wsp.get_gating_strategy(sample_id)
        session = Session(gating_strategy=gs, fcs_samples=fcs_path)

        with BytesIO() as fh_out:
            session.export_wsp(
                fh_out,
                sample_group
            )
            fh_out.seek(0)

            wsp2 = Workspace(fh_out, fcs_samples=fcs_path)

        wsp.analyze_samples(sample_id=sample_id)
        wsp_results = wsp.get_gating_results(sample_id)

        wsp2.analyze_samples(sample_id=sample_id)
        wsp2_results = wsp2.get_gating_results(sample_id)

        gate_refs = wsp.get_gate_ids(sample_id=sample_id)

        self.assertEqual(len(gate_refs), 1)

        wsp_rect1_count = wsp_results.get_gate_count('upper_right')
        wsp2_rect1_count = wsp2_results.get_gate_count('upper_right')

        self.assertEqual(wsp_rect1_count, 50605)
        self.assertEqual(wsp2_rect1_count, 50605)

    def test_export_wsp(self):
        wsp_path = "data/8_color_data_set/8_color_ICS.wsp"
        sample_grp = 'DEN'

        # use a leaf gate to test if the new WSP session is created correctly
        gate_name = 'TNFa+'
        gate_path = ('root', 'Time', 'Singlets', 'aAmine-', 'CD3+', 'CD4+')

        wsp = Workspace(wsp_path, fcs_samples=copy.deepcopy(test_samples_8c_full_set), ignore_missing_files=True)
        sample_id = '101_DEN084Y5_15_E03_009_clean.fcs'
        gs = wsp.get_gating_strategy(sample_id)

        session = Session(gating_strategy=gs, fcs_samples=copy.deepcopy(test_samples_8c_full_set))

        out_file = BytesIO()
        session.export_wsp(out_file, sample_grp)
        out_file.seek(0)

        wsp_out = Workspace(out_file, fcs_samples=copy.deepcopy(test_samples_8c_full_set), ignore_missing_files=True)

        self.assertIsInstance(wsp_out, Workspace)

        wsp_gate = wsp.get_gate(sample_id, gate_name, gate_path)
        wsp_out_gate = wsp_out.get_gate(sample_id, gate_name, gate_path)

        self.assertIsInstance(wsp_gate, gates.RectangleGate)
        self.assertIsInstance(wsp_out_gate, gates.RectangleGate)

        self.assertEqual(wsp_gate.gate_name, gate_name)
        self.assertEqual(wsp_out_gate.gate_name, gate_name)

    def test_export_quadrant_gate_raises(self):
        session = Session()

        session.add_samples(data1_sample)
        session.add_gate(quad1_gate, ("root",))

        self.assertRaises(NotImplementedError, session.export_wsp, 'tmp.wsp', group_name='All Samples')