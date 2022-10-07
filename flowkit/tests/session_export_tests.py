"""
Session Export Tests
"""
import unittest
from io import BytesIO
from flowkit import Session


class SessionExportTestCase(unittest.TestCase):
    def test_wsp_export_simple_poly50(self):
        wsp_path = "data/simple_line_example/simple_poly_and_rect_v2_poly50.wsp"
        fcs_path = "data/simple_line_example/data_set_simple_line_100.fcs"
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
