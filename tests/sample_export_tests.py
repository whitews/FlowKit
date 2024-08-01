"""
Unit tests for exporting Sample as a CSV or FCS file
"""
import os
import numpy as np
import pandas as pd
import unittest
from tests.test_config import test_comp_sample, data1_fcs_path

from flowkit import Sample


class SampleExportTestCase(unittest.TestCase):
    """Tests for export CSV & FCS files from Sample objects"""
    def test_export_as_fcs(self):
        sample = test_comp_sample

        sample.export("test_fcs_export.fcs", source='comp', directory="data")

        exported_fcs_file = "data/test_fcs_export.fcs"
        exported_sample = Sample(fcs_path_or_data=exported_fcs_file)
        os.unlink(exported_fcs_file)

        self.assertIsInstance(exported_sample, Sample)

        # When the comp events are exported, they are saved as single precision (32-bit). We'll test the
        # arrays with the original sample comp data converted to 32-bit float. The original sample data
        # was also originally in 32-bit but the compensation calculation results in 64-bit data. Comparing
        # both in single precision is then the most "correct" thing to do here.
        np.testing.assert_array_equal(
            sample._comp_events.astype(np.float32),
            exported_sample._raw_events.astype(np.float32)
        )

    def test_export_fcs_as_orig_with_timestep(self):
        # This test uses a file where the preprocessing makes the orig & raw events different.
        # File 100715.fcs has a timestep value of 0.08.
        # The purpose here is to verify that importing the exported file has the same raw events
        # as the original file's raw events. Here we export the 'orig' events.
        fcs_file_path = "data/100715.fcs"

        sample = Sample(fcs_path_or_data=fcs_file_path, cache_original_events=True)

        sample.export("test_fcs_export.fcs", source='orig', directory="data")

        exported_fcs_file = "data/test_fcs_export.fcs"
        exported_sample = Sample(fcs_path_or_data=exported_fcs_file)
        os.unlink(exported_fcs_file)

        self.assertIsInstance(exported_sample, Sample)

        # When the events are exported, they are saved as single precision (32-bit). We'll test the
        # arrays with the original sample data converted to 32-bit float. The original sample data
        # was also originally in 32-bit. Comparing both in single precision is then the most
        # "correct" thing to do here.
        np.testing.assert_array_equal(
            sample._raw_events.astype(np.float32),
            exported_sample._raw_events.astype(np.float32)
        )

    def test_export_fcs_as_orig_with_int_type_fails(self):
        # This test uses a file where with int data type.
        # FlowIO only supports creating FCS files with float data type.
        # A NotImplementedError should be raised when attempting to
        # export this file's original events
        sample = Sample(fcs_path_or_data=data1_fcs_path, cache_original_events=True)

        self.assertRaises(
            NotImplementedError,
            sample.export,
            "test_fcs_export.fcs",
            source='orig',
            directory="data",
            include_metadata=True
        )

    def test_export_fcs_as_raw_with_gain(self):
        # This test uses a file where the preprocessing makes the orig & raw events different.
        # File data1.fcs has 2 channels that specify a gain value other than 1.0.
        # The purpose here is to verify that importing the exported file has the same raw events
        # as the original file's raw events.
        sample = Sample(fcs_path_or_data=data1_fcs_path)

        sample.export("test_fcs_export.fcs", source='raw', directory="data", include_metadata=True)

        exported_fcs_file = "data/test_fcs_export.fcs"
        exported_sample = Sample(fcs_path_or_data=exported_fcs_file)
        os.unlink(exported_fcs_file)

        self.assertIsInstance(exported_sample, Sample)

        # When the events are exported, they are saved as single precision (32-bit). We'll test the
        # arrays with the original sample data converted to 32-bit float. The original sample data
        # was also originally in 32-bit. Comparing both in single precision is then the most
        # "correct" thing to do here.
        np.testing.assert_array_equal(
            sample._raw_events.astype(np.float32),
            exported_sample._raw_events.astype(np.float32)
        )

    def test_export_fcs_as_raw(self):
        # This test uses a file where the preprocessing makes the orig & raw events different.
        # The purpose here is to verify that importing the exported file has the same raw events
        # as the original file's raw events.
        fcs_file_path = "data/8_color_data_set/fcs_files/101_DEN084Y5_15_E01_008_clean.fcs"

        sample = Sample(fcs_path_or_data=fcs_file_path)

        sample.export("test_fcs_export.fcs", source='raw', directory="data")

        exported_fcs_file = "data/test_fcs_export.fcs"
        exported_sample = Sample(fcs_path_or_data=exported_fcs_file)
        os.unlink(exported_fcs_file)

        self.assertIsInstance(exported_sample, Sample)

        # When the events are exported, they are saved as single precision (32-bit). We'll test the
        # arrays with the original sample data converted to 32-bit float. The original sample data
        # was also originally in 32-bit. Comparing both in single precision is then the most
        # "correct" thing to do here.
        np.testing.assert_array_equal(
            sample._raw_events.astype(np.float32),
            exported_sample._raw_events.astype(np.float32)
        )

    def test_export_as_csv(self):
        sample = test_comp_sample

        sample.export("test_fcs_export.csv", source='comp', directory="data")

        exported_csv_file = "data/test_fcs_export.csv"
        exported_df = pd.read_csv(exported_csv_file)
        exported_sample = Sample(exported_df, sample_id='exported_sample')
        os.unlink(exported_csv_file)

        self.assertIsInstance(exported_sample, Sample)

        # When the comp events are exported, they are saved as single precision (32-bit). We'll test the
        # arrays with the original sample comp data converted to 32-bit float. The original sample data
        # was also originally in 32-bit but the compensation calculation results in 64-bit data. Comparing
        # both in single precision is then the most "correct" thing to do here.
        np.testing.assert_array_equal(
            sample._comp_events.astype(np.float32),
            exported_sample._raw_events.astype(np.float32)
        )

    def test_export_exclude_negative_scatter(self):
        # there are 2 negative SSC-A events in this file (of 65016 total events)
        fcs_file_path = "data/100715.fcs"
        sample = Sample(fcs_path_or_data=fcs_file_path)
        sample.filter_negative_scatter()

        neg_scatter_count = len(sample.negative_scatter_indices)

        exported_fcs_file = "data/test_fcs_export.fcs"
        sample.export(exported_fcs_file, source='raw', exclude_neg_scatter=True)
        exported_sample = Sample(exported_fcs_file)
        os.unlink(exported_fcs_file)

        orig_event_count = sample.event_count
        exp_event_count = exported_sample.event_count

        self.assertEqual(exp_event_count, orig_event_count - neg_scatter_count)
