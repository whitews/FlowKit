"""
Unit tests for Sample class
"""
import copy
import unittest
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import flowio
import warnings

sys.path.append(os.path.abspath('../..'))

from flowkit import Sample, transforms

data1_fcs_path = 'examples/data/gate_ref/data1.fcs'
data1_sample = Sample(data1_fcs_path)
data1_sample_with_orig = Sample(data1_fcs_path, cache_original_events=True)

xform_logicle = transforms.LogicleTransform('logicle', param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
xform_biex1 = transforms.WSPBiexTransform('neg0', width=-100.0, negative=0.0)
xform_biex2 = transforms.WSPBiexTransform('neg1', width=-100.0, negative=1.0)

fcs_file_path = "examples/data/test_comp_example.fcs"
comp_file_path = "examples/data/comp_complete_example.csv"

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    test_comp_sample = Sample(
        fcs_path_or_data=fcs_file_path,
        compensation=comp_file_path,
        ignore_offset_error=True  # sample has off by 1 data offset
    )

    warnings.simplefilter('ignore')
    test_comp_sample_uncomp = Sample(
        fcs_path_or_data=fcs_file_path,
        ignore_offset_error=True  # sample has off by 1 data offset
    )


class SampleTestCase(unittest.TestCase):
    """Tests for loading FCS files as Sample objects"""
    def test_load_from_fcs_file_path(self):
        """Test creating Sample object from an FCS file path"""
        fcs_file_path = "examples/data/test_data_2d_01.fcs"

        sample = Sample(fcs_path_or_data=fcs_file_path)

        self.assertIsInstance(sample, Sample)

    def test_load_from_flowio_flowdata_object(self):
        """Test creating Sample object from an FCS file path"""
        fcs_file_path = "examples/data/test_data_2d_01.fcs"
        flow_data = flowio.FlowData(fcs_file_path)

        self.assertIsInstance(flow_data, flowio.FlowData)

        sample = Sample(flow_data)

        self.assertIsInstance(sample, Sample)

    def test_load_from_pathlib(self):
        """Test creating Sample object from a pathlib Path object"""
        fcs_file_path = "examples/data/test_data_2d_01.fcs"
        path = Path(fcs_file_path)
        sample = Sample(fcs_path_or_data=path)

        self.assertIsInstance(sample, Sample)

    def test_load_from_io_base(self):
        """Test creating Sample object from a IOBase object"""
        fcs_file_path = "examples/data/test_data_2d_01.fcs"
        f = open(fcs_file_path, 'rb')
        sample = Sample(fcs_path_or_data=f)
        f.close()

        self.assertIsInstance(sample, Sample)

    def test_load_from_numpy_array(self):
        npy_file_path = "examples/data/test_comp_example.npy"
        channels = [
            'FSC-A', 'FSC-W', 'SSC-A',
            'Ax488-A', 'PE-A', 'PE-TR-A',
            'PerCP-Cy55-A', 'PE-Cy7-A', 'Ax647-A',
            'Ax700-A', 'Ax750-A', 'PacBlu-A',
            'Qdot525-A', 'PacOrange-A', 'Qdot605-A',
            'Qdot655-A', 'Qdot705-A', 'Time'
        ]

        npy_data = np.fromfile(npy_file_path)

        sample = Sample(
            npy_data,
            channel_labels=channels
        )

        self.assertIsInstance(sample, Sample)

    def test_load_from_pandas_multi_index(self):
        sample_orig = Sample("examples/data/100715.fcs", cache_original_events=True)
        pnn_orig = sample_orig.pnn_labels
        pns_orig = sample_orig.pns_labels

        df = sample_orig.as_dataframe(source='orig')

        sample_new = Sample(df)
        pnn_new = sample_new.pnn_labels
        pns_new = sample_new.pns_labels

        self.assertListEqual(pnn_orig, pnn_new)
        self.assertListEqual(pns_orig, pns_new)

    def test_load_from_unsupported_object(self):
        """Test Sample constructor raises ValueError loading an unsupported object"""
        self.assertRaises(ValueError, Sample, object())

    def test_comp_matrix_from_csv(self):
        sample = test_comp_sample

        self.assertIsNotNone(sample._comp_events)

    def test_clearing_comp_events(self):
        sample = copy.deepcopy(test_comp_sample)
        sample.apply_compensation(None)

        self.assertIsNone(sample._comp_events)

    def test_comp_matrix_from_pathlib_path(self):
        sample = test_comp_sample

        self.assertIsNotNone(sample._comp_events)

    def test_get_metadata(self):
        """Test Sample method get_metadata"""
        fcs_file_path = "examples/data/test_data_2d_01.fcs"

        sample = Sample(fcs_path_or_data=fcs_file_path)
        meta = sample.get_metadata()

        self.assertEqual(len(meta), 20)
        self.assertEqual(meta['p1n'], 'channel_A')

    @staticmethod
    def test_get_channel_index_by_channel_number_int():
        chan_number = data1_sample.get_channel_index(1)

        np.testing.assert_equal(0, chan_number)

    def test_get_channel_index_fails_by_chan_number_0(self):
        # chan numbers are indexed at 1, not 0
        self.assertRaises(ValueError, data1_sample.get_channel_index, 0)

    def test_get_channel_index_fails(self):
        # give an unsupported list as the arg
        self.assertRaises(ValueError, data1_sample.get_channel_index, [0, 1])

    @staticmethod
    def test_get_channel_events_raw():
        data_idx_0 = data1_sample.get_channel_events(0, source='raw')

        np.testing.assert_equal(data1_sample._raw_events[:, 0], data_idx_0)

    @staticmethod
    def test_get_channel_events_comp():
        sample = test_comp_sample

        data_idx_6 = sample.get_channel_events(6, source='comp')

        np.testing.assert_equal(sample._comp_events[:, 6], data_idx_6)

    @staticmethod
    def test_get_channel_events_xform():
        sample = copy.deepcopy(test_comp_sample)
        sample.apply_transform(xform_logicle)

        data_idx_6 = sample.get_channel_events(6, source='xform')

        np.testing.assert_equal(sample._transformed_events[:, 6], data_idx_6)

    def test_get_channel_events_subsample(self):
        sample = Sample(data1_fcs_path, subsample=500)

        data_idx_6 = sample.get_channel_events(6, source='raw', subsample=True)

        self.assertEqual(len(data_idx_6), 500)

    def test_get_subsampled_orig_events(self):
        sample = Sample(data1_fcs_path, cache_original_events=True, subsample=500)

        events = sample.get_events(source='orig', subsample=True)

        self.assertEqual(events.shape[0], 500)

    def test_get_subsampled_orig_events_not_cached(self):
        sample = Sample(data1_fcs_path, cache_original_events=False, subsample=500)

        self.assertRaises(ValueError, sample.get_events, source='orig', subsample=True)

    def test_get_subsampled_raw_events(self):
        sample = Sample(data1_fcs_path, subsample=500)

        events = sample.get_events(source='raw', subsample=True)

        self.assertEqual(events.shape[0], 500)

    def test_get_subsampled_comp_events(self):
        sample = copy.deepcopy(test_comp_sample)
        sample.subsample_events(500)

        events = sample.get_events(source='comp', subsample=True)

        self.assertEqual(events.shape[0], 500)

    def test_get_subsampled_xform_events(self):
        sample = copy.deepcopy(test_comp_sample)
        sample.subsample_events(500)
        sample.apply_transform(xform_logicle)

        events = sample.get_events(source='xform', subsample=True)

        self.assertEqual(events.shape[0], 500)

    def test_get_compensated_events_if_no_comp(self):
        sample = test_comp_sample_uncomp

        self.assertRaises(AttributeError, sample.get_events, source='comp')

    def test_get_transformed_events_if_no_xform_raises(self):
        sample = test_comp_sample_uncomp

        self.assertRaises(AttributeError, sample.get_events, source='xform')

    @staticmethod
    def test_get_transformed_events_exclude_scatter():
        sample = copy.deepcopy(test_comp_sample)
        sample.apply_transform(xform_logicle, include_scatter=False)

        fsc_a_index = sample.get_channel_index('FSC-A')
        data_fsc_a = sample.get_channel_events(fsc_a_index, source='xform')

        np.testing.assert_equal(sample._raw_events[:, fsc_a_index], data_fsc_a)

    def test_get_transformed_events_include_scatter(self):
        sample = copy.deepcopy(test_comp_sample)
        sample.apply_transform(xform_logicle, include_scatter=True)

        fsc_a_index = sample.get_channel_index('FSC-A')
        data_fsc_a_xform = sample.get_channel_events(fsc_a_index, source='xform')
        data_fsc_a_raw = sample.get_channel_events(fsc_a_index, source='raw')

        np.testing.assert_equal(sample._transformed_events[:, fsc_a_index], data_fsc_a_xform)
        self.assertEqual(data_fsc_a_raw[0], 118103.25)
        self.assertEqual(round(data_fsc_a_xform[0], 3), 1.238)

    def test_get_events_as_data_frame_xform(self):
        sample = copy.deepcopy(data1_sample)
        sample.apply_transform(xform_logicle)
        df = sample.as_dataframe(source='xform')

        self.assertIsInstance(df, pd.DataFrame)
        np.testing.assert_equal(df.values, sample.get_events(source='xform'))

    def test_get_events_as_data_frame_comp(self):
        sample = test_comp_sample

        df = sample.as_dataframe(source='comp')

        self.assertIsInstance(df, pd.DataFrame)
        np.testing.assert_equal(df.values, sample.get_events(source='comp'))

    def test_get_events_as_data_frame_raw(self):
        df = data1_sample.as_dataframe(source='raw')

        self.assertIsInstance(df, pd.DataFrame)
        np.testing.assert_equal(df.values, data1_sample.get_events(source='raw'))

    def test_get_events_as_data_frame_orig(self):
        df = data1_sample_with_orig.as_dataframe(source='orig')

        self.assertIsInstance(df, pd.DataFrame)
        np.testing.assert_equal(df.values, data1_sample_with_orig.get_events(source='orig'))

    def test_get_events_as_data_frame_column_order(self):
        orig_col_order = ['FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H', 'FL2-A', 'FL4-H', 'Time']
        new_col_order = ['FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL2-A', 'FL3-H', 'FL4-H', 'Time']
        col_to_check = 'FL2-A'

        df = data1_sample.as_dataframe(source='raw')
        df_reorder = data1_sample.as_dataframe(source='raw', col_order=new_col_order)

        self.assertListEqual(list(df.columns.get_level_values(0)), orig_col_order)
        self.assertListEqual(list(df_reorder.columns.get_level_values(0)), new_col_order)

        np.testing.assert_equal(df[col_to_check].values, df_reorder[col_to_check])

    def test_get_events_as_data_frame_new_column_names(self):
        new_cols = ['FSC-H', 'SSC-H', 'FLR1-H', 'FLR2-H', 'FLR3-H', 'FLR2-A', 'FLR4-H', 'Time']

        df = data1_sample.as_dataframe(source='raw', col_names=new_cols)

        self.assertListEqual(list(df.columns), new_cols)

    @staticmethod
    def test_fully_custom_transform():
        sample1 = Sample(fcs_path_or_data=data1_fcs_path)
        sample2 = Sample(fcs_path_or_data=data1_fcs_path)

        custom_xforms = {
            'FL1-H': xform_biex1,
            'FL2-H': xform_biex1,
            'FL3-H': xform_biex2,
            'FL2-A': xform_biex1,
            'FL4-H': xform_biex1
        }

        sample1.apply_transform(xform_biex1)
        sample2.apply_transform(custom_xforms)

        fl2_idx = sample1.get_channel_index('FL2-H')
        fl3_idx = sample1.get_channel_index('FL3-H')

        s1_fl2 = sample1.get_channel_events(fl2_idx, source='xform')
        s2_fl2 = sample2.get_channel_events(fl2_idx, source='xform')
        s1_fl3 = sample1.get_channel_events(fl3_idx, source='xform')
        s2_fl3 = sample2.get_channel_events(fl3_idx, source='xform')

        np.testing.assert_equal(s1_fl2, s2_fl2)
        np.testing.assert_raises(AssertionError, np.testing.assert_equal, s1_fl3, s2_fl3)

    def test_export_as_fcs(self):
        sample = test_comp_sample

        sample.export("test_fcs_export.fcs", source='comp', directory="examples")

        exported_fcs_file = "examples/test_fcs_export.fcs"
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
        fcs_file_path = "examples/data/100715.fcs"

        sample = Sample(fcs_path_or_data=fcs_file_path, cache_original_events=True)

        sample.export("test_fcs_export.fcs", source='orig', directory="examples")

        exported_fcs_file = "examples/test_fcs_export.fcs"
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

    def test_export_fcs_as_orig_with_pne_log_raises_error(self):
        # This test uses a file where some PnE values specify log scale
        # FlowIO does not currently support creating FCS files with log scale PnE values
        # Test that the Sample class raises a NotImplementedError for this case.
        sample = Sample(fcs_path_or_data=data1_fcs_path, cache_original_events=True)

        self.assertRaises(
            NotImplementedError,
            sample.export,
            "test_fcs_export.fcs",
            source='orig',
            directory="examples"
        )

    def test_export_fcs_as_raw_with_gain(self):
        # This test uses a file where the preprocessing makes the orig & raw events different.
        # File data1.fcs has 2 channels that specify a gain value other than 1.0.
        # The purpose here is to verify that importing the exported file has the same raw events
        # as the original file's raw events.
        sample = Sample(fcs_path_or_data=data1_fcs_path, cache_original_events=True)

        sample.export("test_fcs_export.fcs", source='raw', directory="examples")

        exported_fcs_file = "examples/test_fcs_export.fcs"
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
        fcs_file_path = "examples/data/8_color_data_set/fcs_files/101_DEN084Y5_15_E01_008_clean.fcs"

        sample = Sample(fcs_path_or_data=fcs_file_path)

        sample.export("test_fcs_export.fcs", source='raw', directory="examples")

        exported_fcs_file = "examples/test_fcs_export.fcs"
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

        sample.export("test_fcs_export.csv", source='comp', directory="examples")

        exported_csv_file = "examples/test_fcs_export.csv"
        exported_df = pd.read_csv(exported_csv_file)
        exported_sample = Sample(exported_df)
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

    def test_filter_negative_scatter(self):
        # there are 2 negative SSC-A events in this file (of 65016 total events)
        fcs_file_path = "examples/data/100715.fcs"
        sample = Sample(fcs_path_or_data=fcs_file_path, subsample=50000)
        sample.filter_negative_scatter(reapply_subsample=False)

        # using the default seed, the 2 negative events are in the subsample
        common_idx = np.intersect1d(sample.subsample_indices, sample.negative_scatter_indices)
        self.assertEqual(len(common_idx), 2)

        sample.filter_negative_scatter(reapply_subsample=True)
        common_idx = np.intersect1d(sample.subsample_indices, sample.negative_scatter_indices)
        self.assertEqual(len(common_idx), 0)

        self.assertEqual(sample.negative_scatter_indices.shape[0], 2)

    def test_export_exclude_negative_scatter(self):
        # there are 2 negative SSC-A events in this file (of 65016 total events)
        fcs_file_path = "examples/data/100715.fcs"
        sample = Sample(fcs_path_or_data=fcs_file_path)
        sample.filter_negative_scatter()

        neg_scatter_count = len(sample.negative_scatter_indices)

        exported_fcs_file = "examples/test_fcs_export.fcs"
        sample.export(exported_fcs_file, source='raw', exclude_neg_scatter=True)
        exported_sample = Sample(exported_fcs_file)
        os.unlink(exported_fcs_file)

        orig_event_count = sample.event_count
        exp_event_count = exported_sample.event_count

        self.assertEqual(exp_event_count, orig_event_count - neg_scatter_count)
