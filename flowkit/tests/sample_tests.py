"""
Unit tests for Sample class
"""
import unittest
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('../..'))

from flowkit import Sample, transforms

data1_fcs_path = 'examples/gate_ref/data1.fcs'
data1_sample = Sample(data1_fcs_path)

xform_logicle = transforms.LogicleTransform('logicle', param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
xform_biex1 = transforms.WSPBiexTransform('neg0', width=-100.0, negative=0.0)
xform_biex2 = transforms.WSPBiexTransform('neg1', width=-100.0, negative=1.0)


class SampleTestCase(unittest.TestCase):
    """Tests for loading FCS files as Sample objects"""
    def test_load_from_fcs_file_path(self):
        """Test creating Sample object from an FCS file path"""
        fcs_file_path = "examples/test_data_2d_01.fcs"

        sample = Sample(fcs_path_or_data=fcs_file_path)

        self.assertIsInstance(sample, Sample)

    def test_load_from_pathlib(self):
        """Test creating Sample object from a pathlib Path object"""
        fcs_file_path = "examples/test_data_2d_01.fcs"
        path = Path(fcs_file_path)
        sample = Sample(fcs_path_or_data=path)

        self.assertIsInstance(sample, Sample)

    def test_load_from_numpy_array(self):
        npy_file_path = "examples/test_comp_example.npy"
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
        sample_orig = Sample("examples/100715.fcs")
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
        fcs_file_path = "examples/test_comp_example.fcs"
        comp_file_path = "examples/comp_complete_example.csv"

        sample = Sample(
            fcs_path_or_data=fcs_file_path,
            compensation=comp_file_path,
            ignore_offset_error=True  # sample has off by 1 data offset
        )

        self.assertIsNotNone(sample._comp_events)

    def test_clearing_comp_events(self):
        fcs_file_path = "examples/test_comp_example.fcs"
        comp_file_path = "examples/comp_complete_example.csv"

        sample = Sample(
            fcs_path_or_data=fcs_file_path,
            compensation=comp_file_path,
            ignore_offset_error=True  # sample has off by 1 data offset
        )

        sample.apply_compensation(None)

        self.assertIsNone(sample._comp_events)

    def test_comp_matrix_from_pathlib_path(self):
        fcs_file_path = "examples/test_comp_example.fcs"
        comp_file_path = Path("examples/comp_complete_example.csv")

        sample = Sample(
            fcs_path_or_data=fcs_file_path,
            compensation=comp_file_path,
            ignore_offset_error=True  # sample has off by 1 data offset
        )

        self.assertIsNotNone(sample._comp_events)

    def test_get_metadata(self):
        """Test Sample method get_metadata"""
        fcs_file_path = "examples/test_data_2d_01.fcs"

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
    def test_get_channel_data_raw():
        data_idx_0 = data1_sample.get_channel_data(0, source='raw')

        np.testing.assert_equal(data1_sample._raw_events[:, 0], data_idx_0)

    @staticmethod
    def test_get_channel_data_comp():
        fcs_file_path = "examples/test_comp_example.fcs"
        comp_file_path = Path("examples/comp_complete_example.csv")

        sample = Sample(
            fcs_path_or_data=fcs_file_path,
            compensation=comp_file_path,
            ignore_offset_error=True  # sample has off by 1 data offset
        )

        data_idx_6 = sample.get_channel_data(6, source='comp')

        np.testing.assert_equal(sample._comp_events[:, 6], data_idx_6)

    @staticmethod
    def test_get_channel_data_xform():
        fcs_file_path = "examples/test_comp_example.fcs"
        comp_file_path = Path("examples/comp_complete_example.csv")

        sample = Sample(
            fcs_path_or_data=fcs_file_path,
            compensation=comp_file_path,
            ignore_offset_error=True  # sample has off by 1 data offset
        )
        sample.apply_transform(xform_logicle)

        data_idx_6 = sample.get_channel_data(6, source='xform')

        np.testing.assert_equal(sample._transformed_events[:, 6], data_idx_6)

    def test_get_channel_data_subsample_fails(self):
        self.assertRaises(
            ValueError,
            data1_sample.get_channel_data,
            0,
            source='raw',
            subsample=True
        )

    def test_get_channel_data_subsample(self):
        sample = Sample(data1_fcs_path)
        sample.subsample_events(500)

        data_idx_6 = sample.get_channel_data(6, source='raw', subsample=True)

        self.assertEqual(len(data_idx_6), 500)

    def test_get_subsampled_orig_events(self):
        sample = Sample(data1_fcs_path)
        sample.subsample_events(500)

        events = sample.get_orig_events(subsample=True)

        self.assertEqual(events.shape[0], 500)

    def test_get_subsampled_raw_events(self):
        sample = Sample(data1_fcs_path)
        sample.subsample_events(500)

        events = sample.get_raw_events(subsample=True)

        self.assertEqual(events.shape[0], 500)

    def test_get_subsampled_comp_events(self):
        fcs_file_path = "examples/test_comp_example.fcs"
        comp_file_path = Path("examples/comp_complete_example.csv")

        sample = Sample(
            fcs_path_or_data=fcs_file_path,
            compensation=comp_file_path,
            ignore_offset_error=True  # sample has off by 1 data offset
        )
        sample.subsample_events(500)

        events = sample.get_comp_events(subsample=True)

        self.assertEqual(events.shape[0], 500)

    def test_get_subsampled_xform_events(self):
        fcs_file_path = "examples/test_comp_example.fcs"
        comp_file_path = Path("examples/comp_complete_example.csv")

        sample = Sample(
            fcs_path_or_data=fcs_file_path,
            compensation=comp_file_path,
            ignore_offset_error=True  # sample has off by 1 data offset
        )
        sample.apply_transform(xform_logicle)

        sample.subsample_events(500)

        events = sample.get_transformed_events(subsample=True)

        self.assertEqual(events.shape[0], 500)

    def test_get_comp_events_if_no_comp(self):
        fcs_file_path = "examples/test_comp_example.fcs"

        sample = Sample(
            fcs_path_or_data=fcs_file_path,
            ignore_offset_error=True  # sample has off by 1 data offset
        )

        comp_events = sample.get_comp_events()

        self.assertIsNone(comp_events)

    def test_get_transformed_events_if_no_xform(self):
        fcs_file_path = "examples/test_comp_example.fcs"

        sample = Sample(
            fcs_path_or_data=fcs_file_path,
            ignore_offset_error=True  # sample has off by 1 data offset
        )

        xform_events = sample.get_transformed_events()

        self.assertIsNone(xform_events)

    @staticmethod
    def test_get_transformed_events_exclude_scatter():
        fcs_file_path = "examples/test_comp_example.fcs"
        comp_file_path = Path("examples/comp_complete_example.csv")

        sample = Sample(
            fcs_path_or_data=fcs_file_path,
            compensation=comp_file_path,
            ignore_offset_error=True  # sample has off by 1 data offset
        )
        sample.apply_transform(xform_logicle, include_scatter=False)

        fsc_a_index = sample.get_channel_index('FSC-A')
        data_fsc_a = sample.get_channel_data(fsc_a_index, source='xform')

        np.testing.assert_equal(sample._raw_events[:, fsc_a_index], data_fsc_a)

    def test_get_transformed_events_include_scatter(self):
        fcs_file_path = "examples/test_comp_example.fcs"
        comp_file_path = Path("examples/comp_complete_example.csv")

        sample = Sample(
            fcs_path_or_data=fcs_file_path,
            compensation=comp_file_path,
            ignore_offset_error=True  # sample has off by 1 data offset
        )
        sample.apply_transform(xform_logicle, include_scatter=True)

        fsc_a_index = sample.get_channel_index('FSC-A')
        data_fsc_a_xform = sample.get_channel_data(fsc_a_index, source='xform')
        data_fsc_a_raw = sample.get_channel_data(fsc_a_index, source='raw')

        np.testing.assert_equal(sample._transformed_events[:, fsc_a_index], data_fsc_a_xform)
        self.assertEqual(data_fsc_a_raw[0], 118103.25)
        self.assertEqual(round(data_fsc_a_xform[0], 3), 1.238)

    def test_get_events_as_data_frame_xform(self):
        data1_sample.apply_transform(xform_logicle)
        df = data1_sample.as_dataframe(source='xform')

        self.assertIsInstance(df, pd.DataFrame)
        np.testing.assert_equal(df.values, data1_sample.get_transformed_events())

    def test_get_events_as_data_frame_comp(self):
        fcs_file_path = "examples/test_comp_example.fcs"
        comp_file_path = "examples/comp_complete_example.csv"

        sample = Sample(
            fcs_path_or_data=fcs_file_path,
            compensation=comp_file_path,
            ignore_offset_error=True  # sample has off by 1 data offset
        )

        df = sample.as_dataframe(source='comp')

        self.assertIsInstance(df, pd.DataFrame)
        np.testing.assert_equal(df.values, sample.get_comp_events())

    def test_get_events_as_data_frame_raw(self):
        df = data1_sample.as_dataframe(source='raw')

        self.assertIsInstance(df, pd.DataFrame)
        np.testing.assert_equal(df.values, data1_sample.get_raw_events())

    def test_get_events_as_data_frame_orig(self):
        df = data1_sample.as_dataframe(source='orig')

        self.assertIsInstance(df, pd.DataFrame)
        np.testing.assert_equal(df.values, data1_sample.get_orig_events())

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

        s1_fl2 = sample1.get_channel_data(fl2_idx, source='xform')
        s2_fl2 = sample2.get_channel_data(fl2_idx, source='xform')
        s1_fl3 = sample1.get_channel_data(fl3_idx, source='xform')
        s2_fl3 = sample2.get_channel_data(fl3_idx, source='xform')

        np.testing.assert_equal(s1_fl2, s2_fl2)
        np.testing.assert_raises(AssertionError, np.testing.assert_equal, s1_fl3, s2_fl3)

    def test_create_fcs(self):
        fcs_file_path = "examples/test_comp_example.fcs"
        comp_file_path = Path("examples/comp_complete_example.csv")

        sample = Sample(
            fcs_path_or_data=fcs_file_path,
            compensation=comp_file_path,
            ignore_offset_error=True  # sample has off by 1 data offset
        )

        sample.export("test_fcs_export.fcs", source='comp', directory="examples")

        exported_fcs_file = "examples/test_fcs_export.fcs"
        exported_sample = Sample(fcs_path_or_data=exported_fcs_file)
        os.unlink(exported_fcs_file)

        self.assertIsInstance(exported_sample, Sample)

        # TODO: Excluding time channel here, as the difference was nearly 0.01. Need to investigate why the
        #       exported comp data isn't exactly equal
        np.testing.assert_almost_equal(sample._comp_events[:, :-1], exported_sample._raw_events[:, :-1], decimal=3)

    def test_create_csv(self):
        fcs_file_path = "examples/test_comp_example.fcs"
        comp_file_path = Path("examples/comp_complete_example.csv")

        sample = Sample(
            fcs_path_or_data=fcs_file_path,
            compensation=comp_file_path,
            ignore_offset_error=True  # sample has off by 1 data offset
        )

        sample.export("test_fcs_export.csv", source='comp', directory="examples")

        exported_csv_file = "examples/test_fcs_export.csv"
        exported_df = pd.read_csv(exported_csv_file)
        exported_sample = Sample(exported_df)
        os.unlink(exported_csv_file)

        self.assertIsInstance(exported_sample, Sample)

        # TODO: Need to investigate why the exported comp data isn't exactly equal
        np.testing.assert_almost_equal(sample._comp_events[:, :], exported_sample._raw_events[:, :], decimal=3)

    def test_filter_negative_scatter(self):
        # there are 2 negative SSC-A events in this file (of 65016 total events)
        fcs_file_path = "examples/100715.fcs"
        sample = Sample(fcs_path_or_data=fcs_file_path)
        sample.subsample_events(50000)
        sample.filter_negative_scatter(reapply_subsample=False)

        # using the default seed, the 2 negative events are in the subsample
        common_idx = np.intersect1d(sample.subsample_indices, sample.negative_scatter_indices)
        self.assertEqual(len(common_idx), 2)

        sample.filter_negative_scatter(reapply_subsample=True)
        common_idx = np.intersect1d(sample.subsample_indices, sample.negative_scatter_indices)
        self.assertEqual(len(common_idx), 0)

        self.assertEqual(sample.negative_scatter_indices.shape[0], 2)

    def test_filter_anomalous_events(self):
        # there are 2 negative SSC-A events in this file (of 65016 total events)
        fcs_file_path = "examples/100715.fcs"
        sample = Sample(fcs_path_or_data=fcs_file_path)
        sample.subsample_events(50000)
        sample.filter_anomalous_events(reapply_subsample=False)

        # using the default seed, the 2 negative events are in the subsample
        common_idx = np.intersect1d(sample.subsample_indices, sample.anomalous_indices)
        self.assertGreater(len(common_idx), 0)

        sample.filter_anomalous_events(reapply_subsample=True)
        common_idx = np.intersect1d(sample.subsample_indices, sample.anomalous_indices)
        self.assertEqual(len(common_idx), 0)

        self.assertGreater(sample.anomalous_indices.shape[0], 0)
