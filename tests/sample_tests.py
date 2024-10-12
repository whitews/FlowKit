"""
Unit tests for Sample class
"""
import copy
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import flowio
import warnings

from flowkit import Sample, transforms, read_multi_dataset_fcs
from flowkit.exceptions import DataOffsetDiscrepancyError

data1_fcs_path = 'data/gate_ref/data1.fcs'
data1_sample = Sample(data1_fcs_path)
data1_sample_with_orig = Sample(data1_fcs_path, cache_original_events=True)

xform_logicle = transforms.LogicleTransform(param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
xform_biex1 = transforms.WSPBiexTransform(width=-100.0, negative=0.0)
xform_biex2 = transforms.WSPBiexTransform(width=-100.0, negative=1.0)

fcs_file_path = "data/test_comp_example.fcs"
comp_file_path = "data/comp_complete_example.csv"

fcs_2d_file_path = "data/test_data_2d_01.fcs"

fcs_index_sorted_path = "data/index_sorted/index_sorted_example.fcs"

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
        sample = Sample(fcs_path_or_data=fcs_2d_file_path)

        self.assertIsInstance(sample, Sample)

    def test_load_from_flowio_flowdata_object(self):
        """Test creating Sample object from an FCS file path"""
        flow_data = flowio.FlowData(fcs_2d_file_path)

        self.assertIsInstance(flow_data, flowio.FlowData)

        sample = Sample(flow_data)

        self.assertIsInstance(sample, Sample)

    def test_load_from_pathlib(self):
        """Test creating Sample object from a pathlib Path object"""
        path = Path(fcs_2d_file_path)
        sample = Sample(fcs_path_or_data=path)

        self.assertIsInstance(sample, Sample)

    def test_load_from_io_base(self):
        """Test creating Sample object from a IOBase object"""
        f = open(fcs_2d_file_path, 'rb')
        sample = Sample(fcs_path_or_data=f)
        f.close()

        self.assertIsInstance(sample, Sample)

    def test_load_from_numpy_array(self):
        npy_file_path = "data/test_comp_example.npy"
        # noinspection SpellCheckingInspection
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
            sample_id='my_sample',
            channel_labels=channels
        )

        self.assertIsInstance(sample, Sample)

    def test_load_numpy_array_no_id_raises(self):
        npy_file_path = "data/test_comp_example.npy"
        # noinspection SpellCheckingInspection
        channels = [
            'FSC-A', 'FSC-W', 'SSC-A',
            'Ax488-A', 'PE-A', 'PE-TR-A',
            'PerCP-Cy55-A', 'PE-Cy7-A', 'Ax647-A',
            'Ax700-A', 'Ax750-A', 'PacBlu-A',
            'Qdot525-A', 'PacOrange-A', 'Qdot605-A',
            'Qdot655-A', 'Qdot705-A', 'Time'
        ]

        npy_data = np.fromfile(npy_file_path)

        self.assertRaises(ValueError, Sample, npy_data, channel_labels=channels)

    def test_load_dataframe_no_id_raises(self):
        npy_file_path = "data/test_comp_example.npy"
        # noinspection SpellCheckingInspection
        channels = [
            'FSC-A', 'FSC-W', 'SSC-A',
            'Ax488-A', 'PE-A', 'PE-TR-A',
            'PerCP-Cy55-A', 'PE-Cy7-A', 'Ax647-A',
            'Ax700-A', 'Ax750-A', 'PacBlu-A',
            'Qdot525-A', 'PacOrange-A', 'Qdot605-A',
            'Qdot655-A', 'Qdot705-A', 'Time'
        ]

        npy_data = np.fromfile(npy_file_path)
        npy_data = np.reshape(npy_data, (-1, len(channels)))
        df_data = pd.DataFrame(npy_data, columns=channels)

        self.assertRaises(ValueError, Sample, df_data)

    def test_load_from_pandas_multi_index(self):
        sample_orig = Sample("data/100715.fcs", cache_original_events=True)
        pnn_orig = sample_orig.pnn_labels
        pns_orig = sample_orig.pns_labels

        df = sample_orig.as_dataframe(source='orig')

        sample_new = Sample(df, sample_id='my_sample')
        pnn_new = sample_new.pnn_labels
        pns_new = sample_new.pns_labels

        self.assertListEqual(pnn_orig, pnn_new)
        self.assertListEqual(pns_orig, pns_new)

    def test_load_from_unsupported_object(self):
        """Test Sample constructor raises ValueError loading an unsupported object"""
        self.assertRaises(ValueError, Sample, object())

    def test_data_start_offset_discrepancy(self):
        fcs_file = "data/noncompliant/data_start_offset_discrepancy_example.fcs"
        self.assertRaises(DataOffsetDiscrepancyError, Sample, fcs_file)

    def test_data_stop_offset_discrepancy(self):
        fcs_file = "data/noncompliant/data_stop_offset_discrepancy_example.fcs"
        self.assertRaises(DataOffsetDiscrepancyError, Sample, fcs_file)

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
        sample = Sample(fcs_path_or_data=fcs_2d_file_path)
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

    def test_get_events_using_event_mast(self):
        sample = Sample(data1_fcs_path, subsample=500)

        # create event mask selecting all odd events
        event_mask = np.zeros(sample.event_count, dtype=bool)
        event_mask[1::2] = True

        events = sample.get_events(source='raw', event_mask=event_mask)
        events_sub = sample.get_events(source='raw', subsample=True, event_mask=event_mask)

        self.assertEqual(events.shape[0], 6683)
        self.assertEqual(events_sub.shape[0], 233)

    def test_get_events_invalid_source_raises(self):
        sample = test_comp_sample_uncomp

        # make up an invalid source type
        self.assertRaises(ValueError, sample.get_events, source='major')

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

    def test_get_events_as_data_frame_col_index(self):
        # verifies 'col_multi_index' option works as expected
        # by default the col index will be MultiIndex
        df_multi = data1_sample.as_dataframe(source='raw')

        # turn off multi-index for simple column index
        df_simple = data1_sample.as_dataframe(source='raw', col_multi_index=False)

        self.assertIsInstance(df_multi.columns, pd.MultiIndex)
        self.assertIsInstance(df_simple.columns, pd.Index)

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

    def test_rename_channel(self):
        sample = copy.deepcopy(data1_sample)
        chan_orig = 'FL1-H'
        chan_new = 'CD4-H'
        chan_pns_new = 'FITC'

        sample.rename_channel(chan_orig, chan_new, new_pns_label=chan_pns_new)

        chan_num = sample.get_channel_number_by_label(chan_new)
        self.assertEqual(chan_num, 3)

        chan_idx = sample.get_channel_index(chan_new)
        self.assertEqual(chan_idx, 2)

        pnn_labels = sample.pnn_labels
        pns_labels = sample.pns_labels

        self.assertEqual(chan_new, pnn_labels[chan_idx])
        self.assertEqual(chan_pns_new, pns_labels[chan_idx])

        df_channels = sample.channels
        self.assertEqual(chan_new, df_channels.loc[chan_idx].pnn)
        self.assertEqual(chan_pns_new, df_channels.loc[chan_idx].pns)

    def test_rename_channel_raises(self):
        sample = copy.deepcopy(data1_sample)
        chan_orig = 'asdf'
        chan_new = 'CD4-H'

        self.assertRaises(ValueError, sample.rename_channel, chan_orig, chan_new)

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

    def test_filter_negative_scatter(self):
        # there are 2 negative SSC-A events in this file (of 65016 total events)
        sample_file_path = "data/100715.fcs"
        sample = Sample(fcs_path_or_data=sample_file_path, subsample=50000)
        sample.filter_negative_scatter(reapply_subsample=False)

        # using the default seed, the 2 negative events are in the subsample
        common_idx = np.intersect1d(sample.subsample_indices, sample.negative_scatter_indices)
        self.assertEqual(len(common_idx), 2)

        sample.filter_negative_scatter(reapply_subsample=True)
        common_idx = np.intersect1d(sample.subsample_indices, sample.negative_scatter_indices)
        self.assertEqual(len(common_idx), 0)

        self.assertEqual(sample.negative_scatter_indices.shape[0], 2)

    def test_get_index_sorted_locations(self):
        sample = Sample(fcs_path_or_data=fcs_index_sorted_path)

        idx_sorted_locations = sample.get_index_sorted_locations()

        # there are 384 events in the file, each should have a well location
        self.assertEqual(len(idx_sorted_locations), 384)

    def test_get_index_sorted_locations_is_empty(self):
        # data 1 sample has no index sort locations, so should return empty list
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            idx_sorted_locations = data1_sample.get_index_sorted_locations()

        # there are 384 events in the file, each should have a well location
        self.assertEqual(len(idx_sorted_locations), 0)

    def test_load_multi_dataset_file(self):
        sample_file_path = "data/multi_dataset_fcs/coulter.lmd"

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            samples = read_multi_dataset_fcs(sample_file_path, ignore_offset_error=True)

        self.assertEqual(len(samples), 2)
        self.assertIsInstance(samples[0], Sample)
        self.assertIsInstance(samples[1], Sample)
