"""
Unit tests for Transform subclasses
"""
import unittest
import numpy as np
import warnings

from flowkit import transforms, generate_transforms

from tests.test_config import data1_sample, null_chan_sample, test_data_range1


data1_raw_events = data1_sample.get_events(source='raw')


class TransformsTestCase(unittest.TestCase):
    """Tests for loading FCS files as Sample objects"""
    def test_transform_sample_linear(self):
        xform = transforms.LinearTransform(param_t=data1_raw_events.max(), param_a=0.0)
        data1_sample.apply_transform(xform)

        xform_events = xform.apply(data1_raw_events)

        self.assertIsInstance(xform_events, np.ndarray)
        self.assertEqual(np.max(xform_events), 1.0)
        self.assertEqual(np.min(xform_events), 0.0)

    def test_transform_sample_linear_1d(self):
        xform = transforms.LinearTransform(param_t=data1_raw_events.max(), param_a=0.0)
        xform_lut = {
            'FL1-H': xform
        }
        data1_sample.apply_transform(xform_lut)

        self.assertIsInstance(data1_sample._transformed_events, np.ndarray)

    @staticmethod
    def test_inverse_linear_transform():
        xform = transforms.LinearTransform(param_t=10000, param_a=0)
        y = xform.apply(test_data_range1)
        x = xform.inverse(y)

        np.testing.assert_array_almost_equal(test_data_range1, x, decimal=10)

    def test_transform_sample_asinh(self):
        xform = transforms.AsinhTransform(param_t=10000, param_m=4.5, param_a=0)
        data1_sample.apply_transform(xform)

        raw_events = data1_sample.get_events(source='raw')
        xform_events = data1_sample.get_events(source='xform')

        self.assertIsInstance(xform_events, np.ndarray)
        self.assertRaises(AssertionError, np.testing.assert_array_equal, raw_events, xform_events)

    def test_transform_sample_asinh_1d(self):
        xform = transforms.AsinhTransform(param_t=10000, param_m=4.5, param_a=0)
        xform_lut = {
            'FL1-H': xform
        }
        data1_sample.apply_transform(xform_lut)

        self.assertIsInstance(data1_sample._transformed_events, np.ndarray)

    @staticmethod
    def test_inverse_asinh_transform():
        xform = transforms.AsinhTransform(param_t=10000, param_m=4.5, param_a=0)
        y = xform.apply(test_data_range1)
        x = xform.inverse(y)

        np.testing.assert_array_almost_equal(test_data_range1, x, decimal=10)

    def test_transform_ratio_raises_not_implemented(self):
        ratio_dims = ['FL1-H', 'FL2-H']
        xform = transforms.RatioTransform(ratio_dims, param_a=1.0, param_b=0.0, param_c=0.0)

        self.assertRaises(NotImplementedError, data1_sample.apply_transform, xform)

    def test_transform_ratio_raises_bad_dim_ids(self):
        ratio_dims = ['FL1-H', 'FL2-H', 'FL3-H']

        self.assertRaises(
            ValueError,
            transforms.RatioTransform,
            ratio_dims, param_a=1.0, param_b=0.0, param_c=0.0
        )

    def test_transform_ratio(self):
        ratio_dims = ['FL1-H', 'FL2-H']
        xform = transforms.RatioTransform(ratio_dims, param_a=1.0, param_b=0.0, param_c=0.0)

        xform_events = xform.apply(data1_sample)

        self.assertIsInstance(xform_events, np.ndarray)
        self.assertTupleEqual(xform_events.shape, (13367,))

    def test_transform_sample_log(self):
        xform = transforms.LogTransform(param_t=10000, param_m=4.5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            data1_sample.apply_transform(xform)

        self.assertIsInstance(data1_sample._transformed_events, np.ndarray)

    def test_transform_sample_log_1d(self):
        xform = transforms.LogTransform(param_t=10000, param_m=4.5)
        xform_lut = {
            'FL1-H': xform
        }
        data1_sample.apply_transform(xform_lut)

        self.assertIsInstance(data1_sample._transformed_events, np.ndarray)

    @staticmethod
    def test_inverse_log_transform():
        xform = transforms.LogTransform(param_t=10000, param_m=4.5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y = xform.apply(test_data_range1)
        x = xform.inverse(y)

        np.testing.assert_array_almost_equal(test_data_range1, x, decimal=10)

    def test_transform_sample_logicle(self):
        xform = transforms.LogicleTransform(param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
        data1_sample.apply_transform(xform)

        self.assertIsInstance(data1_sample._transformed_events, np.ndarray)

    def test_transform_sample_logicle_1d(self):
        xform = transforms.LogicleTransform(param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
        xform_lut = {
            'FL1-H': xform
        }
        data1_sample.apply_transform(xform_lut)

        self.assertIsInstance(data1_sample._transformed_events, np.ndarray)

    @staticmethod
    def test_inverse_logicle_transform():
        xform = transforms.LogicleTransform(param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
        y = xform.apply(test_data_range1)
        x = xform.inverse(y)

        np.testing.assert_array_almost_equal(test_data_range1, x, decimal=10)

    def test_transform_sample_hyperlog(self):
        xform = transforms.HyperlogTransform(param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
        data1_sample.apply_transform(xform)

        self.assertIsInstance(data1_sample._transformed_events, np.ndarray)

    def test_transform_sample_hyperlog_1d(self):
        xform = transforms.HyperlogTransform(param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
        xform_lut = {
            'FL1-H': xform
        }
        data1_sample.apply_transform(xform_lut)

        self.assertIsInstance(data1_sample._transformed_events, np.ndarray)

    @staticmethod
    def test_inverse_hyperlog_transform():
        xform = transforms.HyperlogTransform(param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
        y = xform.apply(test_data_range1)
        x = xform.inverse(y)

        np.testing.assert_array_almost_equal(test_data_range1, x, decimal=10)

    def test_transform_sample_wsp_log(self):
        xform = transforms.WSPLogTransform(offset=0.5, decades=4.5)
        data1_sample.apply_transform(xform)

        self.assertIsInstance(data1_sample._transformed_events, np.ndarray)

    def test_transform_sample_wsp_log_1d(self):
        xform = transforms.WSPLogTransform(offset=0.5, decades=4.5)
        xform_lut = {
            'FL1-H': xform
        }
        data1_sample.apply_transform(xform_lut)

        self.assertIsInstance(data1_sample._transformed_events, np.ndarray)

    @staticmethod
    def test_inverse_wsp_biex_transform():
        xform = transforms.WSPBiexTransform()
        y = xform.apply(test_data_range1)
        x = xform.inverse(y)

        np.testing.assert_array_almost_equal(test_data_range1, x, decimal=10)

    def test_generate_transforms_defaults(self):
        xform_lut = generate_transforms(data1_sample)

        self.assertEqual(len(xform_lut), len(data1_sample.pnn_labels))

        # pick a fluoro channel label and check Transform type
        fluoro_label = data1_sample.pnn_labels[data1_sample.fluoro_indices[0]]
        self.assertIsInstance(xform_lut[fluoro_label], transforms.LogicleTransform)

        # verify time use max time
        time_max = data1_sample.get_channel_events(data1_sample.time_index, source='raw').max()
        self.assertEqual(xform_lut['Time'].param_t, time_max)

    def test_generate_transforms_default_asinh(self):
        xform_lut = generate_transforms(
            data1_sample,
            fluoro_xform_class=transforms.AsinhTransform
        )

        self.assertEqual(len(xform_lut), len(data1_sample.pnn_labels))

        # pick a fluoro channel label and check Transform type
        fluoro_label = data1_sample.pnn_labels[data1_sample.fluoro_indices[0]]
        self.assertIsInstance(xform_lut[fluoro_label], transforms.AsinhTransform)

    def test_generate_transforms_default_hyperlog(self):
        xform_lut = generate_transforms(
            data1_sample,
            fluoro_xform_class=transforms.HyperlogTransform
        )

        self.assertEqual(len(xform_lut), len(data1_sample.pnn_labels))

        # pick a fluoro channel label and check Transform type
        fluoro_label = data1_sample.pnn_labels[data1_sample.fluoro_indices[0]]
        self.assertIsInstance(xform_lut[fluoro_label], transforms.HyperlogTransform)

    def test_generate_transforms_default_log(self):
        xform_lut = generate_transforms(
            data1_sample,
            fluoro_xform_class=transforms.LogTransform
        )

        self.assertEqual(len(xform_lut), len(data1_sample.pnn_labels))

        # pick a fluoro channel label and check Transform type
        fluoro_label = data1_sample.pnn_labels[data1_sample.fluoro_indices[0]]
        self.assertIsInstance(xform_lut[fluoro_label], transforms.LogTransform)

    def test_generate_transforms_default_wsp_biex(self):
        xform_lut = generate_transforms(
            data1_sample,
            fluoro_xform_class=transforms.WSPBiexTransform
        )

        self.assertEqual(len(xform_lut), len(data1_sample.pnn_labels))

        # pick a fluoro channel label and check Transform type
        fluoro_label = data1_sample.pnn_labels[data1_sample.fluoro_indices[0]]
        self.assertIsInstance(xform_lut[fluoro_label], transforms.WSPBiexTransform)

    def test_generate_transforms_default_null_channel(self):
        # null channel sample has 'FL1-H' nullified
        xform_lut = generate_transforms(null_chan_sample)

        self.assertEqual(len(xform_lut), len(data1_sample.pnn_labels) - 1)

        # ensure null channel label is missing from xform LUT
        null_label = 'FL1-H'
        self.assertNotIn(null_label, xform_lut)

    def test_generate_transforms_transform_not_supported(self):
        self.assertRaises(
            NotImplementedError,
            generate_transforms,
            data1_sample,
            fluoro_xform_class=transforms.RatioTransform
        )

    def test_generate_transforms_transform_instance_scatter(self):
        # specify a Transform instance for scatter channels
        scatter_xform = transforms.LogTransform(param_t=262144, param_m=4.1)

        xform_lut = generate_transforms(
            data1_sample,
            scatter_xform_class=scatter_xform
        )

        self.assertEqual(len(xform_lut), len(data1_sample.pnn_labels))

        # pick a scatter channel label and check Transform type
        scatter_label = data1_sample.pnn_labels[data1_sample.scatter_indices[0]]
        scatter_xform = xform_lut[scatter_label]
        self.assertIsInstance(scatter_xform, transforms.LogTransform)

        self.assertEqual(scatter_xform.param_m, 4.1)

    def test_generate_transforms_transform_instance_fluoro(self):
        # specify a Transform instance for fluoro channels
        fluoro_xform = transforms.AsinhTransform(param_t=262144, param_m=4.1, param_a=0.0)

        xform_lut = generate_transforms(
            data1_sample,
            fluoro_xform_class=fluoro_xform
        )

        self.assertEqual(len(xform_lut), len(data1_sample.pnn_labels))

        # pick a fluoro channel label and check Transform type
        fluoro_label = data1_sample.pnn_labels[data1_sample.fluoro_indices[0]]
        fluoro_xform = xform_lut[fluoro_label]
        self.assertIsInstance(fluoro_xform, transforms.AsinhTransform)

        self.assertEqual(fluoro_xform.param_m, 4.1)

    def test_generate_transforms_transform_instance_time(self):
        # specify a Transform instance for time channel
        # make up a max time to check the custom instance is returned
        time_max = 123
        time_xform = transforms.LinearTransform(param_t=time_max, param_a=0.0)

        xform_lut = generate_transforms(
            data1_sample,
            time_xform_class=time_xform
        )

        self.assertEqual(len(xform_lut), len(data1_sample.pnn_labels))

        # check Transform type & params
        time_xform = xform_lut['Time']
        self.assertIsInstance(time_xform, transforms.LinearTransform)

        self.assertEqual(time_xform.param_t, time_max)
