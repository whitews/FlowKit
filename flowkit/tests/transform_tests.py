"""
Unit tests for Transform sub-classes
"""
import unittest
import numpy as np

from flowkit import Sample, transforms

data1_fcs_path = 'examples/gate_ref/data1.fcs'
data1_sample = Sample(data1_fcs_path)
data1_raw_events = data1_sample.get_raw_events()

test_data_range = np.linspace(0.0, 10.0, 101)


class TransformsTestCase(unittest.TestCase):
    """Tests for loading FCS files as Sample objects"""
    def test_transform_sample_asinh(self):
        xform = transforms.AsinhTransform('asinh', param_t=10000, param_m=4.5, param_a=0)
        data1_sample.apply_transform(xform)

        self.assertIsInstance(data1_sample._transformed_events, np.ndarray)

    @staticmethod
    def test_inverse_asinh_transform():
        xform = transforms.AsinhTransform('asinh', param_t=10000, param_m=4.5, param_a=0)
        y = xform.apply(test_data_range)
        x = xform.inverse(y)

        np.testing.assert_array_almost_equal(test_data_range, x, decimal=10)

    def test_transform_sample_logicle(self):
        xform = transforms.LogicleTransform('logicle', param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
        data1_sample.apply_transform(xform)

        self.assertIsInstance(data1_sample._transformed_events, np.ndarray)

    def test_transform_sample_hyperlog(self):
        xform = transforms.HyperlogTransform('hyper', param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
        data1_sample.apply_transform(xform)

        self.assertIsInstance(data1_sample._transformed_events, np.ndarray)

    def test_transform_sample_wsp_log(self):
        xform = transforms.WSPLogTransform('wsp_log', offset=0.5, decades=4.5)
        data1_sample.apply_transform(xform)

        self.assertIsInstance(data1_sample._transformed_events, np.ndarray)
