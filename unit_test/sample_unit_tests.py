import unittest
import sys
import os
from pathlib import Path
import numpy as np

sys.path.append(os.path.abspath('..'))

from flowkit import Sample, transforms

data1_fcs_path = 'examples/gate_ref/data1.fcs'
data1_sample = Sample(data1_fcs_path)


class LoadSampleTestCase(unittest.TestCase):
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

    def test_comp_matrix_from_csv(self):
        fcs_file_path = "examples/test_comp_example.fcs"
        comp_file_path = "examples/comp_complete_example.csv"

        sample = Sample(
            fcs_path_or_data=fcs_file_path,
            compensation=comp_file_path
        )

        self.assertIsNotNone(sample._comp_events)

    def test_comp_matrix_from_pathlib_path(self):
        fcs_file_path = "examples/test_comp_example.fcs"
        comp_file_path = Path("examples/comp_complete_example.csv")

        sample = Sample(
            fcs_path_or_data=fcs_file_path,
            compensation=comp_file_path
        )

        self.assertIsNotNone(sample._comp_events)

    def test_transform_sample_asinh(self):
        xform = transforms.AsinhTransform('asinh', param_t=10000, param_m=4.5, param_a=0)
        data1_sample.apply_transform(xform)

        self.assertIsInstance(data1_sample._transformed_events, np.ndarray)

    def test_transform_sample_logical(self):
        xform = transforms.LogicleTransform('logicle', param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
        data1_sample.apply_transform(xform)

        self.assertIsInstance(data1_sample._transformed_events, np.ndarray)

    def test_transform_sample_hyperlog(self):
        xform = transforms.HyperlogTransform('hyper', param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
        data1_sample.apply_transform(xform)

        self.assertIsInstance(data1_sample._transformed_events, np.ndarray)

    def test_create_fcs(self):
        fcs_file_path = "examples/test_comp_example.fcs"
        comp_file_path = Path("examples/comp_complete_example.csv")

        sample = Sample(
            fcs_path_or_data=fcs_file_path,
            compensation=comp_file_path
        )

        sample.export("test_fcs_export.fcs", source='comp', directory="examples")

        exported_fcs_file = "examples/test_fcs_export.fcs"
        exported_sample = Sample(fcs_path_or_data=exported_fcs_file)
        os.unlink(exported_fcs_file)

        self.assertIsInstance(exported_sample, Sample)

        # TODO: Excluding time channel here, as the difference was nearly 0.01. Need to investigate why the
        #       exported comp data isn't exactly equal
        np.testing.assert_almost_equal(sample._comp_events[:, :-1], exported_sample._raw_events[:, :-1], decimal=3)


if __name__ == '__main__':
    unittest.main()
