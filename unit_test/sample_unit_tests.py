import unittest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath('..'))

from flowkit import Sample


class LoadSampleTestCase(unittest.TestCase):
    """Tests for logicle transformation"""
    def test_load_from_fcs_file_path(self):
        """Test creating Sample object from an FCS file path"""
        fcs_file_path = "examples/test_data_2d_01.fcs"

        sample = Sample(fcs_path_or_data=fcs_file_path)

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

        sample = Sample(npy_data, channel_labels=channels)

        self.assertIsInstance(sample, Sample)

    def test_comp_matrix_from_csv(self):
        fcs_file_path = "examples/test_comp_example.fcs"
        comp_file_path = "examples/comp_complete_example.csv"

        sample = Sample(fcs_path_or_data=fcs_file_path, compensation=comp_file_path)

        self.assertIsNotNone(sample._comp_events)

    def test_plot_scatter(self):
        fcs_file_path = "examples/test_comp_example.fcs"
        comp_file_path = "examples/comp_complete_example.csv"

        sample = Sample(fcs_path_or_data=fcs_file_path, compensation=comp_file_path)
        sample.apply_asinh_transform()

        self.assertIsNotNone(sample._transformed_events)

        sample.plot_scatter(4, 5, contours=True)


if __name__ == '__main__':
    unittest.main()
