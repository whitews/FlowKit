"""
Matrix class tests
"""

import copy
import unittest
import numpy as np
import pandas as pd
import warnings
import flowkit as fk

from tests.test_config import (
    fcs_spill,
    fcs_spill_header,
    csv_8c_comp_file_path,
    detectors_8c,
    fluorochromes_8c,
    csv_8c_comp_null_channel_file_path,
    spectral_event_data,
    spectral_fluoro_indices,
    spectral_comp_matrix,
    spectral_all_detectors,
    spectral_true_detectors,
    spectral_sample,
    spectral_truth_comp_events
)


class MatrixTestCase(unittest.TestCase):
    """Tests related to compensation matrices and the Matrix class"""

    def test_matrix_from_fcs_spill(self):
        comp_mat = fk.Matrix(fcs_spill, fcs_spill_header)

        self.assertIsInstance(comp_mat, fk.Matrix)

    def test_parse_csv_file(self):
        comp_mat = fk.Matrix(csv_8c_comp_file_path, detectors_8c)

        self.assertIsInstance(comp_mat, fk.Matrix)

    def test_matrix_equals(self):
        comp_mat = fk.Matrix(csv_8c_comp_file_path, detectors_8c)

        comp_mat2 = copy.deepcopy(comp_mat)

        self.assertEqual(comp_mat, comp_mat2)

    def test_matrix_equals_fails(self):
        comp_mat = fk.Matrix(csv_8c_comp_file_path, detectors_8c)

        # copy & modify matrix array
        comp_mat2 = copy.deepcopy(comp_mat)
        comp_mat2.matrix[0, 1] = comp_mat2.matrix[0, 1] + 0.01

        self.assertNotEqual(comp_mat, comp_mat2)

    def test_matrix_as_dataframe(self):
        comp_mat = fk.Matrix(
            csv_8c_comp_file_path, detectors_8c, fluorochromes=fluorochromes_8c
        )

        # test with detectors as labels
        comp_df_detectors = comp_mat.as_dataframe()

        # test with fluorochromes as labels
        comp_df_fluorochromes = comp_mat.as_dataframe(fluoro_labels=True)

        self.assertIsInstance(comp_df_detectors, pd.DataFrame)
        self.assertIsInstance(comp_df_fluorochromes, pd.DataFrame)

    def test_reserved_matrix_id_uncompensated(self):
        self.assertRaises(
            ValueError, fk.Matrix, "uncompensated", fcs_spill, fcs_spill_header
        )

    @staticmethod
    def test_matrix_inverse():
        fcs_file_path = "data/test_comp_example.fcs"
        comp_file_path = "data/comp_complete_example.csv"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sample = fk.Sample(
                fcs_path_or_data=fcs_file_path,
                compensation=comp_file_path,
                ignore_offset_error=True,  # sample has off by 1 data offset
            )
        matrix = sample.compensation

        data_raw = sample.get_events(source="raw")
        inv_data = matrix.inverse(sample)

        np.testing.assert_almost_equal(inv_data, data_raw, 10)

    def test_null_channels(self):
        # pretend FITC is a null channel
        null_channels = ["TNFa FITC FLR-A"]

        comp_mat = fk.Matrix(
            csv_8c_comp_null_channel_file_path,
            detectors_8c,
            null_channels=null_channels,
        )

        fcs_file_path = (
            "data/8_color_data_set/fcs_files/101_DEN084Y5_15_E01_008_clean.fcs"
        )

        # test with a sample not using null channels and one using null channels
        sample1 = fk.Sample(fcs_file_path, null_channel_list=None)
        sample2 = fk.Sample(fcs_file_path, null_channel_list=null_channels)

        comp_events1 = comp_mat.apply(sample1)
        comp_events2 = comp_mat.apply(sample2)

        self.assertIsInstance(comp_events1, np.ndarray)
        self.assertIsInstance(comp_events2, np.ndarray)

class SpectralMatrixTestCase(unittest.TestCase):
    """
    Tests related to the SpectralMatrix class
    """
    def setUp(self):
        self.spectral_event_data = spectral_event_data
        self.spectral_fluoro_indices = spectral_fluoro_indices
        self.spectral_comp_matrix = spectral_comp_matrix
        self.spectral_all_detectors = spectral_all_detectors
        self.spectral_true_detectors = spectral_true_detectors

    def test_spectral_matrix_create(self):
        spec_matrix = fk.SpectralMatrix(
            self.spectral_comp_matrix,
            detectors=self.spectral_all_detectors,
            true_detectors=self.spectral_true_detectors
        )
        self.assertIsInstance(spec_matrix, fk.SpectralMatrix)

    def test_spectral_matrix_equal(self):
        spec_matrix = fk.SpectralMatrix(
            self.spectral_comp_matrix,
            detectors=self.spectral_all_detectors,
            true_detectors=self.spectral_true_detectors
        )
        spec_matrix2 = copy.deepcopy(spec_matrix)

        self.assertEqual(spec_matrix, spec_matrix2)

    def test_spectral_matrix_not_equal(self):
        spec_matrix = fk.SpectralMatrix(
            self.spectral_comp_matrix,
            detectors=self.spectral_all_detectors,
            true_detectors=self.spectral_true_detectors
        )
        spec_matrix2 = copy.deepcopy(spec_matrix)
        spec_matrix2.matrix += 0.00001

        self.assertNotEqual(spec_matrix, spec_matrix2)

    def test_spectral_matrix_as_dataframe(self):
        spec_matrix = fk.SpectralMatrix(
            self.spectral_comp_matrix,
            detectors=self.spectral_all_detectors,
            true_detectors=self.spectral_true_detectors
        )
        df_spec_matrix = spec_matrix.as_dataframe()

        self.assertIsInstance(df_spec_matrix, pd.DataFrame)
        self.assertListEqual(list(df_spec_matrix.columns), spec_matrix.detectors)
        self.assertListEqual(list(df_spec_matrix.index), spec_matrix.true_detectors)

    def test_spectral_matrix_apply(self):
        spec_matrix = fk.SpectralMatrix(
            self.spectral_comp_matrix,
            detectors=self.spectral_all_detectors,
            true_detectors=self.spectral_true_detectors
        )
        comp_events = spec_matrix.apply(spectral_sample)

        # Compare only the fluoro channel events
        # The spectral Sample was created from a npy array and
        # went through the FlowIO conversion process so the events
        # are slightly different.
        np.testing.assert_array_almost_equal(
            spectral_truth_comp_events[:, spectral_sample.fluoro_indices],
            comp_events[:, spectral_sample.fluoro_indices],
            decimal=7
        )
