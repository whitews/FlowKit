"""
Matrix class tests
"""

import copy
import io
import tempfile
import unittest
from pathlib import Path
import flowio
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


def _create_synthetic_fcs_sample(
        events,
        channel_names,
        metadata_dict=None,
        null_channel_list=None,
        compensation=None,
        use_flowjo_labels=False
):
    buf = io.BytesIO()
    flowio.create_fcs(
        buf,
        events.flatten().tolist(),
        channel_names=channel_names,
        metadata_dict=metadata_dict
    )
    buf.seek(0)
    return fk.Sample(
        buf,
        null_channel_list=null_channel_list,
        compensation=compensation,
        use_flowjo_labels=use_flowjo_labels
    )


class NumericSpilloverTestCase(unittest.TestCase):
    """Tests for non-standard 1-based numeric $SPILLOVER headers (e.g., Beckman Coulter Navios)."""

    def setUp(self):
        rng = np.random.default_rng(264)
        self.channels = ['FSC-A', 'FSC-H', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A', 'Time']
        self.detectors = ['FL1-A', 'FL2-A', 'FL3-A']
        self.events = rng.uniform(500.0, 20000.0, size=(60, len(self.channels))).astype(np.float32)
        self.spill_values = '1,0.12,0.02,0.05,1,0.08,0.01,0.04,1'
        self.navios_spill = f'3,4,5,6,{self.spill_values}'
        self.std_spill = f'3,FL1-A,FL2-A,FL3-A,{self.spill_values}'

    def test_navios_numeric_spillover_sample_apply_compensation(self):
        s_navios = _create_synthetic_fcs_sample(
            self.events,
            self.channels,
            metadata_dict={'SPILLOVER': self.navios_spill}
        )
        s_std = _create_synthetic_fcs_sample(
            self.events,
            self.channels,
            metadata_dict={'SPILLOVER': self.std_spill}
        )

        s_navios.apply_compensation(s_navios.get_metadata()['spillover'])
        s_std.apply_compensation(s_std.get_metadata()['spillover'])

        self.assertEqual(s_navios.compensation, s_std.compensation)
        self.assertEqual(s_navios.compensation.detectors, self.detectors)
        pd.testing.assert_frame_equal(
            s_navios.compensation.as_dataframe(),
            s_std.compensation.as_dataframe()
        )
        np.testing.assert_array_equal(
            s_navios.get_events(source='comp'),
            s_std.get_events(source='comp')
        )

        # Also verify Sample(..., compensation=...) constructor, file path str, and pathlib.Path
        s_ctor = _create_synthetic_fcs_sample(
            self.events,
            self.channels,
            compensation=self.navios_spill
        )
        np.testing.assert_array_equal(
            s_ctor.get_events(source='comp'),
            s_std.get_events(source='comp')
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            spill_path = Path(tmpdir) / 'navios_spill.csv'
            spill_path.write_text(self.navios_spill + '\n')

            s_path = _create_synthetic_fcs_sample(self.events, self.channels)
            s_path.apply_compensation(spill_path)
            np.testing.assert_array_equal(
                s_path.get_events(source='comp'),
                s_std.get_events(source='comp')
            )

            s_str_path = _create_synthetic_fcs_sample(self.events, self.channels)
            s_str_path.apply_compensation(str(spill_path))
            np.testing.assert_array_equal(
                s_str_path.get_events(source='comp'),
                s_std.get_events(source='comp')
            )

    def test_navios_numeric_spillover_reordered_channels(self):
        # Channels given in permuted order: 6 (FL3-A), 4 (FL1-A), 5 (FL2-A)
        perm_values = '1,0.01,0.04,0.02,1,0.12,0.08,0.05,1'
        navios_perm = f'3,6,4,5,{perm_values}'
        std_perm = f'3,FL3-A,FL1-A,FL2-A,{perm_values}'

        s_navios = _create_synthetic_fcs_sample(self.events, self.channels)
        s_std = _create_synthetic_fcs_sample(self.events, self.channels)

        s_navios.apply_compensation(navios_perm)
        s_std.apply_compensation(std_perm)

        self.assertEqual(s_navios.compensation, s_std.compensation)
        np.testing.assert_array_equal(
            s_navios.get_events(source='comp'),
            s_std.get_events(source='comp')
        )

    def test_navios_numeric_spillover_with_null_channels(self):
        # Typical Beckman Coulter Navios LMD channel names: FS, SS, FL1..FL3, TIME
        navios_channels = ['FS', 'SS', 'FL1', 'FL2', 'FL3', 'TIME']
        rng = np.random.default_rng(265)
        events = rng.uniform(500.0, 20000.0, size=(40, len(navios_channels))).astype(np.float32)
        navios_spill = f'3,3,4,5,{self.spill_values}'
        std_spill = f'3,FL1,FL2,FL3,{self.spill_values}'

        s_navios = _create_synthetic_fcs_sample(
            events,
            navios_channels,
            metadata_dict={'SPILLOVER': navios_spill},
            null_channel_list=['FS', 'SS']
        )
        s_std = _create_synthetic_fcs_sample(
            events,
            navios_channels,
            metadata_dict={'SPILLOVER': std_spill},
            null_channel_list=['FS', 'SS']
        )

        s_navios.apply_compensation(s_navios.get_metadata()['spillover'])
        s_std.apply_compensation(s_std.get_metadata()['spillover'])

        self.assertEqual(s_navios.compensation, s_std.compensation)
        np.testing.assert_array_equal(
            s_navios.get_events(source='comp'),
            s_std.get_events(source='comp')
        )

        # Without declaring FS/SS as null channels, resolved labels ['FL1','FL2','FL3']
        # do not match sample fluoro labels ['FS','SS','FL1','FL2','FL3'] -> must raise ValueError
        s_missing_null = _create_synthetic_fcs_sample(
            events,
            navios_channels,
            metadata_dict={'SPILLOVER': navios_spill}
        )
        with self.assertRaises(ValueError) as ctx:
            s_missing_null.apply_compensation(s_missing_null.get_metadata()['spillover'])
        self.assertIn('FS', str(ctx.exception))
        self.assertIn('SS', str(ctx.exception))

    def test_navios_numeric_spillover_gating_strategy_and_session_fcs_ref(self):
        for meta_key in ('SPILLOVER', 'SPILL'):
            s_navios = _create_synthetic_fcs_sample(
                self.events,
                self.channels,
                metadata_dict={meta_key: self.navios_spill, 'FIL': 'navios.fcs'}
            )
            s_std = _create_synthetic_fcs_sample(
                self.events,
                self.channels,
                metadata_dict={meta_key: self.std_spill, 'FIL': 'std.fcs'}
            )

            gs = fk.GatingStrategy()
            dim_x = fk.Dimension('FL1-A', compensation_ref='FCS', range_min=5000.0)
            dim_y = fk.Dimension('FL2-A', compensation_ref='FCS', range_min=5000.0)
            gs.add_gate(fk.gates.RectangleGate('gate1', [dim_x, dim_y]), gate_path=('root',))

            res_navios = gs.gate_sample(s_navios)
            res_std = gs.gate_sample(s_std)

            np.testing.assert_array_equal(
                res_navios.get_gate_membership('gate1'),
                res_std.get_gate_membership('gate1')
            )

            # Verify Session workflow with compensation_ref='FCS'
            sess = fk.Session(fcs_samples=[s_navios, s_std])
            sess.add_gate(fk.gates.RectangleGate('gate1', [dim_x, dim_y]), gate_path=('root',))
            sess.analyze_samples()
            np.testing.assert_array_equal(
                sess.get_gate_membership('navios.fcs', 'gate1'),
                sess.get_gate_membership('std.fcs', 'gate1')
            )

    def test_matrix_direct_numeric_spillover(self):
        mat_std = fk.Matrix(self.std_spill, self.detectors)

        # With explicit full channel_labels (including scatter/time)
        mat_navios = fk.Matrix(
            self.navios_spill,
            self.detectors,
            channel_labels=self.channels
        )
        self.assertEqual(mat_navios, mat_std)

        # Without channel_labels, numeric headers MUST raise ValueError rather than
        # guessing physical FCS channel positions from detectors order
        with self.assertRaises(ValueError):
            fk.Matrix(f'3,1,2,3,{self.spill_values}', self.detectors)

        with self.assertRaises(ValueError):
            fk.Matrix(self.navios_spill, self.detectors)

        # Witness check (C03): physical ['A', 'B'], detectors in matrix-column order ['B', 'A'],
        # raw event [10, 9] with 0.5 spillover from A -> B => exact compensated [10, 4]
        witness_events = np.array([[10.0, 9.0]], dtype=np.float32)
        witness_sample = _create_synthetic_fcs_sample(witness_events, ['A', 'B'])
        numeric_ba = '2,2,1,1,0,0.5,1'
        named_ba = '2,B,A,1,0,0.5,1'

        with self.assertRaises(ValueError):
            fk.Matrix(numeric_ba, ['B', 'A'])

        mat_ba_named = fk.Matrix(named_ba, ['B', 'A'])
        mat_ba_num = fk.Matrix(numeric_ba, ['B', 'A'], channel_labels=['A', 'B'])
        self.assertEqual(mat_ba_num, mat_ba_named)

        comp_named = mat_ba_named.apply(witness_sample)
        comp_num = mat_ba_num.apply(witness_sample)
        np.testing.assert_array_equal(comp_num, comp_named)
        np.testing.assert_allclose(comp_num, np.array([[10.0, 4.0]]), atol=1e-9)

        # With null_channels and channel_labels
        mat_null = fk.Matrix(
            '2,4,6,1,0.12,0.01,1',
            self.detectors,
            null_channels=['FL2-A'],
            channel_labels=self.channels
        )
        mat_null_std = fk.Matrix(
            '2,FL1-A,FL3-A,1,0.12,0.01,1',
            self.detectors,
            null_channels=['FL2-A']
        )
        self.assertEqual(mat_null, mat_null_std)

    def test_numeric_pnn_labels_exact_match_takes_precedence(self):
        # Case 1a: Numeric $PnN labels ('1', '2', '3') after scatter channels
        channels_num = ['FSC-A', 'SSC-A', '1', '2', '3']
        rng = np.random.default_rng(266)
        events = rng.uniform(500.0, 20000.0, size=(40, len(channels_num))).astype(np.float32)

        s_exact = _create_synthetic_fcs_sample(events, channels_num)
        s_exact.apply_compensation(f'3,1,2,3,{self.spill_values}')
        self.assertEqual(s_exact.compensation.detectors, ['1', '2', '3'])

        # Header '3,4,5' overlaps existing $PnN label '3'; must refuse positional fallback
        s_overlap = _create_synthetic_fcs_sample(events, channels_num)
        with self.assertRaises(ValueError):
            s_overlap.apply_compensation(f'3,3,4,5,{self.spill_values}')

        # Case 1b: Numeric $PnN labels disjoint from 1-based indices ('10', '20', '30')
        channels_disjoint_num = ['FSC-A', 'SSC-A', '10', '20', '30']
        s_disjoint_exact = _create_synthetic_fcs_sample(events, channels_disjoint_num)
        s_disjoint_exact.apply_compensation(f'3,10,20,30,{self.spill_values}')
        s_disjoint_idx = _create_synthetic_fcs_sample(events, channels_disjoint_num)
        s_disjoint_idx.apply_compensation(f'3,3,4,5,{self.spill_values}')
        self.assertEqual(s_disjoint_idx.compensation, s_disjoint_exact.compensation)
        np.testing.assert_array_equal(
            s_disjoint_idx.get_events(source='comp'),
            s_disjoint_exact.get_events(source='comp')
        )

        # Case 2: Permuted numeric $PnN labels ['3', '1', '2'] where channel 1='3', 2='1', 3='2'
        # Exact $PnN match MUST win over 1-based index fallback!
        perm_channels = ['3', '1', '2']
        events_3 = rng.uniform(500.0, 20000.0, size=(30, 3)).astype(np.float32)
        s_perm = _create_synthetic_fcs_sample(events_3, perm_channels)
        # Spill header is '1,2,3' -> row 0 is '1', row 1 is '2', row 2 is '3'.
        # Reordered to detector order ['3', '1', '2'], row 0 must be old row 2 ('3').
        spill_text = f'3,1,2,3,{self.spill_values}'
        s_perm.apply_compensation(spill_text)
        expected_raw = np.array([
            [1.0, 0.12, 0.02],
            [0.05, 1.0, 0.08],
            [0.01, 0.04, 1.0],
        ])
        # Old order ['1', '2', '3'] -> detector order ['3', '1', '2'] = indices [2, 0, 1]
        expected_reordered = expected_raw[[2, 0, 1], :][:, [2, 0, 1]]
        np.testing.assert_array_equal(s_perm.compensation.matrix, expected_reordered)

        # Case 3: Ambiguous subset of existing numeric $PnN labels must NOT be reinterpreted as indices
        ambig_channels = ['4', '1', '2', '3']
        events_4 = rng.uniform(500.0, 20000.0, size=(20, 4)).astype(np.float32)
        s_ambig = _create_synthetic_fcs_sample(
            events_4,
            ambig_channels,
            null_channel_list=['4']
        )
        # '2', '3', '4' are all actual $PnN labels in the file (including null channel '4');
        # must raise ValueError rather than reinterpreting 2->'1', 3->'2', 4->'3'.
        with self.assertRaises(ValueError):
            s_ambig.apply_compensation(f'3,2,3,4,{self.spill_values}')

    def test_numeric_spillover_invalid_headers_raise_value_error(self):
        sample = _create_synthetic_fcs_sample(self.events, self.channels)

        invalid_spills = [
            f'3,0,5,6,{self.spill_values}',          # 0 (out of 1-based range)
            f'3,4,5,8,{self.spill_values}',          # > num_channels (7)
            f'3,-1,5,6,{self.spill_values}',         # negative index
            f'3,4.0,5,6,{self.spill_values}',        # float string
            f'3,4.5,5,6,{self.spill_values}',        # non-integer float
            f'3,04,5,6,{self.spill_values}',         # leading zero
            f'3,+4,5,6,{self.spill_values}',         # leading plus sign
            f'3,\u00b2,5,6,{self.spill_values}',     # Unicode superscript digit
            f'3,4,4,6,{self.spill_values}',          # duplicate index
            f'3,FL1-A,5,6,{self.spill_values}',      # mixed $PnN and numeric
            f'3,4,5,UNKNOWN,{self.spill_values}',    # mixed numeric and unknown
            f'3,1,2,3,{self.spill_values}',          # scatter channel indices
            f'3,FL1-A,FL2-A,BAD,{self.spill_values}',  # non-numeric unknown label
            '3,4,5,6,1,0,0',                         # truncated values
            '0',                                     # zero channels
            '-1,4,5',                                # negative count
        ]

        for bad_spill in invalid_spills:
            with self.subTest(spill=bad_spill), self.assertRaises(ValueError):
                sample.apply_compensation(bad_spill)

    def test_navios_10_color_multi_digit_channel_numbers(self):
        # 10-color Navios layout: FS, SS, FL1..FL10, TIME (channels 3..12 include multi-digit indices)
        channels_10c = ['FS', 'SS'] + [f'FL{i}' for i in range(1, 11)] + ['TIME']
        rng = np.random.default_rng(267)
        events = rng.uniform(500.0, 20000.0, size=(30, len(channels_10c))).astype(np.float32)
        mat_vals = np.eye(10)
        mat_vals[0, 1] = 0.05
        mat_vals[8, 9] = 0.07
        flat_vals = ','.join(f'{v:g}' for v in mat_vals.ravel())
        navios_10c = '10,' + ','.join(str(i) for i in range(3, 13)) + ',' + flat_vals
        std_10c = '10,' + ','.join(f'FL{i}' for i in range(1, 11)) + ',' + flat_vals

        s_navios = _create_synthetic_fcs_sample(
            events,
            channels_10c,
            metadata_dict={'SPILLOVER': navios_10c},
            null_channel_list=['FS', 'SS']
        )
        s_std = _create_synthetic_fcs_sample(
            events,
            channels_10c,
            metadata_dict={'SPILLOVER': std_10c},
            null_channel_list=['FS', 'SS']
        )

        s_navios.apply_compensation(s_navios.get_metadata()['spillover'])
        s_std.apply_compensation(s_std.get_metadata()['spillover'])

        self.assertEqual(s_navios.compensation, s_std.compensation)
        np.testing.assert_array_equal(
            s_navios.get_events(source='comp'),
            s_std.get_events(source='comp')
        )

    def test_navios_numeric_spillover_flowjo_labels_and_rename(self):
        slash_channels = ['FSC-A', 'FSC-H', 'SSC-A', 'FL1/A', 'FL2/A', 'FL3/A', 'Time']
        s_fj = _create_synthetic_fcs_sample(
            self.events,
            slash_channels,
            metadata_dict={'SPILLOVER': self.navios_spill},
            use_flowjo_labels=True
        )
        s_fj.apply_compensation(s_fj.get_metadata()['spillover'])
        self.assertEqual(s_fj.compensation.detectors, ['FL1_A', 'FL2_A', 'FL3_A'])

        s_renamed = _create_synthetic_fcs_sample(
            self.events,
            self.channels,
            metadata_dict={'SPILLOVER': self.navios_spill}
        )
        s_renamed.rename_channel('FL1-A', 'CD4-A')
        s_renamed.apply_compensation(s_renamed.get_metadata()['spillover'])
        self.assertEqual(s_renamed.compensation.detectors, ['CD4-A', 'FL2-A', 'FL3-A'])

    def test_numeric_pnn_rename_channel_raises_value_error(self):
        # Regression test: renaming a numeric $PnN channel must NOT cause an existing
        # $PnN-labeled $SPILLOVER header to fall through to 1-based positional indexing
        # and silently swap channels.
        events_3 = np.array(
            [
                [10.3125, 16.25, 9.0],
                [20.0, 5.0, 12.0],
                [8.0, 14.0, 6.0],
            ],
            dtype=np.float32
        )
        spill_text = f'3,2,1,3,{self.spill_values}'

        for channel_order, rename_map, gate_dims in (
            (['2', '1', '3'], [('3', 'Red')], ('2', 'Red')),
            (['3', '1', '2'], [('3', 'Red')], ('Red', '1')),
            (
                ['3', '1', '2'],
                [('3', 'Red'), ('1', 'Green'), ('2', 'Blue')],
                ('Red', 'Green')
            ),
        ):
            with self.subTest(channel_order=channel_order, rename_map=rename_map):
                sample = _create_synthetic_fcs_sample(
                    events_3,
                    channel_order,
                    metadata_dict={'SPILLOVER': spill_text, 'FIL': 'num_rename.fcs'}
                )
                for old_lbl, new_lbl in rename_map:
                    sample.rename_channel(old_lbl, new_lbl)

                with self.assertRaises(ValueError):
                    sample.apply_compensation(sample.get_metadata()['spillover'])

                gs = fk.GatingStrategy()
                dim_x = fk.Dimension(gate_dims[0], compensation_ref='FCS', range_min=5.0)
                dim_y = fk.Dimension(gate_dims[1], compensation_ref='FCS', range_min=5.0)
                gs.add_gate(fk.gates.RectangleGate('gate1', [dim_x, dim_y]), gate_path=('root',))

                with self.assertRaises(ValueError):
                    gs.gate_sample(sample)

                sess = fk.Session(fcs_samples=[sample])
                sess.add_gate(fk.gates.RectangleGate('gate1', [dim_x, dim_y]), gate_path=('root',))
                with self.assertRaises(ValueError):
                    sess.analyze_samples()

    def test_ambiguous_numeric_null_channels_refuses_positional_fallback(self):
        # When declared null channels have numeric $PnN names ('3', '4') matching
        # positional indices of fluorescent channels ('A', 'B'), labeled headers
        # succeed while ambiguous numeric headers deliberately raise ValueError.
        channels = ['3', '4', 'A', 'B']
        rng = np.random.default_rng(268)
        events = rng.uniform(500.0, 20000.0, size=(30, len(channels))).astype(np.float32)
        spill_vals = '1,0.15,0.05,1'
        labeled_spill = f'2,B,A,{spill_vals}'
        positional_spill = f'2,4,3,{spill_vals}'

        s_labeled = _create_synthetic_fcs_sample(
            events,
            channels,
            metadata_dict={'SPILLOVER': labeled_spill, 'FIL': 'labeled.fcs'},
            null_channel_list=['3', '4']
        )
        s_labeled.apply_compensation(s_labeled.get_metadata()['spillover'])
        self.assertEqual(s_labeled.compensation.detectors, ['A', 'B'])

        dim_x = fk.Dimension('A', compensation_ref='FCS', range_min=5000.0)
        dim_y = fk.Dimension('B', compensation_ref='FCS', range_min=5000.0)
        gs_labeled = fk.GatingStrategy()
        gs_labeled.add_gate(fk.gates.RectangleGate('gate1', [dim_x, dim_y]), gate_path=('root',))
        res_labeled = gs_labeled.gate_sample(s_labeled)
        self.assertEqual(len(res_labeled.get_gate_membership('gate1')), len(events))

        sess_labeled = fk.Session(fcs_samples=[s_labeled])
        sess_labeled.add_gate(fk.gates.RectangleGate('gate1', [dim_x, dim_y]), gate_path=('root',))
        sess_labeled.analyze_samples()
        np.testing.assert_array_equal(
            sess_labeled.get_gate_membership('labeled.fcs', 'gate1'),
            res_labeled.get_gate_membership('gate1')
        )

        s_ambig = _create_synthetic_fcs_sample(
            events,
            channels,
            metadata_dict={'SPILLOVER': positional_spill, 'FIL': 'ambig.fcs'},
            null_channel_list=['3', '4']
        )
        with self.assertRaises(ValueError):
            s_ambig.apply_compensation(s_ambig.get_metadata()['spillover'])

        gs_ambig = fk.GatingStrategy()
        gs_ambig.add_gate(fk.gates.RectangleGate('gate1', [dim_x, dim_y]), gate_path=('root',))
        with self.assertRaises(ValueError):
            gs_ambig.gate_sample(s_ambig)

        sess_ambig = fk.Session(fcs_samples=[s_ambig])
        sess_ambig.add_gate(fk.gates.RectangleGate('gate1', [dim_x, dim_y]), gate_path=('root',))
        with self.assertRaises(ValueError):
            sess_ambig.analyze_samples()

        with self.assertRaises(ValueError):
            fk.Matrix(positional_spill, ['A', 'B'], null_channels=['3', '4'])

        # Absent declared null channel '2' on ['A', 'B', 'C'] must also refuse '3,3,2,1,...'
        # consistently across Sample, GatingStrategy, Session, and direct Matrix
        abc_events = rng.uniform(500.0, 20000.0, size=(20, 3)).astype(np.float32)
        abc_spill = f'3,3,2,1,{self.spill_values}'
        with self.assertRaises(ValueError):
            _create_synthetic_fcs_sample(
                abc_events,
                ['A', 'B', 'C'],
                null_channel_list=['2'],
                compensation=abc_spill
            )
        s_absent_null = _create_synthetic_fcs_sample(
            abc_events,
            ['A', 'B', 'C'],
            metadata_dict={'SPILLOVER': abc_spill, 'FIL': 'absent_null.fcs'},
            null_channel_list=['2']
        )
        with self.assertRaises(ValueError):
            s_absent_null.apply_compensation(s_absent_null.get_metadata()['spillover'])

        gs_absent = fk.GatingStrategy()
        gs_absent.add_gate(
            fk.gates.RectangleGate('gate1', [dim_x, dim_y]),
            gate_path=('root',)
        )
        with self.assertRaises(ValueError):
            gs_absent.gate_sample(s_absent_null)

        sess_absent = fk.Session(fcs_samples=[s_absent_null])
        sess_absent.add_gate(
            fk.gates.RectangleGate('gate1', [dim_x, dim_y]),
            gate_path=('root',)
        )
        with self.assertRaises(ValueError):
            sess_absent.analyze_samples()

    def test_renamed_scatter_aliasing_fluoro_name_raises_value_error(self):
        # Regression test (C01): renaming scatter channel 1 ('FSC-A') to match fluorescent
        # channel 2 ('A') must NEVER allow numeric header '2,3,1,...' (or ambiguous '2,3,2,...')
        # to pass across Sample, GatingStrategy, Session, or direct Matrix.
        events = np.array(
            [
                [100.0, 10.0, 9.0],
                [200.0, 20.0, 14.0],
                [50.0, 8.0, 12.0],
            ],
            dtype=np.float32
        )
        for spill_text in ('2,3,1,1,0,0.5,1', '2,3,2,1,0,0.5,1'):
            with self.subTest(spill=spill_text):
                sample = _create_synthetic_fcs_sample(
                    events,
                    ['FSC-A', 'A', 'B'],
                    metadata_dict={'SPILLOVER': spill_text, 'FIL': 'scatter_alias.fcs'}
                )
                sample.rename_channel('FSC-A', 'A')

                with self.assertRaises(ValueError):
                    sample.apply_compensation(sample.get_metadata()['spillover'])

                dim_b = fk.Dimension('B', compensation_ref='FCS', range_min=5.0)
                gs = fk.GatingStrategy()
                gs.add_gate(fk.gates.RectangleGate('gate_b', [dim_b]), gate_path=('root',))
                with self.assertRaises(ValueError):
                    gs.gate_sample(sample)

                sess = fk.Session(fcs_samples=[sample])
                sess.add_gate(fk.gates.RectangleGate('gate_b', [dim_b]), gate_path=('root',))
                with self.assertRaises(ValueError):
                    sess.analyze_samples()

                with self.assertRaises(ValueError):
                    fk.Matrix(
                        spill_text,
                        ['A', 'B'],
                        channel_labels=['A', 'A', 'B'],
                        fluoro_indices=[1, 2]
                    )

                with self.assertRaises(ValueError):
                    fk.Matrix(
                        spill_text,
                        ['A', 'B'],
                        channel_labels=['A', 'A', 'B']
                    )

    def test_flowjo_label_collision_raises_value_error(self):
        # Regression test (C01): with raw ['X/A', 'X_A', 'B'] and use_flowjo_labels=True,
        # '/' -> '_' normalization aliases channel 1 and channel 2 to 'X_A'. Numeric headers
        # must raise ValueError rather than silently compensating the wrong physical column.
        events = np.array(
            [
                [100.0, 10.0, 9.0],
                [2.0, 20.0, 18.0],
                [50.0, 12.0, 16.0],
                [10.0, 6.0, 15.0],
                [4.0, 14.0, 22.0],
            ],
            dtype=np.float32
        )
        for null_chan in ('X/A', 'X_A'):
            for spill_text in ('2,3,1,1,0,0.5,1', '2,3,2,1,0,0.5,1'):
                with self.subTest(null_chan=null_chan, spill=spill_text):
                    with self.assertRaises(ValueError):
                        _create_synthetic_fcs_sample(
                            events,
                            ['X/A', 'X_A', 'B'],
                            null_channel_list=[null_chan],
                            compensation=spill_text,
                            use_flowjo_labels=True
                        )

                    sample = _create_synthetic_fcs_sample(
                        events,
                        ['X/A', 'X_A', 'B'],
                        metadata_dict={'SPILLOVER': spill_text, 'FIL': 'fj_collision.fcs'},
                        null_channel_list=[null_chan],
                        use_flowjo_labels=True
                    )
                    with self.assertRaises(ValueError):
                        sample.apply_compensation(sample.get_metadata()['spillover'])

                    dim_b = fk.Dimension('B', compensation_ref='FCS', range_min=10.5)
                    gs = fk.GatingStrategy()
                    gs.add_gate(fk.gates.RectangleGate('gate_b', [dim_b]), gate_path=('root',))
                    with self.assertRaises(ValueError):
                        gs.gate_sample(sample)

                    sess = fk.Session(fcs_samples=[sample])
                    sess.add_gate(fk.gates.RectangleGate('gate_b', [dim_b]), gate_path=('root',))
                    with self.assertRaises(ValueError):
                        sess.analyze_samples()

        # Control: with use_flowjo_labels=False and null_channel_list=['X/A'],
        # physical channel labels are unique; header '2,3,2,...' succeeds and matches
        # named '2,B,X_A,...', while '2,3,1,...' (referencing null channel 1) raises ValueError.
        s_num = _create_synthetic_fcs_sample(
            events,
            ['X/A', 'X_A', 'B'],
            metadata_dict={'SPILLOVER': '2,3,2,1,0,0.5,1'},
            null_channel_list=['X/A'],
            use_flowjo_labels=False
        )
        s_named = _create_synthetic_fcs_sample(
            events,
            ['X/A', 'X_A', 'B'],
            metadata_dict={'SPILLOVER': '2,B,X_A,1,0,0.5,1'},
            null_channel_list=['X/A'],
            use_flowjo_labels=False
        )
        s_num.apply_compensation(s_num.get_metadata()['spillover'])
        s_named.apply_compensation(s_named.get_metadata()['spillover'])
        self.assertEqual(s_num.compensation, s_named.compensation)
        np.testing.assert_array_equal(
            s_num.get_events(source='comp'),
            s_named.get_events(source='comp')
        )

        s_bad_null_idx = _create_synthetic_fcs_sample(
            events,
            ['X/A', 'X_A', 'B'],
            null_channel_list=['X/A'],
            use_flowjo_labels=False
        )
        with self.assertRaises(ValueError):
            s_bad_null_idx.apply_compensation('2,3,1,1,0,0.5,1')

    def test_vacated_null_channel_name_reused_for_active_detector(self):
        # Regression test (r3 C04): declaring slot 1 ('Unused') as null, renaming
        # 'Unused' -> 'Off' and active 'A' -> 'Unused' leaves unique display names
        # ['Off', 'Unused', 'B'] with active detectors ['Unused', 'B'].
        # Both named ('2,Unused,B,...') and positional ('2,2,3,...') matrices must
        # succeed in Sample.apply_compensation and produce exact [40, 10, 4].
        events = np.array([[40.0, 10.0, 9.0]], dtype=np.float32)
        expected = np.array([[40.0, 10.0, 4.0]], dtype=np.float64)

        for named_spill, pos_spill in (
            ('2,Unused,B,1,0.5,0,1', '2,2,3,1,0.5,0,1'),
            ('2,B,Unused,1,0,0.5,1', '2,3,2,1,0,0.5,1'),
        ):
            with self.subTest(named=named_spill, positional=pos_spill):
                s_named = _create_synthetic_fcs_sample(
                    events,
                    ['Unused', 'A', 'B'],
                    null_channel_list=['Unused']
                )
                s_named.rename_channel('Unused', 'Off')
                s_named.rename_channel('A', 'Unused')
                s_named.apply_compensation(named_spill)

                s_pos = _create_synthetic_fcs_sample(
                    events,
                    ['Unused', 'A', 'B'],
                    null_channel_list=['Unused']
                )
                s_pos.rename_channel('Unused', 'Off')
                s_pos.rename_channel('A', 'Unused')
                s_pos.apply_compensation(pos_spill)

                self.assertEqual(s_pos.compensation, s_named.compensation)
                self.assertEqual(s_named.compensation.detectors, ['Unused', 'B'])
                np.testing.assert_allclose(
                    s_named.get_events(source='comp'),
                    expected,
                    atol=1e-9
                )
                np.testing.assert_array_equal(
                    s_pos.get_events(source='comp'),
                    s_named.get_events(source='comp')
                )

                # Referencing physical null slot 1 ('Off') must still raise ValueError
                with self.assertRaises(ValueError):
                    s_pos.apply_compensation('2,1,3,1,0.5,0,1')
