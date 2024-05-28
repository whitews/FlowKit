"""
Matrix class tests
"""
import copy
import unittest
import numpy as np
import pandas as pd
import warnings
import flowkit as fk

fcs_spill = '13,B515-A,R780-A,R710-A,R660-A,V800-A,V655-A,V585-A,V450-A,G780-A,G710-A,G660-A,G610-A,G560-A,'\
    '1,0,0,0.00008841570561316703,0.0002494559842740046,0.0006451591561972469,0.007198401782797728,0,0,'\
    '0.00013132619816952714,0.0000665251222573374,0.0005815839652764308,0.0025201730479353047,0,1,'\
    '0.07118758880093266,0.14844804153215532,0.3389031912802132,0.00971660311243448,0,0,0.3013801753249257,'\
    '0.007477611134717788,0.0123543122066652,0,0,0,0.33140488468849205,1,0.0619647566095391,0.12097867005182314,'\
    '0.004052554840959644,0,0,0.1091165124197372,0.10031383324016652,0.005831773047424356,0,0,0,'\
    '0.08862108746390694,0.38942413967608824,1,0.029758767352535288,0.06555281586224636,0,0,0.03129393154653089,'\
    '0.039305936245674286,0.09137451237674046,0.00039591122341817164,0.000056659766405160846,0,'\
    '0.13661791418865094,0.010757316236957385,0,1,0.00015647113960278087,0,0,0.48323487842103036,'\
    '0.01485813345103798,0,0,0,0,0.00012365104122837034,0.019462610460087203,0.2182062762553545,'\
    '0.004953221988365214,1,0.003582785726251024,0,0.0013106968243292993,0.029645575685206288,0.4089015923558522,'\
    '0.006505826616588717,0.00011917703878954761,0,0,0,0,0.001055595075733903,0.002287122431059274,1,0,'\
    '0.0003885172922042414,0.0001942589956485108,0,0.06255131165904257,0.13248446095817049,0,0,0,0,0,'\
    '0.008117870042687002,0.17006643956891296,1,0,0,0,0,0,0.003122390646560834,0.008525685683831916,'\
    '0.001024237027323255,0.0011626412849951272,0.12540105131097395,0.018142202256893485,0.19364562239714117,'\
    '0,1,0.06689784643460173,0.16145640353506588,0.2868231743828476,1.2380368696528024,0.002015498041918758,'\
    '0.06964529385206036,0.19471548842271394,0.0010077719457714136,0.15161117217667686,0.0012703511702660231,'\
    '0.007133491446011225,0,1.1500323358669722,1,0.016076827046672983,0.014674146885307975,0.055351746085494,'\
    '0.001685226072130514,0.05433993817875603,0.27785224557295884,0.34300826602551504,0.06175281041168121,'\
    '0.07752283973796613,0.0042628794531131406,0,0.49748791920091034,0.7439226300384197,1,0.010329232815907474,'\
    '0.03763461149817695,0,0.008713148000954844,0.04821275078920058,0.07319044343609345,0.1505631929508567,'\
    '0.3862934410767249,0.10189631814602482,0,0.3702770755789083,0.6134900271606913,1.2180240147472128,1,'\
    '0.06521131251063482,0.0016842378874079343,0,0,0.00009533732312150527,0.0034630076700367675,'\
    '0.01571183587491517,0.17412189188164517,0,0.023802192010810994,0.049474451704249904,0.13251071256825273,'\
    '0.23921555785727822,1'

fcs_spill_header = [
    'B515-A', 'R780-A', 'R710-A', 'R660-A',
    'V800-A', 'V655-A', 'V585-A', 'V450-A',
    'G780-A', 'G710-A', 'G660-A', 'G610-A',
    'G560-A'
]

csv_8c_comp_file_path = 'data/8_color_data_set/den_comp.csv'
detectors_8c = [
    'TNFa FITC FLR-A',
    'CD8 PerCP-Cy55 FLR-A',
    'IL2 BV421 FLR-A',
    'Aqua Amine FLR-A',
    'IFNg APC FLR-A',
    'CD3 APC-H7 FLR-A',
    'CD107a PE FLR-A',
    'CD4 PE-Cy7 FLR-A'
]
fluorochromes_8c = [
    'TNFa',
    'CD8',
    'IL2',
    'Aqua Amine',
    'IFNg',
    'CD3',
    'CD107a',
    'CD4'
]

csv_8c_comp_null_channel_file_path = 'data/8_color_data_set/den_comp_null_channel.csv'


class MatrixTestCase(unittest.TestCase):
    """Tests related to compensation matrices and the Matrix class"""
    def test_matrix_from_fcs_spill(self):
        comp_mat = fk.Matrix('my_spill', fcs_spill, fcs_spill_header)

        self.assertIsInstance(comp_mat, fk.Matrix)

    def test_parse_csv_file(self):
        comp_mat = fk.Matrix(
            'my_spill',
            csv_8c_comp_file_path,
            detectors_8c
        )

        self.assertIsInstance(comp_mat, fk.Matrix)

    def test_matrix_equals(self):
        comp_mat = fk.Matrix(
            'my_spill',
            csv_8c_comp_file_path,
            detectors_8c
        )

        comp_mat2 = copy.deepcopy(comp_mat)

        self.assertEqual(comp_mat, comp_mat2)

    def test_matrix_equals_fails(self):
        comp_mat = fk.Matrix(
            'my_spill',
            csv_8c_comp_file_path,
            detectors_8c
        )

        # copy & modify matrix array
        comp_mat2 = copy.deepcopy(comp_mat)
        comp_mat2.matrix[0, 1] = comp_mat2.matrix[0, 1] + 0.01

        self.assertNotEqual(comp_mat, comp_mat2)

    def test_matrix_as_dataframe(self):
        comp_mat = fk.Matrix(
            'my_spill',
            csv_8c_comp_file_path,
            detectors_8c,
            fluorochromes=fluorochromes_8c
        )

        # test with detectors as labels
        comp_df_detectors = comp_mat.as_dataframe()

        # test with fluorochromes as labels
        comp_df_fluorochromes = comp_mat.as_dataframe(fluoro_labels=True)

        self.assertIsInstance(comp_df_detectors, pd.DataFrame)
        self.assertIsInstance(comp_df_fluorochromes, pd.DataFrame)

    def test_reserved_matrix_id_uncompensated(self):
        self.assertRaises(ValueError, fk.Matrix, 'uncompensated', fcs_spill, fcs_spill_header)

    @staticmethod
    def test_matrix_inverse():
        fcs_file_path = "data/test_comp_example.fcs"
        comp_file_path = "data/comp_complete_example.csv"

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sample = fk.Sample(
                fcs_path_or_data=fcs_file_path,
                compensation=comp_file_path,
                ignore_offset_error=True  # sample has off by 1 data offset
            )
        matrix = sample.compensation

        data_raw = sample.get_events(source='raw')
        inv_data = matrix.inverse(sample)

        np.testing.assert_almost_equal(inv_data, data_raw, 10)

    def test_null_channels(self):
        # pretend FITC is a null channel
        null_channels = ['TNFa FITC FLR-A']

        comp_mat = fk.Matrix(
            'my_spill',
            csv_8c_comp_null_channel_file_path,
            detectors_8c,
            null_channels=null_channels
        )

        fcs_file_path = "data/8_color_data_set/fcs_files/101_DEN084Y5_15_E01_008_clean.fcs"

        # test with a sample not using null channels and one using null channels
        sample1 = fk.Sample(fcs_file_path, null_channel_list=None)
        sample2 = fk.Sample(fcs_file_path, null_channel_list=null_channels)

        comp_events1 = comp_mat.apply(sample1)
        comp_events2 = comp_mat.apply(sample2)

        self.assertIsInstance(comp_events1, np.ndarray)
        self.assertIsInstance(comp_events2, np.ndarray)
