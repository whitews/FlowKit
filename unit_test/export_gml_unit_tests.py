import unittest
import sys
import os
from io import BytesIO
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('..'))

from flowkit import Sample, GatingStrategy

data1_fcs_path = 'examples/gate_ref/data1.fcs'
data1_sample = Sample(data1_fcs_path)


class ExportGMLTestCase(unittest.TestCase):
    @staticmethod
    def test_min_range_gate():
        gml_path = 'examples/gate_ref/gml/gml_range_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Range1.txt'

        gs = GatingStrategy(gml_path)

        out_file = BytesIO()

        gs.export_gml(out_file)
        out_file.seek(0)

        gs_out = GatingStrategy(out_file)
        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        result = gs_out.gate_sample(data1_sample, 'Range1')

        np.testing.assert_array_equal(truth, result.get_gate_indices('Range1'))

    @staticmethod
    def test_rect1_gate():
        gml_path = 'examples/gate_ref/gml/gml_rect1_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Rectangle1.txt'

        gs = GatingStrategy(gml_path)

        out_file = BytesIO()

        gs.export_gml(out_file)
        out_file.seek(0)

        gs_out = GatingStrategy(out_file)

        result = gs_out.gate_sample(data1_sample, 'Rectangle1')
        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        np.testing.assert_array_equal(truth, result.get_gate_indices('Rectangle1'))

    @staticmethod
    def test_rect2_gate():
        gml_path = 'examples/gate_ref/gml/gml_rect2_gate.xml'
        res_path = 'examples/gate_ref/truth/Results_Rectangle2.txt'

        gs = GatingStrategy(gml_path)

        out_file = BytesIO()

        gs.export_gml(out_file)
        out_file.seek(0)

        gs_out = GatingStrategy(out_file)

        result = gs_out.gate_sample(data1_sample, 'Rectangle2')
        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        np.testing.assert_array_equal(truth, result.get_gate_indices('Rectangle2'))
