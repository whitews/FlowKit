import unittest
import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('..'))

from flowkit import Sample, GatingStrategy, Dimension, gates

data1_fcs_path = 'examples/gate_ref/data1.fcs'
data1_sample = Sample(data1_fcs_path)


class GatingTestCase(unittest.TestCase):
    @staticmethod
    def test_add_rect1_gate():
        gs = GatingStrategy()

        dim1 = Dimension("SSC-H", compensation_ref="uncompensated", range_min=20, range_max=80)
        dim2 = Dimension("FL1-H", compensation_ref="uncompensated", range_min=70, range_max=200)
        dims = [dim1, dim2]

        rect_gate = gates.RectangleGate("Rectangle1", None, dims, gs)
        gs.add_gate(rect_gate)

        res_path = 'examples/gate_ref/truth/Results_Rectangle1.txt'
        truth = pd.read_csv(res_path, header=None, squeeze=True, dtype='bool').values

        result = gs.gate_sample(data1_sample, 'Rectangle1')

        np.testing.assert_array_equal(truth, result.get_gate_indices('Rectangle1'))
