"""
Unit tests for plotting functions
"""
import copy
import unittest

import bokeh.models
import numpy as np
from bokeh.plotting import figure as bk_Figure
from bokeh.layouts import GridPlot as bk_GridPlot
import flowkit as fk

from tests.test_config import test_sample, gml_path


test_gating_strategy = fk.parse_gating_xml(gml_path)


class PlotTestCase(unittest.TestCase):
    """
    Tests for plot functions/methods

    NOTE: Due to the difficulty of introspecting figures and images at a
          pixel-level, this TestCase only tests that plots are returned
          from plotting functions.
    """
    def test_plot_scatter_zero_points(self):
        # from issue #197
        arr = np.array([], float)
        # noinspection PyProtectedMember
        p = fk._utils.plot_utils.plot_scatter(arr, arr)

        self.assertIsInstance(p, bk_Figure)

    def test_plot_scatter_one_point(self):
        # from issue #197
        arr = np.array([1., ], float)
        # noinspection PyProtectedMember
        p = fk._utils.plot_utils.plot_scatter(arr, arr)

        self.assertIsInstance(p, bk_Figure)

    def test_plot_scatter_two_points_with_extents(self):
        # from issue #197
        # noinspection PyProtectedMember
        p = fk._utils.plot_utils.plot_scatter(
            np.array([0.44592386, 0.52033713]),
            np.array([0.6131338, 0.60149982]),
            x_min=0, x_max=.997, y_min=0, y_max=.991
        )

        self.assertIsInstance(p, bk_Figure)

    def test_sample_plot_histogram(self):
        sample = copy.deepcopy(test_sample)
        xform_logicle = fk.transforms.LogicleTransform(param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
        sample.apply_transform(xform_logicle)

        p = sample.plot_histogram(
            'FL2-H',
            source='xform',
            subsample=True,
            data_min=0.2,
            data_max=0.5,
            x_range=(0, 1)
        )

        self.assertIsInstance(p, bk_Figure)

    def test_sample_plot_channel(self):
        sample = copy.deepcopy(test_sample)
        xform_logicle = fk.transforms.LogicleTransform(param_t=1024, param_w=0.5, param_m=4.5, param_a=0)
        sample.apply_transform(xform_logicle)

        flagged_events = list(range(1000))
        flagged_events.extend(list(range(8000, 9000)))
        sample.set_flagged_events(flagged_events)

        fig = sample.plot_channel(
            'FSC-H',
            source='xform'
        )

        self.assertIsInstance(fig, bk_Figure)

    def test_sample_plot_contour(self):
        sample = copy.deepcopy(test_sample)
        xform_logicle = fk.transforms.LogicleTransform(param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
        sample.apply_transform(xform_logicle)

        p = sample.plot_contour(
            'FL1-H',
            'FL2-H',
            source='xform',
            plot_events=True,
            subsample=True
        )

        self.assertIsInstance(p, bk_Figure)

    def test_sample_plot_scatter(self):
        sample = copy.deepcopy(test_sample)
        xform_logicle = fk.transforms.LogicleTransform(param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
        sample.apply_transform(xform_logicle)

        p = sample.plot_scatter(
            'FL1-H',
            'FL2-H',
            source='xform',
            subsample=True
        )

        self.assertIsInstance(p, bk_Figure)

    def test_sample_plot_scatter_matrix(self):
        sample = copy.deepcopy(test_sample)

        # reduce # of events for plotting performance
        sample.subsample_events(500)

        xform_logicle = fk.transforms.LogicleTransform(param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
        sample.apply_transform(xform_logicle)

        grid = sample.plot_scatter_matrix(
            ['FL1-H', 'FL2-H', 'FL3-H'],
            source='xform',
            subsample=True
        )

        self.assertIsInstance(grid, bk_GridPlot)

    def test_session_plot_gate(self):
        fks = fk.Session(
            gating_strategy=copy.deepcopy(test_gating_strategy),
            fcs_samples=copy.deepcopy(test_sample)
        )
        sample_name = 'B07'
        gate_tuples = fks.get_gate_ids()
        fks.analyze_samples(sample_id=sample_name)

        for gate_name, ancestors in gate_tuples:
            try:
                gate = fks.get_gate(gate_name, sample_id=sample_name)
            except fk.exceptions.QuadrantReferenceError:
                # cannot plot single quadrants of a quadrant gate
                continue

            if isinstance(gate, fk.gates.BooleanGate):
                # can't plot Boolean gates
                continue

            try:
                p = fks.plot_gate(sample_name, gate_name)
            except NotImplementedError:
                continue

            self.assertIsInstance(p, bk_Figure)

    def test_session_plot_scatter(self):
        fks = fk.Session(
            gating_strategy=copy.deepcopy(test_gating_strategy),
            fcs_samples=copy.deepcopy(test_sample)
        )
        sample_name = 'B07'
        fks.analyze_samples(sample_id=sample_name)

        x_dim = fk.Dimension('FL2-H', compensation_ref='MySpill', transformation_ref='Logicle_10000_0.5_4.5_0')
        y_dim = fk.Dimension('FL3-H', compensation_ref='MySpill', transformation_ref='Logicle_10000_0.5_4.5_0')

        p = fks.plot_scatter(
            sample_name,
            x_dim,
            y_dim,
            gate_name='ScaleRect1'
        )

        self.assertIsInstance(p, bk_Figure)

    def test_workspace_plot_gate(self):
        wsp_path = "data/simple_line_example/simple_poly_and_rect.wsp"
        simple_line_fcs_path = "data/simple_line_example/data_set_simple_line_100.fcs"
        sample_id = 'data_set_simple_line_100.fcs'

        wsp = fk.Workspace(wsp_path, fcs_samples=simple_line_fcs_path)
        wsp.analyze_samples()

        p = wsp.plot_gate(
            sample_id,
            gate_name='poly1'
        )

        self.assertIsInstance(p, bk_Figure)

    def test_workspace_plot_scatter(self):
        wsp_path = "data/simple_line_example/single_ellipse_51_events.wsp"
        simple_line_fcs_path = "data/simple_line_example/data_set_simple_line_100.fcs"
        sample_id = 'data_set_simple_line_100.fcs'

        wsp = fk.Workspace(wsp_path, fcs_samples=simple_line_fcs_path)
        wsp.analyze_samples()

        p = wsp.plot_scatter(
            sample_id,
            'channel_A',
            'channel_B',
            gate_name='ellipse1'
        )

        self.assertIsInstance(p, bk_Figure)

    def test_workspace_plot_scatter_no_events_raises(self):
        wsp_path = "data/simple_line_example/simple_poly_and_rect.wsp"
        simple_line_fcs_path = "data/simple_line_example/data_set_simple_line_100.fcs"
        sample_id = 'data_set_simple_line_100.fcs'

        wsp = fk.Workspace(wsp_path, fcs_samples=simple_line_fcs_path)
        wsp.analyze_samples()

        self.assertRaises(
            fk.exceptions.FlowKitException,
            wsp.plot_scatter,
            sample_id,
            'channel_A',
            'channel_B',
            gate_name='rect1'
        )
