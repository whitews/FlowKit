"""
Unit tests for plotting functions
"""
import copy
import unittest
from bokeh.plotting.figure import Figure as bk_Figure
from bokeh.layouts import Column as bk_Column
from matplotlib.pyplot import Figure as mpl_Figure
import flowkit as fk

fcs_path = 'data/gate_ref/data1.fcs'
gml_path = 'data/gate_ref/gml/gml_all_gates.xml'
test_sample = fk.Sample(fcs_path, subsample=2000)
test_gating_strategy = fk.parse_gating_xml(gml_path)


class PlotTestCase(unittest.TestCase):
    """
    Tests for plot functions/methods

    NOTE: Due to the difficulty of introspecting figures and images at a
          pixel-level, this TestCase only tests that plots are returned
          from plotting functions.
    """

    def test_plot_sample_histogram(self):
        sample = copy.deepcopy(test_sample)
        xform_logicle = fk.transforms.LogicleTransform('logicle', param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
        sample.apply_transform(xform_logicle)

        p = sample.plot_histogram(
            'FL2-H',
            source='xform',
            subsample=True
        )

        self.assertIsInstance(p, bk_Figure)

    def test_sample_plot_channel(self):
        sample = copy.deepcopy(test_sample)
        xform_logicle = fk.transforms.LogicleTransform('logicle', param_t=1024, param_w=0.5, param_m=4.5, param_a=0)
        sample.apply_transform(xform_logicle)

        flagged_events = list(range(1000))
        flagged_events.extend(list(range(8000, 9000)))
        sample.set_flagged_events(flagged_events)

        fig = sample.plot_channel(
            'FSC-H',
            source='xform',
            flag_events=True
        )

        self.assertIsInstance(fig, mpl_Figure)

    def test_plot_sample_contour(self):
        sample = copy.deepcopy(test_sample)
        xform_logicle = fk.transforms.LogicleTransform('logicle', param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
        sample.apply_transform(xform_logicle)

        p = sample.plot_contour(
            'FL1-H',
            'FL2-H',
            source='xform',
            plot_events=True,
            subsample=True
        )

        self.assertIsInstance(p, mpl_Figure)

    def test_plot_sample_scatter(self):
        sample = copy.deepcopy(test_sample)
        xform_logicle = fk.transforms.LogicleTransform('logicle', param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
        sample.apply_transform(xform_logicle)

        p = sample.plot_scatter(
            'FL1-H',
            'FL2-H',
            source='xform',
            subsample=True
        )

        self.assertIsInstance(p, bk_Figure)

    def test_plot_sample_scatter_matrix(self):
        sample = copy.deepcopy(test_sample)

        # reduce # of events for plotting performance
        sample.subsample_events(500)

        xform_logicle = fk.transforms.LogicleTransform('logicle', param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
        sample.apply_transform(xform_logicle)

        grid = sample.plot_scatter_matrix(
            ['FL1-H', 'FL2-H', 'FL3-H'],
            source='xform',
            subsample=True
        )

        self.assertIsInstance(grid, bk_Column)

    def test_plot_gates(self):
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

    def test_plot_gated_scatter(self):
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
