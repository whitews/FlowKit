"""
Unit tests for plotting functions
"""
import unittest
from bokeh.plotting.figure import Figure as bk_Figure
from matplotlib.pyplot import Figure as mpl_Figure
import flowkit as fk

fcs_path = 'examples/gate_ref/data1.fcs'
gml_path = 'examples/gate_ref/gml/gml_all_gates.xml'


class PlotTestCase(unittest.TestCase):
    """
    Tests for plot functions/methods

    NOTE: Due to the difficulty of introspecting figures and images at a
          pixel-level, this TestCase only tests that plots are returned
          from plotting functions.
    """

    def test_plot_sample_histogram(self):
        sample = fk.Sample(fcs_path)
        xform_logicle = fk.transforms.LogicleTransform('logicle', param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
        sample.apply_transform(xform_logicle)

        p = sample.plot_histogram(
            'FL2-H',
            source='xform'
        )

        self.assertIsInstance(p, bk_Figure)

    def test_plot_sample_contour(self):
        sample = fk.Sample(fcs_path)
        xform_logicle = fk.transforms.LogicleTransform('logicle', param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
        sample.apply_transform(xform_logicle)

        p1 = sample.plot_contour(
            'FL1-H',
            'FL2-H',
            source='xform'
        )
        p2 = sample.plot_contour(
            'FL1-H',
            'FL2-H',
            source='xform',
            plot_events=True,
        )

        self.assertIsInstance(p1, mpl_Figure)
        self.assertIsInstance(p2, mpl_Figure)

    def test_plot_sample_scatter(self):
        sample = fk.Sample(fcs_path)
        xform_logicle = fk.transforms.LogicleTransform('logicle', param_t=10000, param_w=0.5, param_m=4.5, param_a=0)
        sample.apply_transform(xform_logicle)

        p = sample.plot_scatter(
            'FL1-H',
            'FL2-H',
            source='xform'
        )

        self.assertIsInstance(p, bk_Figure)

    def test_plot_gates(self):
        fks = fk.Session(fcs_path)
        fks.add_sample_group('my_group', gml_path)

        group_name = 'my_group'
        sample_name = 'B07'
        fks.assign_sample(sample_name, group_name)
        gate_tuples = fks.get_gate_ids(group_name)

        for gate_id, ancestors in gate_tuples:
            gate = fks.get_gate(group_name, sample_name, gate_id)
            if isinstance(gate, fk.gates.Quadrant):
                # cannot plot single quadrants of a quadrant gate
                continue
            try:
                p = fks.plot_gate('my_group', sample_name, gate_id)
            except NotImplementedError:
                continue

            self.assertIsInstance(p, bk_Figure)

    def test_plot_gated_scatter(self):
        fks = fk.Session(fcs_path)
        fks.add_sample_group('my_group', gml_path)

        group_name = 'my_group'
        sample_name = 'B07'
        fks.assign_sample(sample_name, group_name)
        x_dim = fk.Dimension('FL2-H', compensation_ref='MySpill', transformation_ref='Logicle_10000_0.5_4.5_0')
        y_dim = fk.Dimension('FL3-H', compensation_ref='MySpill', transformation_ref='Logicle_10000_0.5_4.5_0')

        p = fks.plot_scatter(
            sample_name,
            x_dim,
            y_dim,
            group_name=group_name,
            gate_id='ScaleRect1'
        )

        self.assertIsInstance(p, bk_Figure)
