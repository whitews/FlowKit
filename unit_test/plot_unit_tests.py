import unittest
from bokeh.plotting.figure import Figure
import flowkit as fk

fcs_path = 'examples/gate_ref/data1.fcs'
gml_path = 'examples/gate_ref/gml/gml_all_gates.xml'


class PlotTestCase(unittest.TestCase):
    """Tests for Session class"""
    def test_plot_gates(self):
        fks = fk.Session(fcs_path)
        fks.add_sample_group('my_group', gml_path)

        group_name = 'my_group'
        sample_name = 'B07'
        fks.assign_sample(sample_name, group_name)
        gate_ids = fks.get_gate_ids(group_name)

        for gate_id in gate_ids:
            gate = fks.get_gate(group_name, sample_name, gate_id)
            if isinstance(gate, fk.gates.Quadrant):
                # cannot plot single quadrants of a quadrant gate
                continue
            try:
                p = fks.plot_gate('my_group', sample_name, gate_id)
            except NotImplementedError:
                pass

            self.assertIsInstance(p, Figure)
