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

        fks.assign_sample('B07', 'my_group')
        gate_ids = fks.get_gate_ids('my_group')

        for gate_id in gate_ids:
            try:
                p = fks.plot_gate('my_group', 'B07', gate_id)
                self.assertIsInstance(p, Figure)
            except NotImplementedError:
                pass
