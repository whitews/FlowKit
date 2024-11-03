"""
GateNode Class
"""
import anytree
from .._models import gates as fk_gates
from ..exceptions import GateTreeError


class GateNode(anytree.Node):
    """
    Represents a node in a GatingStrategy gate tree and also contains Gate
    instance (and any custom sample gates). A GateNode can also be used to
    navigate the gate tree independent of the GatingStrategy.
    """
    def __init__(self, gate, parent_node):
        self.gate = gate
        self.gate_type = type(gate).__name__
        self.custom_gates = {}

        if isinstance(gate, fk_gates.Quadrant):
            super().__init__(gate.id, parent_node)
        else:
            super().__init__(gate.gate_name, parent_node)

        # Quadrant gates need special handling to add their individual quadrants as children.
        # Other gates cannot have the main quadrant gate as a parent, they can only reference
        # the individual quadrants as parents.
        if isinstance(gate, fk_gates.QuadrantGate):
            for _, q in gate.quadrants.items():
                GateNode(q, parent_node=self)

    def add_custom_gate(self, sample_id, gate):
        """
        Adds a custom gate variation to the node, useful for samples needing
        custom gate boundaries. Note the custom gate must be the same gate
        type, have the same gate name, and the same dimension IDs as the
        GateNode's template gate.

        :param sample_id: text string used to identify the custom gate
        :param gate: a Gate instance to use for the new custom gate. Must
            match the template gate type.
        :return: None
        """
        # First, check the gate type matches the template gate
        gate_type = type(gate).__name__
        if gate_type != self.gate_type:
            raise GateTreeError(
                "Custom gate must match the template gate type (given %s, should be %s)" %
                (gate_type, self.gate_type)
            )

        # check if a custom gate already exists for given sample ID
        if sample_id in self.custom_gates:
            raise GateTreeError("A custom gate already exists for %s" % sample_id)

        # Finally, check dimension IDs
        template_dim_ids = self.gate.get_dimension_ids()
        new_dim_ids = gate.get_dimension_ids()
        if template_dim_ids != new_dim_ids:
            raise GateTreeError("Custom gate dimensions IDs must match the template gate")

        # If all the above passed, we simply store the custom gate
        self.custom_gates[sample_id] = gate

    def is_custom_gate(self, sample_id):
        """
        Determine if a sample ID has a custom gate
        :param sample_id: Sample ID string
        :return: Boolean value for whether the sample ID has a custom gate
        """
        if sample_id in self.custom_gates:
            return True

        return False

    def remove_custom_gate(self, sample_id):
        """
        Removes a custom gate variation from the node. No error is thrown if a
        custom gate for the sample ID does not exist.

        :param sample_id: text string used to identify the custom gate
        :return: None
        """
        self.custom_gates.pop(sample_id, None)

    def get_gate(self, sample_id=None):
        """
        Get Gate instance from GateNode. Specify sample_id to get sample custom gate.
        If sample_id is None or not found in GateNode, the template gate is returned.

        :param sample_id: Sample ID string to lookup custom gate. If None or not found, template gate is returned
        :return: Gate instance
        """
        if sample_id in self.custom_gates:
            return self.custom_gates[sample_id]

        return self.gate
