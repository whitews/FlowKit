"""
GatingStrategy class
"""
import json
import anytree
from anytree.exporter import DotExporter
# noinspection PyProtectedMember
from .._models.gates._base_gate import Gate
from .._models import gates as fk_gates
# noinspection PyProtectedMember
from .._models.transforms._base_transform import Transform
# noinspection PyProtectedMember
from .._models.transforms._matrix import Matrix
from .._models.gating_results import GatingResults


class GatingStrategy(object):
    """
    Represents a flow cytometry gating strategy, including instructions
    for compensation and transformation.
    """
    def __init__(self):
        self._cached_compensations = {}

        # keys are the object's ID (xform or matrix),
        # values are the object itself
        self.transformations = {}
        self.comp_matrices = {}

        self._gate_tree = anytree.Node('root')

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{len(self._gate_tree.descendants)} gates, {len(self.transformations)} transforms, '
            f'{len(self.comp_matrices)} compensations)'
        )

    def add_gate(self, gate, gate_path=None):
        """
        Add a gate to the gating strategy, see `gates` module. The gate ID and gate path must be
        unique in the gating strategy. A gate with a unique gate ID and parent can be added without
        specifying a gate_path. However, if the gate's ID and parent combination already exists in
        the gating strategy, a unique gate path must be provided.

        :param gate: instance from a sub-class of the Gate class
        :param gate_path: complete list of gate IDs for unique set of gate ancestors.
            Required if gate.id and gate.parent combination is ambiguous

        :return: None
        """
        if not isinstance(gate, Gate):
            raise ValueError("gate must be a sub-class of the Gate class")

        parent_id = gate.parent
        if parent_id is None:
            # If no parent gate is specified, use root
            parent_id = 'root'

        # Verify the gate parent matches the last item in the gate path (if given)
        if gate_path is not None:
            if len(gate_path) != 0:
                if parent_id != gate_path[-1]:
                    raise ValueError("The gate parent and the last item in gate path are different.")

        # Find simple case of matching gate ID + parent where no gate_path is specified.
        matched_nodes = anytree.findall(
            self._gate_tree,
            filter_=lambda g_node:
                g_node.name == gate.id and
                g_node.parent.name == parent_id
        )
        match_count = len(matched_nodes)

        if match_count != 0 and gate_path is None:
            raise KeyError(
                "A gate with ID '%s' and parent '%s' is already defined. " 
                "You must specify a gate_path as a unique list of ancestors." % (gate.id, parent_id)
            )

        # Here we either have a unique gate ID + parent, or an ambiguous gate ID + parent with a gate path.
        # It is still possible that the given gate_path already exists, we'll check that.
        # We'll find the parent node from the ID, and if there are multiple parent matches then resort
        # to using the given gate path.
        if parent_id == 'root':
            # Easiest case since a root parent is also the full path.
            parent_node = self._gate_tree
        else:
            # Find all nodes with the parent ID name. If there's only one, we've identified the correct parent.
            # If there are none, the parent doesn't exist (or is a Quad gate).
            # If there are >1, we have to compare the gate paths.
            matching_parent_nodes = anytree.search.findall_by_attr(self._gate_tree, parent_id)
            matched_parent_count = len(matching_parent_nodes)
            match_idx = None

            if matched_parent_count == 0:
                # TODO: could be in a quadrant gate
                raise ValueError("Parent gate %s does not exist in the gating strategy" % parent_id)
            elif matched_parent_count == 1:
                # There's only one match for the parent, so we're done
                match_idx = 0
            elif matched_parent_count > 1:
                for i, matched_parent_node in enumerate(matching_parent_nodes):
                    matched_parent_ancestors = [pn.name for pn in matched_parent_node.path]
                    if matched_parent_ancestors == gate_path:
                        match_idx = i
                        break

            # look up the parent node, then do one final check to make sure the new gate doesn't
            # already exist as a child of the parent
            parent_node = matching_parent_nodes[match_idx]
            parent_child_nodes = anytree.search.findall_by_attr(parent_node, gate.id, maxlevel=1)
            if len(parent_child_nodes) > 0:
                raise ValueError(
                    "A gate already exist matching gate ID %s and the specified gate path" % gate.id
                )

        node = anytree.Node(gate.id, parent=parent_node, gate=gate)

        # Quadrant gates need special handling to add their individual quadrants as children.
        # Other gates cannot have the main quadrant gate as a parent, they can only reference
        # the individual quadrants as parents.
        if isinstance(gate, fk_gates.QuadrantGate):
            for q_id, q in gate.quadrants.items():
                anytree.Node(q_id, parent=node, gate=q)

    def add_transform(self, transform):
        """
        Add a transform to the gating strategy, see `transforms` module. The transform ID must be unique in the
        gating strategy.

        :param transform: instance from a sub-class of the Transform class
        :return: None
        """
        if not isinstance(transform, Transform):
            raise ValueError("transform must be a sub-class of the Transform class")

        if transform.id in self.transformations:
            raise KeyError("Transform ID '%s' is already defined" % transform.id)

        self.transformations[transform.id] = transform

    def get_transform(self, transform_id):
        """
        Retrieve transform instance from gating strategy.

        :param transform_id: String of a valid transform ID stored in the gating strategy
        :return: Instance of a Transform sub-class
        """
        return self.transformations[transform_id]

    def add_comp_matrix(self, matrix):
        """
        Add a compensation matrix to the gating strategy, see `transforms` module. The matrix ID must be unique in the
        gating strategy.

        :param matrix: an instance of the Matrix class
        :return: None
        """
        # Only accept Matrix class instances as we need the ID
        if not isinstance(matrix, Matrix):
            raise ValueError("matrix must be an instance of the Matrix class")

        if matrix.id in self.comp_matrices:
            raise KeyError("Matrix ID '%s' is already defined" % matrix.id)

        self.comp_matrices[matrix.id] = matrix

    def get_comp_matrix(self, matrix_id):
        """
        Retrieve Matrix instance from gating strategy.

        :param matrix_id: String of a valid Matrix ID stored in the gating strategy
        :return: Matrix instance
        """
        return self.comp_matrices[matrix_id]

    def _get_gate_node(self, gate_id, gate_path=None):
        # It's not safe to just look at the gates dictionary as
        # QuadrantGate IDs cannot be parents themselves, only their component
        # Quadrant IDs can be parents.
        node_matches = anytree.findall_by_attr(self._gate_tree, gate_id)
        node_match_count = len(node_matches)

        if node_match_count == 1:
            node = node_matches[0]
        elif node_match_count > 1:
            # need to match on full gate path
            if gate_path is None:
                raise ValueError(
                    "Found multiple gates with ID %s. Provide full 'gate_path' to disambiguate." % gate_id
                )

            gate_matches = []
            gate_path_length = len(gate_path)
            for n in node_matches:
                if len(n.ancestors) != gate_path_length:
                    continue
                ancestor_matches = [a.name for a in n.ancestors if a.name in gate_path]
                if ancestor_matches == gate_path:
                    gate_matches.append(n)

            if len(gate_matches) == 1:
                node = gate_matches[0]
            elif len(gate_matches) > 1:
                raise ValueError("Report as bug: Found multiple gates with ID %s and given gate path." % gate_id)
            else:
                node = None
        else:
            node = None

        if node is None:
            # may be in a Quadrant gate
            for d in self._gate_tree.descendants:
                if isinstance(d.gate, fk_gates.QuadrantGate):
                    if gate_id in d.gate.quadrants:
                        node = d
                        continue
        if node is None:
            raise ValueError("Gate ID %s was not found in gating strategy" % gate_id)

        return node

    def get_root_gates(self):
        """
        Retrieve list of root-level gate instances.

        :return: list of Gate instances
        """
        root = self._gate_tree.root
        root_children = root.children

        root_gates = []

        for node in root_children:
            root_gates.append(node.gate)

        return root_gates

    def get_gate(self, gate_id, gate_path=None):
        """
        Retrieve a gate instance by its gate ID.

        :param gate_id: text string of a gate ID
        :param gate_path: complete list of gate IDs for unique set of gate ancestors. Required if gate_id is ambiguous
        :return: Subclass of a Gate object
        :raises KeyError: if gate ID is not found in gating strategy
        """
        node = self._get_gate_node(gate_id, gate_path)

        return node.gate

    def get_parent_gate(self, gate_id, gate_path=None):
        """
        Retrieve the gate ID for a parent gate of the given gate ID.

        :param gate_id: text string of a gate ID
        :param gate_path: complete list of gate IDs for unique set of gate ancestors. Required if gate_id is ambiguous
        :return: Subclass of a Gate object
        """
        node = self._get_gate_node(gate_id, gate_path)

        return node.parent.gate

    def get_child_gates(self, gate_id, gate_path=None):
        """
        Retrieve list of child gate instances by their parent's gate ID.

        :param gate_id: text string of a gate ID
        :param gate_path: complete list of gate IDs for unique set of gate ancestors. Required if gate_id is ambiguous
        :return: list of Gate instances
        :raises KeyError: if gate ID is not found in gating strategy
        """
        node = self._get_gate_node(gate_id, gate_path)

        child_gates = []

        for n in node.children:
            child_gates.append(n.gate)

        return child_gates

    def get_gate_ids(self):
        """
        Retrieve the list of gate IDs (with ancestors) for the gating strategy
        :return: list of tuples where the 1st item is the gate ID string and 2nd item is
                 a list of ancestor gates
        """
        gates = []
        for node in self._gate_tree.descendants:
            ancestors = [a.name for a in node.ancestors]
            gates.append((node.name, ancestors))

        return gates

    def get_gate_hierarchy(self, output='ascii', **kwargs):
        """
        Retrieve the hierarchy of gates in the gating strategy in several formats, including text,
        dictionary, or JSON. If output == 'json', extra keyword arguments are passed to json.dumps

        :param output: Determines format of hierarchy returned, either 'ascii',
            'dict', or 'JSON' (default is 'ascii')
        :return: gate hierarchy as a text string or a dictionary
        """
        if output == 'ascii':
            tree = anytree.RenderTree(self._gate_tree, style=anytree.render.ContRoundStyle())
            lines = []

            for row in tree:
                lines.append("%s%s" % (row.pre, row.node.name))

            return "\n".join(lines)
        elif output == 'json':
            dict_exporter = anytree.exporter.DictExporter(
                attriter=lambda attrs: [(k, v) for k, v in attrs if k != 'gate']
            )

            gs_dict = dict_exporter.export(self._gate_tree)
            gs_json = json.dumps(gs_dict, **kwargs)

            return gs_json
        elif output == 'dict':
            exporter = anytree.exporter.DictExporter()
            gs_dict = exporter.export(self._gate_tree)

            return gs_dict
        else:
            raise ValueError("'output' must be either 'ascii', 'json', or 'dict'")

    def export_gate_hierarchy_image(self, output_file_path):
        """
        Saves an image of the gate hierarchy in many common formats
        according to the extension given in `output_file_path`, including

            - SVG  ('svg')
            - PNG  ('png')
            - JPEG ('jpeg', 'jpg')
            - TIFF ('tiff', 'tif')
            - GIF  ('gif')
            - PS   ('ps')
            - PDF  ('pdf')

        *Requires that `graphviz` is installed.*

        :param output_file_path: File path (including file name) of image
        :return: None
        """
        DotExporter(self._gate_tree).to_picture(output_file_path)

    def _get_cached_compensation(self, sample, comp_ref):
        """
        Retrieve cached comp events if they exist
        :param sample: a Sample instance
        :param comp_ref: text string for a Matrix ID
        :return: NumPy array of the cached compensated events for the given sample, None if no cache exists.
        """
        try:
            # return a copy of cached events in case downstream modifies them
            return self._cached_compensations[sample.original_filename][comp_ref].copy()
        except KeyError:
            return None

    def _cache_compensated_events(self, sample, comp_ref, comp_events):
        """
        Cache comp events for a sample
        :param sample: a Sample instance
        :param comp_ref: text string for a Matrix ID
        :param comp_events: NumPy array of the cached compensated events for the given sample
        :return: None
        """
        if sample.original_filename not in self._cached_compensations:
            self._cached_compensations[sample.original_filename] = {
                comp_ref: comp_events
            }
        else:
            self._cached_compensations[sample.original_filename][comp_ref] = comp_events

    def gate_sample(self, sample, gate_id=None, verbose=False):
        """
        Apply a gate to a sample, returning a GatingResults instance. If the gate
        has a parent gate, all gates in the hierarchy above will be included in
        the results. If 'gate_id' is None, then all gates will be evaluated.

        :param sample: an FCS Sample instance
        :param gate_id: A gate ID or list of gate IDs to evaluate on given
            Sample. If None, all gates will be evaluated
        :param verbose: If True, print a line for each gate processed
        :return: GatingResults instance
        """
        # anytree tree allows us to iterate from the root down to the leaves
        # in an order that follows the hierarchy, thereby avoiding duplicate
        # processing of parent gates
        if gate_id is None:
            nodes = self._gate_tree.descendants
        elif isinstance(gate_id, list):
            nodes = []
            for g_id in gate_id:
                nodes.extend(anytree.search.findall_by_attr(self._gate_tree, g_id))
        else:
            nodes = anytree.search.findall_by_attr(self._gate_tree, gate_id)

        results = {}

        for item in nodes:
            g_id = item.name
            if g_id == 'root':
                continue
            gate = item.gate
            if isinstance(gate, fk_gates.QuadrantGate) and g_id in gate.quadrants:
                # TODO: think this conditional is now unnecessary and unreachable
                # This is a sub-gate, we'll process the sub-gates all at once
                # with the main QuadrantGate ID
                continue
            elif isinstance(gate, fk_gates.Quadrant):
                continue

            if verbose:
                print("%s: processing gate %s" % (sample.original_filename, g_id))
            if gate.parent is not None and gate.parent in results:
                parent_results = results[gate.parent]
            else:
                parent_results = None
            # to make the dict key unique, make a string from the ancestors,
            # but also add the set of them as a set to avoid repeated & fragile
            # string splitting in the GatingResults class
            gate_path_list = [a.name for a in item.ancestors]
            gate_path_str = "/".join(gate_path_list)
            results[g_id, gate_path_str] = gate.apply(sample, parent_results, self, gate_path_list)

            if isinstance(gate, fk_gates.QuadrantGate):
                for quad_res in results[g_id, gate_path_str].values():
                    quad_res['gate_path'] = gate_path_list
            else:
                results[g_id, gate_path_str]['gate_path'] = gate_path_list

        return GatingResults(results, sample_id=sample.original_filename)
