"""
GatingStrategy class
"""
import json
import anytree
import pandas as pd
from anytree.exporter import DotExporter
import numpy as np
import networkx as nx
from .._models import dimension
from .._models.gate_node import GateNode
# noinspection PyProtectedMember
from .._models.gates._base_gate import Gate
from .._models import gates as fk_gates
# noinspection PyProtectedMember
from .._models.transforms._base_transform import Transform
# noinspection PyProtectedMember
from .._models.transforms._matrix import Matrix, SpectralMatrix
from .._models.gating_results import GatingResults
from ..exceptions import GateTreeError, GateReferenceError, QuadrantReferenceError


class GatingStrategy(object):
    """
    Represents a flow cytometry gating strategy, including instructions
    for compensation and transformation.
    """
    resolver = anytree.Resolver()

    def __init__(self):
        self._cached_preprocessed_events = {}

        # keys are the object's ID (xform or matrix),
        # values are the object itself
        self.transformations = {}
        self.comp_matrices = {}

        self._gate_tree = anytree.Node('root')

        # use a directed acyclic graph for later processing and enforcing there
        # are no cyclical gate relationships.
        # Each node is made from a tuple representing its full gate path
        self._dag = nx.DiGraph()
        self._dag.add_node(('root',))

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{len(self._gate_tree.descendants)} gates, {len(self.transformations)} transforms, '
            f'{len(self.comp_matrices)} compensations)'
        )

    def add_gate(self, gate, gate_path, sample_id=None):
        """
        Add a gate to the gating strategy, see `gates` module. The gate ID and gate path
        must be unique in the gating strategy. Custom sample gates may be added by specifying
        an optional sample ID. Note, the gate & gate path must already exist prior to adding
        custom sample gates.

        :param gate: instance from a subclass of the Gate class
        :param gate_path: complete ordered tuple of gate names for unique set of gate ancestors
        :param sample_id: text string for specifying given gate as a custom Sample gate

        :return: None
        """
        if not isinstance(gate, Gate):
            raise TypeError("gate must be a sub-class of the Gate class")

        # Verify gate_path is a tuple, else user gets a cryptic error for
        # something that is simple to fix
        if not isinstance(gate_path, tuple):
            raise TypeError("gate_path must be a tuple not %s" % str(type(gate_path)))

        # make string representation of parent path, used for anytree Resolver later
        parent_abs_gate_path = "/" + "/".join(gate_path)

        # Verify gate name is not "." or ".." as these are incompatible w/ the
        # current version of anytree (see open issue https://github.com/c0fec0de/anytree/issues/269)
        if gate.gate_name in ['.', '..']:
            raise GateTreeError(
                "Gate name '%s' is incompatible with FlowKit. Gate was found in path: %s" %
                (gate.gate_name, parent_abs_gate_path)
            )

        # We need the parent gate (via its node) for 2 reasons:
        #   1) To verify the parent exists when creating a new node
        #   2) Verify the parent is NOT a QuadrantGate, as only
        #      Quadrants of a QuadrantGate can be a parent.
        try:
            parent_node = self.resolver.get(self._gate_tree, parent_abs_gate_path)
        except anytree.ResolverError:
            # this should never happen unless someone messed with the gate tree
            raise GateTreeError("Parent gate %s doesn't exist" % parent_abs_gate_path)

        # If parent is root, then there is no parent gate
        if parent_node.name != 'root':
            if isinstance(parent_node.gate, fk_gates.QuadrantGate):
                raise GateTreeError(
                    "Parent gate %s is a QuadrantGate and cannot be used as a parent gate directly. "
                    "Only an individual Quadrant can be used as a parent."
                )

        # determine if gate already exists with name and path
        abs_gate_path = list(gate_path) + [gate.gate_name]
        abs_gate_path = "/" + "/".join(abs_gate_path)
        try:
            node = self.resolver.get(self._gate_tree, abs_gate_path)
        except anytree.ResolverError:
            # this is expected if the gate doesn't already exist
            node = None

        if node is not None:
            # A node was found. If no sample ID is given, the gate
            # already exists and is the template gate.
            if sample_id is None:
                raise GateTreeError("Gate %s already exists" % abs_gate_path)
            else:
                # Attempt to create sample custom gate, GateNode.add_gate will
                # raise a GateTreeError if it already exists.
                node.add_custom_gate(sample_id, gate)
        else:
            # We need to create a new node in the tree.
            GateNode(gate, parent_node)
            self._rebuild_dag()

    def is_custom_gate(self, sample_id, gate_name, gate_path=None):
        """
        Determine if a custom gate exists for a sample ID.

        :param sample_id: Sample ID string
        :param gate_name: text string of a gate name
        :param gate_path: complete ordered tuple of gate names for unique set of gate ancestors.
            Required if gate_name is ambiguous
        :return: Boolean value for whether the sample ID has a custom gate
        """
        node = self._get_gate_node(gate_name, gate_path)

        return node.is_custom_gate(sample_id)

    def get_gate(self, gate_name, gate_path=None, sample_id=None):
        """
        Retrieve a gate instance by its gate ID (gate name and optional gate_path).
        If a sample_id is specified, the custom sample gate will be returned if it
        exists.

        :param gate_name: text string of a gate name
        :param gate_path: complete ordered tuple of gate names for unique set of gate ancestors.
            Required if gate_name is ambiguous
        :param sample_id: Sample ID string to lookup custom gate. If None or not found, template gate is returned
        :return: Subclass of a Gate object
        :raises GateReferenceError: if gate ID is not found in gating strategy
        :raises QuadrantReferenceError: if gate ID references a single Quadrant (specify the QuadrantGate ID instead)
        """
        node = self._get_gate_node(gate_name, gate_path)

        if isinstance(node.gate, fk_gates.Quadrant):
            # A Quadrant isn't a true gate, raise error indicating to call its QuadrantGate
            raise QuadrantReferenceError(
                "%s references a Quadrant, specify the owning QuadrantGate %s instead" % (gate_name, node.parent)
            )

        return node.get_gate(sample_id=sample_id)

    def _rebuild_dag(self):
        dag_edges = []

        for node in self._gate_tree.descendants:
            node_tuple = tuple([n.name for n in node.path])
            parent_node_tuple = node_tuple[:-1]
            dag_edges.append((parent_node_tuple, node_tuple))

            gate = node.gate

            if isinstance(gate, fk_gates.BooleanGate):
                bool_gate_refs = gate.gate_refs
                for gate_ref in bool_gate_refs:
                    gate_ref_node_tuple = tuple(gate_ref['path']) + (gate_ref['ref'],)
                    dag_edges.append((gate_ref_node_tuple, node_tuple))

            if isinstance(gate, fk_gates.QuadrantGate):
                for q_id, q in gate.quadrants.items():
                    q_node_tuple = node_tuple + (q_id,)

                    dag_edges.append((node_tuple, q_node_tuple))

        self._dag = nx.DiGraph(dag_edges)

    def _get_successor_node_paths(self, gate_node):
        gate = gate_node.gate

        # use a set of tuples since a Boolean gate (AND / OR) can
        # reference >1 gate, the Boolean gate would get referenced
        # twice. We don't need duplicates.
        successor_node_tuples = set()

        # some special handling if given a QuadrantGate to remove
        if isinstance(gate, fk_gates.QuadrantGate):
            # need to collect the Quadrant references as these
            # may be referenced in a BooleanGate. In that case,
            # we won't find the BooleanGate in a normal successor
            # check of the DAG. See comment about using networkx
            # to find successors below in non-QuadrantGate case.
            for quadrant_child_node in gate_node.children:
                quad_node_tuple = tuple(n.name for n in quadrant_child_node.path)
                quad_successor_node_tuples = set(self._dag.successors(quad_node_tuple))

                successor_node_tuples.update(quad_successor_node_tuples)
        else:
            # Use networkx graph to get dependent gates instead of anytree,
            # since the DAG keeps track of all dependencies (incl. bool gates),
            # which also covers bool gate dependencies of the children.
            # Networkx descendants works for DAGs & returns a set of node strings.
            node_tuple = tuple(n.name for n in gate_node.path)
            successor_node_tuples = set(self._dag.successors(node_tuple))

        return successor_node_tuples

    def rename_gate(self, gate_name, new_gate_name, gate_path=None):
        """
        Rename a gate in the gating strategy. Any custom sample gates associated with the gate
        will also be renamed.

        :param gate_name: text string of existing gate name
        :param new_gate_name: text string for new gate name
        :param gate_path: complete ordered tuple of gate names for unique set of gate ancestors.
            Required if gate_name is ambiguous
        :return: None
        """
        # First, get the gate node from anytree
        # Note, this will raise an error on ambiguous gates so no need
        # to handle that case
        gate_node = self._get_gate_node(gate_name, gate_path=gate_path)
        gate = gate_node.gate
        orig_full_gate_path = tuple(n.name for n in gate_node.path)
        new_full_gate_path = orig_full_gate_path[:-1] + (new_gate_name,)  # needed for updating Boolean refs later

        # check successors for any Boolean gates that reference the renamed gate
        # Note, these needs to be retrieved before modifying the gate name
        successor_node_tuples = self._get_successor_node_paths(gate_node)

        # At this point we're about to modify the tree and
        # renaming a gate nullifies any previous results,
        # so clear cached events
        self.clear_cache()

        # Check successor gates for a Boolean gate.
        # If present, it references the renamed gate & that reference needs updating
        for s_tuple in successor_node_tuples:
            s_gate_node = self._get_gate_node(s_tuple[-1], gate_path=s_tuple[:-1])
            s_gate = s_gate_node.gate

            if isinstance(s_gate, fk_gates.BooleanGate):
                bool_gate_refs = s_gate.gate_refs
                for bool_gate_ref in bool_gate_refs:
                    # Determine whether the modified gate name's path is
                    # referenced within the Boolean gate's reference gate.
                    # This is tricky b/c tuples don't have a built-in
                    # function like a set's 'issubset' & converting to
                    # sets won't work b/c we want a matching sequence where
                    # repeat values can exist.
                    # The most straight-forward case: the reference is the
                    # modified gate.
                    bool_gate_ref_full_path = bool_gate_ref['path'] + tuple((bool_gate_ref['ref'],))
                    if bool_gate_ref_full_path == orig_full_gate_path:
                        # just need to change the 'ref' key value
                        bool_gate_ref['ref'] = new_gate_name
                    elif len(bool_gate_ref_full_path) >= len(orig_full_gate_path):
                        # The reference may contain the path of the original gate
                        #
                        bgr_path_prefix = bool_gate_ref['path'][:len(orig_full_gate_path)]
                        bgr_path_postfix = bool_gate_ref['path'][len(orig_full_gate_path):]
                        if bgr_path_prefix == orig_full_gate_path:
                            # need to update the Boolean ref 'path' value with
                            # this the new path
                            bool_gate_ref['path'] = new_full_gate_path + bgr_path_postfix

                    # Any other case, the reference gate path is longer than the modified gate,
                    # so not affected by the change.

        # Need to update both the gate node & the gate.
        # The gate node name is straight-forward for all cases.
        gate_node.name = new_gate_name

        if isinstance(gate, fk_gates.Quadrant):
            # individual quadrants have an 'id' & not a 'gate_name'
            # attribute since they aren't true gates themselves.
            gate.id = new_gate_name

            # And their owning QuadrantGate references them via a
            # dict whose key needs updating as well.
            owning_quad_gate = gate_node.parent.gate
            owning_quad_gate.quadrants[new_gate_name] = owning_quad_gate.quadrants.pop(gate_name)

            # Finally, check if the QuadrantGate node has any custom gates.
            # These Quadrant instances and dict keys need updating too.
            for custom_quad_gate in gate_node.parent.custom_gates.values():
                # find quadrant
                custom_quadrant = custom_quad_gate.quadrants[gate_name]

                # update 'id' attribute
                custom_quadrant.id = new_gate_name

                # update key
                custom_quad_gate.quadrants[new_gate_name] = custom_quad_gate.quadrants.pop(gate_name)
        else:
            # All other gate types are simpler, and only have gate_name
            gate.gate_name = new_gate_name

            # check for custom gates, need to change those too
            for custom_gate in gate_node.custom_gates.values():
                custom_gate.gate_name = new_gate_name

        # rebuild DAG
        self._rebuild_dag()

    def _remove_template_gate(self, gate_node, keep_children=False):
        """
        Handles case for removing template gate from gate tree.

        :param gate_node: GateNode to remove
        :param keep_children: Whether to keep child gates. If True, the child gates will be
            remapped to the removed gate's parent. Default is False, which will delete all
            descendant gates.
        :return: None
        """
        gate = gate_node.gate

        # single quadrants can't be removed, their "parent" QuadrantGate must be removed
        if isinstance(gate, fk_gates.Quadrant):
            raise QuadrantReferenceError(
                "Quadrant '%s' cannot be removed, remove the full QuadrantGate '%s' instead"
                % (gate.id, gate_node.parent.name)
            )

        successor_node_tuples = self._get_successor_node_paths(gate_node)

        # check successor gates for a boolean gate,
        # if present throw a GateTreeError
        for s_tuple in successor_node_tuples:
            s_gate_node = self._get_gate_node(s_tuple[-1], gate_path=s_tuple[:-1])
            s_gate = s_gate_node.gate

            if isinstance(s_gate, fk_gates.BooleanGate):
                raise GateTreeError("BooleanGate %s references gate %s" % (s_gate.gate_name, gate.gate_name))

        # At this point we're about to modify the tree and
        # removing a gate nullifies any previous results,
        # so clear cached events
        self.clear_cache()

        if keep_children:
            parent_node = gate_node.parent

            # quadrant gates need to be handled differently from other gates
            if isinstance(gate, fk_gates.QuadrantGate):
                # The immediate children will be quadrants, but they will get deleted.
                # We do need to check if the quadrants have children and set their
                # parent to the quadrant gate parent.
                child_nodes = []
                for quad in gate_node.children:
                    child_nodes.extend(quad.children)
            else:
                child_nodes = gate_node.children

            for cn in child_nodes:
                # set each child node to the parent of the removed node
                cn.parent = parent_node

        # remove from anytree, any descendants get removed so fine for keep_children=False
        gate_node.parent = None

        # Now, rebuild the DAG (easier than modifying it)
        self._rebuild_dag()

    def remove_gate(self, gate_name, gate_path=None, sample_id=None, keep_children=False):
        """
        Remove a gate from the gating strategy. Any descendant gates will also be removed
        unless keep_children=True. In all cases, if a BooleanGate exists that references
        the gate to remove, a GateTreeError will be thrown indicating the BooleanGate
        must be removed prior to removing the gate.

        :param gate_name: text string of a gate name
        :param gate_path: complete ordered tuple of gate names for unique set of gate ancestors.
            Required if gate_name is ambiguous
        :param sample_id: text string for Sample ID to remove only its custom Sample gate and
            retain the template gate (and other custom gates if they exist).
        :param keep_children: Whether to keep child gates. If True, the child gates will be
            remapped to the removed gate's parent. Default is False, which will delete all
            descendant gates.
        :return: None
        """
        # First, get the gate node from anytree
        # Note, this will raise an error on ambiguous gates so no need
        # to handle that case
        gate_node = self._get_gate_node(gate_name, gate_path=gate_path)

        # determine whether user requested to remove the template gate or a custom Sample gate
        if sample_id is None:
            # Remove template gate, which removes the entire node from the tree
            self._remove_template_gate(gate_node, keep_children=keep_children)
        else:
            # Remove custom sample gate, which is simpler.
            # Stay silent if key doesn't exist.
            gate_node.remove_custom_gate(sample_id)

    def add_transform(self, transform_id, transform):
        """
        Add a transform to the gating strategy, see `transforms` module. The transform ID must be unique in the
        gating strategy.

        :param transform_id: A string identifying the transform
        :param transform: instance from a subclass of the Transform class
        :return: None
        """
        if not isinstance(transform, Transform):
            raise TypeError("transform must be a sub-class of the Transform class")

        if transform_id in self.transformations:
            raise KeyError("Transform ID '%s' is already defined" % transform_id)

        self.transformations[transform_id] = transform

    def get_transform(self, transform_id):
        """
        Retrieve transform instance from gating strategy.

        :param transform_id: String of a valid transform ID stored in the gating strategy
        :return: Instance of a Transform subclass
        """
        return self.transformations[transform_id]

    def add_comp_matrix(self, matrix_id, matrix):
        """
        Add a compensation matrix to the gating strategy, see `transforms` module. The matrix ID must be unique in the
        gating strategy.

        :param matrix_id: Text string used to identify the matrix (cannot be 'uncompensated' or 'fcs')
        :param matrix: an instance of the Matrix class
        :return: None
        """
        # Technically, the GML 2.0 spec states 'FCS' (note uppercase) is reserved,
        # but we'll cover that case-insensitive
        if matrix_id == 'uncompensated' or matrix_id.lower() == 'fcs':
            raise ValueError(
                "Matrix IDs 'uncompensated' and 'FCS' are reserved compensation references " +
                "used in Dimension instances to specify that channel data should either be " +
                "uncompensated or compensated using the spill value from a Sample's metadata"
            )

        # Only accept Matrix class instances as we need the ID
        if not isinstance(matrix, (Matrix, SpectralMatrix)):
            raise TypeError("matrix must be an instance of the Matrix class")

        if matrix_id in self.comp_matrices:
            raise KeyError("Matrix ID '%s' is already defined" % matrix_id)

        self.comp_matrices[matrix_id] = matrix

    def get_comp_matrix(self, matrix_id):
        """
        Retrieve Matrix instance from gating strategy.

        :param matrix_id: String of a valid Matrix ID stored in the gating strategy
        :return: Matrix instance
        """
        return self.comp_matrices[matrix_id]

    def _get_gate_node(self, gate_name, gate_path=None):
        """
        Retrieve a GateNode instance by its gate ID (gate name and optional gate_path).
        A GateNode contains the Gate instance (and any custom sample gates). A GateNode
        can also be used to navigate the gate tree independent of the GatingStrategy,
        though the GateNode should not be used to modify the gate tree.

        :param gate_name: text string of a gate name
        :param gate_path: complete ordered tuple of gate names for unique set of gate ancestors.
            Required if gate_name is ambiguous
        :return: GateNode object
        :raises GateReferenceError: if gate ID is not found in gating strategy
        """
        node_matches = anytree.findall_by_attr(self._gate_tree, gate_name)
        node_match_count = len(node_matches)

        if node_match_count == 1:
            node = node_matches[0]
        elif node_match_count > 1:
            # need to match on full gate path
            # TODO: what if QuadrantGate is re-used, does this still work for that case
            if gate_path is None:
                raise GateReferenceError(
                    "Found multiple gates with name %s. Provide full 'gate_path' to disambiguate." % gate_name
                )

            gate_matches = []
            gate_path_length = len(gate_path)
            for n in node_matches:
                if len(n.ancestors) != gate_path_length:
                    continue
                ancestor_matches = tuple((a.name for a in n.ancestors if a.name in gate_path))
                if ancestor_matches == gate_path:
                    gate_matches.append(n)

            if len(gate_matches) == 1:
                node = gate_matches[0]
            elif len(gate_matches) > 1:
                raise ValueError("Report as bug: Found multiple gates with ID %s and given gate path." % gate_name)
            else:
                node = None
        else:
            node = None

        if node is None:
            raise GateReferenceError("Gate name %s was not found in gating strategy" % gate_name)

        return node

    def find_matching_gate_paths(self, gate_name):
        """
        Find all gate paths for given gate name.

        :param gate_name: text string of a gate name
        :return: list of gate paths (list of tuples)
        """
        node_matches = anytree.findall_by_attr(self._gate_tree, gate_name)

        gate_path_list = []

        for node in node_matches:
            gate_path_list.append(
                tuple((a.name for a in node.ancestors))
            )

        return gate_path_list

    def get_root_gates(self, sample_id=None):
        """
        Retrieve list of root-level gate instances.

        :param sample_id: Sample ID string to retrieve custom gates if present. If None, template gates are returned
        :return: list of Gate instances
        """
        root = self._gate_tree.root
        root_children = root.children

        root_gates = []

        for node in root_children:
            root_gates.append(node.get_gate(sample_id=sample_id))

        return root_gates

    def get_parent_gate_id(self, gate_name, gate_path=None):
        """
        Retrieve the parent Gate ID for the given gate ID.

        :param gate_name: text string of a gate name
        :param gate_path: complete ordered tuple of gate names for unique set of gate ancestors.
            Required if gate_name is ambiguous
        :return: a gate ID (tuple of gate name and gate path)
        """
        node = self._get_gate_node(gate_name, gate_path)

        if node.parent.name == 'root':
            return None

        parent_gate_path = tuple((a.name for a in node.parent.ancestors))

        return node.parent.name, parent_gate_path

    def get_child_gate_ids(self, gate_name, gate_path=None):
        """
        Retrieve list of child gate instances by their parent's gate ID.

        :param gate_name: text string of a gate name
        :param gate_path: complete ordered tuple of gate names for unique set of gate ancestors.
            Required if gate_name is ambiguous
        :return: list of Gate IDs (tuple of gate name plus gate path). Returns an empty
            list if no child gates exist.
        :raises GateReferenceError: if gate ID is not found in gating strategy or if gate
            name is ambiguous
        """
        node = self._get_gate_node(gate_name, gate_path)

        child_gate_ids = []

        for n in node.children:
            ancestor_path = tuple((a.name for a in n.ancestors))
            child_gate_ids.append((n.name, ancestor_path))

        return child_gate_ids

    def get_gate_ids(self):
        """
        Retrieve the list of gate IDs (with ancestors) for the gating strategy.

        :return: list of tuples (1st item is gate name string, 2nd item is a list of ancestor gate names)
        """
        gates = []
        for node in self._gate_tree.descendants:
            ancestors = tuple((a.name for a in node.ancestors))
            gates.append((node.name, ancestors))

        return gates

    def get_max_depth(self):
        """
        Returns the max depth of the gating hierarchy.
        """
        return self._gate_tree.height

    def get_gate_hierarchy(self, output='ascii', **kwargs):
        """
        Retrieve the hierarchy of gates in the gating strategy in several formats, including text,
        dictionary, or JSON. If output == 'json', extra keyword arguments are passed to 'json.dumps'

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

    def _get_cached_preprocessed_events(self, sample_id, comp_ref, xform_ref, dim_idx=None):
        """
        Retrieve cached pre-processed events (if they exist).

        :param sample_id: a text string for a Sample ID
        :param comp_ref: text string for a Matrix ID
        :param xform_ref: text string for a Transform ID
        :param dim_idx: channel index for requested preprocessed events. If None, all channels are requested.
        :return: NumPy array of the cached compensated events for the given sample, None if no cache exists.
        """
        ref_tuple = (comp_ref, xform_ref, dim_idx)

        try:
            # return a copy of cached events in case downstream modifies them
            return self._cached_preprocessed_events[sample_id][ref_tuple].copy()
        except KeyError:
            return None

    def clear_cache(self):
        """
        Clears all cached pre-processed events stored in the GatingStrategy. This is useful to
        reduce memory usage after analyzing large data sets. Clearing the cache will not affect
        any results previously retrieved.

        :return: None
        """
        self._cached_preprocessed_events = {}

    def _cache_preprocessed_events(self, preprocessed_events, sample_id, comp_ref, xform_ref, dim_idx=None):
        """
        Cache pre-processed events for a Sample instance

        :param preprocessed_events: NumPy array of the cached compensated events for the given sample
        :param sample_id: a text string for a Sample ID
        :param comp_ref: text string for a Matrix ID
        :param xform_ref: text string for a Transform ID
        :param dim_idx: channel index for given preprocessed events. If None, all channels where preprocessed
        :return: None
        """
        if sample_id not in self._cached_preprocessed_events:
            self._cached_preprocessed_events[sample_id] = {}

        sample_cache = self._cached_preprocessed_events[sample_id]
        ref_tuple = (comp_ref, xform_ref, dim_idx)

        if ref_tuple not in sample_cache:
            # copy the given events here to decouple from any use analysis
            sample_cache[ref_tuple] = preprocessed_events.copy()

    def _compensate_sample(self, dim_comp_refs, sample):
        dim_comp_ref_count = len(dim_comp_refs)

        if dim_comp_ref_count == 0:
            events = sample.get_events(source='raw')
            return events.copy()
        elif dim_comp_ref_count > 1:
            raise NotImplementedError(
                "Mixed compensation between individual channels is not "
                "implemented. Never seen it, but if you are reading this "
                "message, submit an issue to have it implemented."
            )
        else:
            comp_ref = list(dim_comp_refs)[0]

        # noinspection PyProtectedMember
        events = self._get_cached_preprocessed_events(
            sample.id,
            comp_ref,
            None,
            None
        )

        if events is not None:
            return events

        if comp_ref == 'FCS':
            meta = sample.get_metadata()

            if 'spill' not in meta and 'spillover' not in meta:
                # GML 2.0 spec states if 'FCS' is specified but no spill is present, treat as uncompensated
                events = sample.get_events(source='raw')
                return events.copy()

            try:
                spill = meta['spillover']  # preferred, per FCS standard
            except KeyError:
                spill = meta['spill']

            detectors = [sample.pnn_labels[i] for i in sample.fluoro_indices]
            fluorochromes = [sample.pns_labels[i] for i in sample.fluoro_indices]
            matrix = Matrix(spill, detectors, fluorochromes, null_channels=sample.null_channels)
        else:
            # lookup specified comp-ref in gating strategy
            matrix = self.comp_matrices[comp_ref]

        if matrix is not None:
            events = matrix.apply(sample)
            # cache the comp events
            # noinspection PyProtectedMember
            self._cache_preprocessed_events(
                events,
                sample.id,
                comp_ref,
                None,
                None
            )

        return events

    def _preprocess_sample_events(self, sample, gate, cache_events=False):
        # TODO: consider making method public, could be useful for users
        pnn_labels = sample.pnn_labels
        pns_labels = sample.pns_labels
        # FlowJo replaces slashes with underscores, so make a set of labels with that replacement
        flowjo_pnn_labels = [label.replace('/', '_') for label in pnn_labels]

        dim_indices = []
        dim_ids = []
        dim_comp_refs = set()
        new_dims = []
        new_dim_ids = []
        dim_xform = []

        for dim in gate.dimensions:
            dim_comp = False
            if dim.compensation_ref not in [None, 'uncompensated']:
                dim_comp_refs.add(dim.compensation_ref)
                dim_comp = True

            if isinstance(dim, dimension.RatioDimension):
                # dimension is a transform of other dimensions
                new_dims.append(dim)
                new_dim_ids.append(dim.ratio_ref)
                continue
            elif isinstance(dim, dimension.QuadrantDivider):
                dim_id = dim.dimension_ref
            else:
                dim_id = dim.id

            if dim_id in pnn_labels:
                dim_indices.append(pnn_labels.index(dim_id))
            elif dim_id in pns_labels:
                dim_indices.append(pns_labels.index(dim_id))
            elif dim_id in flowjo_pnn_labels:
                dim_indices.append(flowjo_pnn_labels.index(dim_id))
            else:
                # for a referenced comp, the label may have been the
                # fluorochrome instead of the channel's PnN label. If so,
                # the referenced matrix object will also have the detector
                # names that will match
                if not dim_comp:
                    raise LookupError(
                        "%s is not found as a channel label or channel reference in %s" % (dim_id, sample)
                    )
                matrix = self.comp_matrices[dim.compensation_ref]
                try:
                    matrix_dim_idx = matrix.fluorochromes.index(dim_id)
                except ValueError:
                    raise ValueError("%s not found in list of matrix fluorochromes" % dim_id)
                detector = matrix.detectors[matrix_dim_idx]
                dim_indices.append(pnn_labels.index(detector))

            dim_ids.append(dim_id)

            dim_xform.append(dim.transformation_ref)

        # compensate will cache events regardless of cache_events arg value
        # this is much more universally useful to speed up analysis as the
        # same comp matrix is nearly always used for all sample gates.
        events = self._compensate_sample(dim_comp_refs, sample)

        for i, dim in enumerate(dim_indices):
            if dim_xform[i] is not None:
                if len(dim_comp_refs) > 0:
                    comp_ref = list(dim_comp_refs)[0]
                else:
                    comp_ref = None

                xform = self.transformations[dim_xform[i]]

                if cache_events:
                    cached_events = self._get_cached_preprocessed_events(
                        sample.id,
                        comp_ref,
                        dim_xform[i],
                        dim_idx=dim
                    )
                else:
                    cached_events = None

                if cached_events is not None:
                    events[:, [dim]] = cached_events
                else:
                    xform_events = xform.apply(events[:, [dim]])
                    events[:, [dim]] = xform_events
                    if cache_events:
                        self._cache_preprocessed_events(
                            xform_events,
                            sample.id,
                            comp_ref,
                            dim_xform[i],
                            dim_idx=dim
                        )

        df_events = pd.DataFrame(events[:, dim_indices], columns=dim_ids)
        if len(new_dims) > 0:
            new_dim_events = self._process_new_dims(sample, new_dims)
            df_new_dim_events = pd.DataFrame(new_dim_events, columns=new_dim_ids)
            df_events = pd.concat([df_events, df_new_dim_events], axis=1)

        return df_events

    def _process_new_dims(self, sample, new_dims):
        new_dims_events = []

        for new_dim in new_dims:
            # Allows gate classes to handle new dimensions created from
            # ratio transforms. Note, the ratio transform's apply method is
            # different from other transforms in that it takes a sample argument
            # and not an events argument

            # new dimensions are defined by transformations of other dims
            try:
                new_dim_xform = self.transformations[new_dim.ratio_ref]
            except KeyError:
                raise KeyError("New dimensions must provide a transformation")

            xform_events = new_dim_xform.apply(sample)

            if new_dim.transformation_ref is not None:
                xform = self.transformations[new_dim.transformation_ref]
                xform_events = xform.apply(xform_events)

            new_dims_events.append(xform_events)

        return np.hstack(new_dims_events)

    @staticmethod
    def _apply_parent_results(sample, gate, results, parent_results):
        if parent_results is not None:
            if isinstance(gate, fk_gates.QuadrantGate):
                results_and_parent = {}
                for q_id, q_result in results.items():
                    results_and_parent[q_id] = np.logical_and(parent_results['events'], q_result)
            else:
                results_and_parent = np.logical_and(parent_results['events'], results)
            parent_count = parent_results['count']
        else:
            # no parent, so results are unchanged & parent count is total count
            parent_count = sample.event_count
            results_and_parent = results

        if isinstance(gate, fk_gates.QuadrantGate):
            final_results = {}

            for q_id, q_result in results_and_parent.items():
                q_event_count = q_result.sum()

                final_results[q_id] = {
                    'sample': sample.id,
                    'events': q_result,
                    'count': q_event_count,
                    'absolute_percent': (q_event_count / float(sample.event_count)) * 100.0,
                    'relative_percent': (q_event_count / float(parent_count)) * 100.0,
                    'gate_type': gate.gate_type
                }
        else:
            event_count = results_and_parent.sum()

            # check parent_count to avoid div by zero
            if parent_count == 0:
                relative_percent = 0.0
            else:
                relative_percent = (event_count / float(parent_count)) * 100.0

            final_results = {
                'sample': sample.id,
                'events': results_and_parent,
                'count': event_count,
                'absolute_percent': (event_count / float(sample.event_count)) * 100.0,
                'relative_percent': relative_percent,
                'gate_type': gate.gate_type
            }

        return final_results

    def gate_sample(self, sample, cache_events=False, verbose=False):
        """
        Apply GatingStrategy to a Sample instance, returning a GatingResults instance.

        :param sample: an FCS Sample instance
        :param cache_events: Whether to cache pre-processed events (compensated and transformed). This can
            be useful to speed up processing of gates that share the same pre-processing instructions for
            the same channel data, but can consume significantly more memory space. See the related
            clear_cache method for additional information. Default is False.
        :param verbose: If True, print a line for each gate processed
        :return: GatingResults instance
        """
        sample_id = sample.id
        results = {}

        # The goal here is to avoid re-analyzing any gates.
        # For every gate processed, the results will be cached.
        # Since the nodes are retrieved in hierarchy order,
        # we simply check our cached results for the gate's
        # parent results. This should avoid having any gate
        # processed more than once.
        process_order = list(nx.algorithms.topological_sort(self._dag))

        process_order.remove(('root',))

        for item in process_order:
            # to make the dict key unique, make a string from the ancestors,
            # but also add the set of them as a set to avoid repeated & fragile
            # string splitting in the GatingResults class
            g_id = item[-1]
            g_path = item[:-1]
            gate_path_str = "/".join(g_path)

            p_id = g_path[-1]
            p_path = g_path[:-1]
            parent_gate_path_str = "/".join(p_path)

            g_uid = (g_id, gate_path_str)
            p_uid = (p_id, parent_gate_path_str)

            # check if this gate has already been processed
            if g_uid in results:
                continue

            # get_gate returns QuadrantReferenceError when requesting a single quadrant,
            # we'll check for that. All quadrant sub-gates will get processed with the
            # main QuadrantGate
            try:
                gate = self.get_gate(g_id, g_path, sample_id=sample_id)
            except QuadrantReferenceError:
                continue

            if verbose:
                is_custom_gate = self.is_custom_gate(sample_id, g_id, g_path)
                if is_custom_gate:
                    custom_gate_str = ' [custom]'
                else:
                    custom_gate_str = ''

                print("%s: processing gate %s%s" % (sample_id, g_id, custom_gate_str))

            # look up parent results
            parent_results = None  # default to None
            if p_id != 'root':
                try:
                    _ = self.get_gate(p_id, p_path)
                    if p_uid in results:
                        parent_results = results[p_uid]
                except QuadrantReferenceError:
                    # get quadrant results from quadrant gate
                    q_gate_name = p_path[-1]
                    q_gate_path = p_path[:-1]
                    q_gate_res_key = (q_gate_name, "/".join(q_gate_path))
                    if q_gate_res_key in results:
                        parent_results = results[q_gate_res_key][p_id]

            # Call appropriate gate apply method
            if isinstance(gate, fk_gates.BooleanGate):
                # BooleanGate is a bit different, needs a DataFrame of gate ref results
                bool_gate_ref_results = {}
                for gate_ref in gate.gate_refs:
                    gate_ref_res_key = (gate_ref['ref'], "/".join(gate_ref['path']))

                    try:
                        # get_gate used just to differentiate between quadrant results & regular gate results
                        _ = self.get_gate(gate_ref['ref'], gate_ref['path'])
                        bool_gate_ref_results[gate_ref_res_key] = results[gate_ref_res_key]['events']
                    except QuadrantReferenceError:
                        quad_gate_name = gate_ref['path'][-1]
                        quad_gate_path = gate_ref['path'][:-1]

                        quad_gate_res_key = (quad_gate_name, "/".join(quad_gate_path))
                        quad_gate_results = results[quad_gate_res_key]

                        # but the quadrant result is what we're after
                        bool_gate_ref_results[gate_ref_res_key] = quad_gate_results[gate_ref['ref']]['events']

                gate_results = gate.apply(pd.DataFrame(bool_gate_ref_results))
            else:
                # all other gate types process our labelled event DataFrame
                # first, preprocess events then apply gate
                df_events = self._preprocess_sample_events(sample, gate, cache_events=cache_events)
                gate_results = gate.apply(df_events)

            # Combine gate and parent results
            results[g_id, gate_path_str] = self._apply_parent_results(sample, gate, gate_results, parent_results)

            if isinstance(gate, fk_gates.QuadrantGate):
                for quad_res in results[g_id, gate_path_str].values():
                    quad_res['parent'] = p_id
                    quad_res['gate_path'] = g_path
            else:
                results[g_id, gate_path_str]['parent'] = p_id
                results[g_id, gate_path_str]['gate_path'] = g_path

        if not cache_events:
            self.clear_cache()

        return GatingResults(results, sample_id=sample_id)
