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
        self._cached_preprocessed_events = {}

        # keys are the object's ID (xform or matrix),
        # values are the object itself
        self.transformations = {}
        self.comp_matrices = {}

        self._gate_tree = anytree.Node('root')

        # use a directed acyclic graph for later processing and enforcing there
        # are no cyclical gate relationships.
        # Each node is made from a tuple representing it's full gate path
        self._dag = nx.DiGraph()
        self._dag.add_node(('root',))

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
        :param gate_path: complete tuple of gate IDs for unique set of gate ancestors.
            Required if gate.gate_name and gate.parent combination is ambiguous

        :return: None
        """
        if not isinstance(gate, Gate):
            raise ValueError("gate must be a sub-class of the Gate class")

        parent_gate_name = gate.parent
        if parent_gate_name is None:
            # If no parent gate is specified, use root
            parent_gate_name = 'root'

        # Verify the gate parent matches the last item in the gate path (if given)
        if gate_path is not None:
            if len(gate_path) != 0:
                if parent_gate_name != gate_path[-1]:
                    raise ValueError("The gate parent and the last item in gate path are different.")

        # Find simple case of matching gate name + parent where no gate_path is specified.
        matched_nodes = anytree.findall(
            self._gate_tree,
            filter_=lambda g_node:
            g_node.name == gate.gate_name and
            g_node.parent.name == parent_gate_name
        )
        match_count = len(matched_nodes)

        if match_count != 0 and gate_path is None:
            raise KeyError(
                "A gate with name '%s' and parent gate name '%s' is already defined. " 
                "You must specify a gate_path as a unique tuple of ancestors." % (gate.gate_name, parent_gate_name)
            )

        # Here we either have a unique gate ID + parent, or an ambiguous gate ID + parent with a gate path.
        # It is still possible that the given gate_path already exists, we'll check that.
        # We'll find the parent node from the ID, and if there are multiple parent matches then resort
        # to using the given gate path.
        if parent_gate_name == 'root':
            # Easiest case since a root parent is also the full path.
            parent_node = self._gate_tree
        else:
            # Find all nodes with the parent ID name. If there's only one, we've identified the correct parent.
            # If there are none, the parent doesn't exist (or is a Quad gate).
            # If there are >1, we have to compare the gate paths.
            matching_parent_nodes = anytree.search.findall_by_attr(self._gate_tree, parent_gate_name)
            matched_parent_count = len(matching_parent_nodes)
            match_idx = None

            if matched_parent_count == 0:
                # TODO: need to double-check whether this scenario could be in a quadrant gate
                raise ValueError("Parent gate %s does not exist in the gating strategy" % parent_gate_name)
            elif matched_parent_count == 1:
                # There's only one match for the parent, so we're done
                match_idx = 0
            elif matched_parent_count > 1:
                for i, matched_parent_node in enumerate(matching_parent_nodes):
                    matched_parent_ancestors = tuple((pn.name for pn in matched_parent_node.path))
                    if matched_parent_ancestors == gate_path:
                        match_idx = i
                        break

            # look up the parent node, then do one final check to make sure the new gate doesn't
            # already exist as a child of the parent
            parent_node = matching_parent_nodes[match_idx]
            parent_child_nodes = anytree.search.findall_by_attr(parent_node, gate.gate_name, maxlevel=1)
            if len(parent_child_nodes) > 0:
                raise ValueError(
                    "A gate already exists matching gate name %s and the specified gate path" % gate.gate_name
                )

        node = anytree.Node(gate.gate_name, parent=parent_node, gate=gate)
        parent_node_tuple = tuple(n.name for n in parent_node.path)
        new_node_tuple = parent_node_tuple + (gate.gate_name,)
        self._dag.add_node(new_node_tuple)
        self._dag.add_edge(parent_node_tuple, new_node_tuple)

        # Quadrant gates need special handling to add their individual quadrants as children.
        # Other gates cannot have the main quadrant gate as a parent, they can only reference
        # the individual quadrants as parents.
        if isinstance(gate, fk_gates.QuadrantGate):
            for q_id, q in gate.quadrants.items():
                anytree.Node(q_id, parent=node, gate=q)

                q_node_tuple = new_node_tuple + (q_id,)

                self._dag.add_node(q_node_tuple)
                self._dag.add_edge(new_node_tuple, q_node_tuple)

        if isinstance(gate, fk_gates.BooleanGate):
            bool_gate_refs = gate.gate_refs
            for gate_ref in bool_gate_refs:
                gate_ref_node_tuple = tuple(gate_ref['path']) + (gate_ref['ref'],)
                self._dag.add_edge(gate_ref_node_tuple, new_node_tuple)

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

    def _get_gate_node(self, gate_name, gate_path=None):
        node_matches = anytree.findall_by_attr(self._gate_tree, gate_name)
        node_match_count = len(node_matches)

        if node_match_count == 1:
            node = node_matches[0]
        elif node_match_count > 1:
            # need to match on full gate path
            # TODO: what if QuadrantGate is re-used, does this still work for that case
            if gate_path is None:
                raise ValueError(
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
            raise ValueError("Gate name %s was not found in gating strategy" % gate_name)

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

    def get_gate(self, gate_name, gate_path=None):
        """
        Retrieve a gate instance by its gate ID.

        :param gate_name: text string of a gate name
        :param gate_path: complete tuple of gate IDs for unique set of gate ancestors.
            Required if gate_name is ambiguous
        :return: Subclass of a Gate object
        :raises KeyError: if gate ID is not found in gating strategy
        """
        node = self._get_gate_node(gate_name, gate_path)

        if isinstance(node.gate, fk_gates.Quadrant):
            # return the full QuadrantGate b/c a Quadrant by itself has no parent reference
            node = node.parent

        return node.gate

    def get_parent_gate(self, gate_name, gate_path=None):
        """
        Retrieve the parent Gate instance for the given gate ID.

        :param gate_name: text string of a gate name
        :param gate_path: complete tuple of gate IDs for unique set of gate ancestors.
            Required if gate_name is ambiguous
        :return: Subclassed Gate instance
        """
        node = self._get_gate_node(gate_name, gate_path)

        if node.parent.name == 'root':
            return None

        return node.parent.gate

    def get_child_gates(self, gate_name, gate_path=None):
        """
        Retrieve list of child gate instances by their parent's gate ID.

        :param gate_name: text string of a gate name
        :param gate_path: complete tuple of gate IDs for unique set of gate ancestors.
            Required if gate_name is ambiguous
        :return: list of Gate instances
        :raises KeyError: if gate ID is not found in gating strategy
        """
        node = self._get_gate_node(gate_name, gate_path)

        child_gates = []

        for n in node.children:
            child_gates.append(n.gate)

        return child_gates

    def get_gate_ids(self):
        """
        Retrieve the list of gate IDs (with ancestors) for the gating strategy

        :return: list of tuples (1st item is gate name string, 2nd item is a list of ancestor gate names)
        """
        gates = []
        for node in self._gate_tree.descendants:
            ancestors = tuple((a.name for a in node.ancestors))
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

    def _get_cached_preprocessed_events(self, sample_id, comp_ref, xform_ref, dim_idx=None):
        """
        Retrieve cached pre-processed events (if they exist)

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
            sample.original_filename,
            comp_ref,
            None,
            None
        )

        if events is not None:
            return events

        if comp_ref == 'FCS':
            meta = sample.get_metadata()

            if 'spill' not in meta or 'spillover' not in meta:
                # GML 2.0 spec states if 'FCS' is specified but no spill is present, treat as uncompensated
                events = sample.get_events(source='raw')
                return events.copy()

            try:
                spill = meta['spillover']  # preferred, per FCS standard
            except KeyError:
                spill = meta['spill']

            detectors = [sample.pnn_labels[i] for i in sample.fluoro_indices]
            fluorochromes = [sample.pns_labels[i] for i in sample.fluoro_indices]
            matrix = Matrix('fcs', spill, detectors, fluorochromes, null_channels=sample.null_channels)
        else:
            # lookup specified comp-ref in gating strategy
            matrix = self.comp_matrices[comp_ref]

        if matrix is not None:
            events = matrix.apply(sample)
            # cache the comp events
            # noinspection PyProtectedMember
            self._cache_preprocessed_events(
                events,
                sample.original_filename,
                comp_ref,
                None,
                None
            )

        return events

    def _preprocess_sample_events(self, sample, gate, cache_events=False):
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
                    matrix_dim_idx = matrix.fluorochomes.index(dim_id)
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
                        sample.original_filename,
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
                            sample.original_filename,
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
                    'sample': sample.original_filename,
                    'events': q_result,
                    'count': q_event_count,
                    'absolute_percent': (q_event_count / float(sample.event_count)) * 100.0,
                    'relative_percent': (q_event_count / float(parent_count)) * 100.0,
                    'parent': gate.parent,
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
                'sample': sample.original_filename,
                'events': results_and_parent,
                'count': event_count,
                'absolute_percent': (event_count / float(sample.event_count)) * 100.0,
                'relative_percent': relative_percent,
                'parent': gate.parent,
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

            gate = self.get_gate(g_id, g_path)

            # get_gate returns QuadrantGate when given on of its quadrant, let's check if we have
            # just a quadrant or the full QuadrantGate
            if isinstance(gate, fk_gates.QuadrantGate):
                if g_id in gate.quadrants.keys():
                    # This is a quadrant sub-gate, we'll process the quadrant sub-gates
                    # all at once with the main QuadrantGate ID
                    continue

            if verbose:
                print("%s: processing gate %s" % (sample.original_filename, g_id))

            # look up parent results
            parent_results = None  # default to None
            if gate.parent is not None:
                parent_gate = self.get_gate(p_id, p_path)
                if p_uid in results:
                    parent_results = results[p_uid]
                elif isinstance(parent_gate, fk_gates.QuadrantGate):
                    # need to check for quadrant results in a quadrant gate
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
                    gate_ref_gate = self.get_gate(gate_ref['ref'], gate_ref['path'])
                    gate_ref_res_key = (gate_ref['ref'], "/".join(gate_ref['path']))

                    if isinstance(gate_ref_gate, fk_gates.QuadrantGate):
                        quad_gate_name = gate_ref_gate.gate_name
                        quad_gate_path = gate_ref['path'][:-1]

                        quad_gate_res_key = (quad_gate_name, "/".join(quad_gate_path))
                        quad_gate_results = results[quad_gate_res_key]

                        # but the quadrant result is what we're after
                        bool_gate_ref_results[gate_ref_res_key] = quad_gate_results[gate_ref['ref']]['events']
                    else:
                        bool_gate_ref_results[gate_ref_res_key] = results[gate_ref_res_key]['events']

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
                    quad_res['gate_path'] = g_path
            else:
                results[g_id, gate_path_str]['gate_path'] = g_path

        if not cache_events:
            self.clear_cache()

        return GatingResults(results, sample_id=sample.original_filename)
