"""
GatingStrategy & GatingResults classes
"""
import json
import anytree
from anytree.exporter import DotExporter
import pandas as pd
from .._models.gates._base_gate import Gate
from .._models import gates as fk_gates
# noinspection PyProtectedMember
from .._models.transforms._base_transform import Transform
# noinspection PyProtectedMember
from .._models.transforms._matrix import Matrix


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

    def add_gate(self, gate):
        """
        Add a gate to the gating strategy, see `gates` module. The gate ID must be unique in the gating strategy.

        :param gate: instance from a sub-class of the Gate class
        :return: None
        """

        if not isinstance(gate, Gate):
            raise ValueError("gate must be a sub-class of the Gate class")

        parent_id = gate.parent
        if parent_id is None:
            parent_id = 'root'

        # TODO: check for uniqueness of gate ID + gate path combo
        matched_nodes = anytree.findall(
            self._gate_tree,
            filter_=lambda g_node:
                g_node.name == gate.id and
                g_node.parent.name == parent_id
        )
        if len(matched_nodes) != 0:
            raise KeyError("Gate ID '%s' is already defined" % gate.id)

        parent_id = gate.parent
        if parent_id is None:
            parent_node = self._gate_tree
        else:
            matching_nodes = anytree.search.findall_by_attr(self._gate_tree, parent_id)

            if len(matching_nodes) == 0:
                # TODO: could be in a quadrant gate
                raise ValueError("Parent gate %s does not exist in the gating strategy" % parent_id)
            elif len(matching_nodes) > 1:
                raise ValueError("Multiple gates exist matching parent ID %s, specify full gate path" % parent_id)

            parent_node = matching_nodes[0]

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

    def get_gate(self, gate_id, gate_path=None):
        """
        Retrieve a gate instance from its gate ID. reference gates in their parent gating
        strategy.

        :param gate_id: text string of a gate ID
        :param gate_path: complete list of gate IDs for unique set of gate ancestors. Required if gate_id is ambiguous
        :return: Subclass of a Gate object
        :raises KeyError: if gate ID is not found in gating strategy
        """
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

        return node.gate

    def get_parent_gate(self, gate_id):
        """
        Retrieve the gate ID for a parent gate of the given gate ID.

        :param gate_id: text string of a gate ID
        :return: text string of the parent gate ID, None if given gate reference has no parent gate.
        """
        n = anytree.find_by_attr(self._gate_tree, gate_id)

        if n is None:
            raise ValueError("Gate ID %s was not found in gating strategy" % gate_id)

        return n.parent.gate

    def get_gate_ids(self):
        """
        Retrieve the list of gate IDs for the gating strategy
        :return: list of gate ID strings
        """
        return [node.name for node in self._gate_tree.descendants]

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
        Apply a gate to a sample, returning a dictionary where gate ID is the
        key, and the value contains the event indices for events in the Sample
        which are contained by the gate. If the gate has a parent gate, all
        gates in the hierarchy above will be included in the results. If 'gate'
        is None, then all gates will be evaluated.

        :param sample: an FCS Sample instance
        :param gate_id: A gate ID or list of gate IDs to evaluate on given
            Sample. If None, all gates will be evaluated
        :param verbose: If True, print a line for each gate processed
        :return: Dictionary where keys are gate IDs, values are boolean arrays
            of length matching the number sample events. Events in the gate are
            True.
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
                print("%s: processing gate %s" % (sample, g_id))
            if gate.parent is not None and gate.parent in results:
                parent_results = results[gate.parent]
            else:
                parent_results = None
            # to make the dict key unique, make a string from the ancestors,
            # but also add the set of them as a set to avoid repeated & fragile
            # string splitting in the GatingResults class
            gate_path_str = "/".join([a.name for a in item.ancestors])
            results[g_id, gate_path_str] = gate.apply(sample, parent_results, self)

            gate_path_list = [a.name for a in item.ancestors]
            if isinstance(gate, fk_gates.QuadrantGate):
                for quad_res in results[g_id, gate_path_str].values():
                    quad_res['gate_path'] = gate_path_list
            else:
                results[g_id, gate_path_str]['gate_path'] = gate_path_list

        return GatingResults(results, sample_id=sample.original_filename)


class GatingResults(object):
    """
    A GatingResults instance is returned from the GatingStrategy `gate_samples` method
    as well as the Session `get_gating_results` method. End users will never create an
    instance of GatingResults directly, only via these GatingStrategy and Session
    methods. However, there are several GatingResults methods to retrieve the results.
    """
    def __init__(self, results_dict, sample_id):
        self._raw_results = results_dict
        self._gate_lut = {}
        self.report = None
        self.sample_id = sample_id
        self._process_results()

    @staticmethod
    def _get_pd_result_dict(res_dict, gate_id):
        return {
            'sample': res_dict['sample'],
            'gate_path': res_dict['gate_path'],
            'parent': res_dict['parent'],
            'gate_id': gate_id,
            'gate_type': res_dict['gate_type'],
            'count': res_dict['count'],
            'absolute_percent': res_dict['absolute_percent'],
            'relative_percent': res_dict['relative_percent'],
            'quadrant_parent': None
        }

    def _process_results(self):
        pd_list = []

        for (g_id, g_path), res in self._raw_results.items():
            if 'events' not in res:
                # it's a quad gate with sub-gates
                for sub_g_id, sub_res in res.items():
                    pd_dict = self._get_pd_result_dict(sub_res, sub_g_id)
                    pd_dict['quadrant_parent'] = g_id
                    pd_list.append(pd_dict)
                    if sub_g_id not in self._gate_lut:
                        self._gate_lut[sub_g_id] = {
                            'paths': [g_path]
                        }
                    else:
                        self._gate_lut[sub_g_id]['paths'].append(g_path)
            else:
                pd_list.append(self._get_pd_result_dict(res, g_id))
                if g_id not in self._gate_lut:
                    self._gate_lut[g_id] = {
                        'paths': [g_path]
                    }
                else:
                    self._gate_lut[g_id]['paths'].append(g_path)

        df = pd.DataFrame(
            pd_list,
            columns=[
                'sample',
                'gate_path',
                'gate_id',
                'gate_type',
                'quadrant_parent',
                'parent',
                'count',
                'absolute_percent',
                'relative_percent'
            ]
        )
        df['level'] = df.gate_path.map(len)

        # ???: sorting by non-index column will result in Pandas PerformanceWarning
        #   when looking up rows using .loc. The hit in this case is minimal and we
        #   really want the DataFrame sorted by 'level' for better readability.
        #   Maybe consider not setting a MultiIndex for this?
        self.report = df.set_index(['sample', 'gate_id']).sort_index().sort_values('level')

    def get_gate_indices(self, gate_id, gate_path=None):
        """
        Retrieve a boolean array indicating gate membership for the events in the GatingResults sample.
        Note, the same gate ID may be found in multiple gate paths, i.e. the gate ID can be ambiguous.
        In this case, specify the full gate path to retrieve gate indices.

        :param gate_id: text string of a gate ID
        :param gate_path: A list of ancestor gate IDs for the given gate ID. Alternatively, a string path delimited
            by forward slashes can also be given, e.g. ('/root/singlets/lymph/live')
        :return: NumPy boolean array (length of sample event count)
        """
        gate_paths = self._gate_lut[gate_id]['paths']
        if len(gate_paths) > 1:
            if gate_path is None:
                raise ValueError("Gate ID %s is ambiguous, specify the full gate path")
            elif isinstance(gate_path, list):
                gate_path = "/".join(gate_path)
        else:
            gate_path = gate_paths[0]

        gate_series = self.report.loc[(self.sample_id, gate_id)]
        if isinstance(gate_series, pd.DataFrame):
            gate_series = gate_series.iloc[0]

        quad_parent = gate_series['quadrant_parent']

        if quad_parent is not None:
            return self._raw_results[quad_parent, gate_path][gate_id]['events']
        else:
            return self._raw_results[gate_id, gate_path]['events']

    def get_gate_count(self, gate_id):
        """
        Retrieve event count for the specified gate ID for the gating results sample

        :param gate_id: text string of a gate ID
        :return: integer count of events in gate ID
        """
        return self.report.loc[(self.sample_id, gate_id), 'count']

    def get_gate_absolute_percent(self, gate_id):
        """
        Retrieve percent of events, relative to the total sample events, of the specified gate ID for the
        gating results sample

        :param gate_id: text string of a gate ID
        :return: floating point number of the absolute percent
        """
        return self.report.loc[(self.sample_id, gate_id), 'absolute_percent']

    def get_gate_relative_percent(self, gate_id):
        """
        Retrieve percent of events, relative to parent gate, of the specified gate ID for the gating results sample

        :param gate_id: text string of a gate ID
        :return: floating point number of the relative percent
        """
        return self.report.loc[(self.sample_id, gate_id), 'relative_percent']
