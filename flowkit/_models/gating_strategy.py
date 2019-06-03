import anytree
from anytree.exporter import DotExporter
import pandas as pd
from flowkit import _gml_utils
from flowkit._models.gates.gml_gates import GMLQuadrantGate


class GatingStrategy(object):
    """
    Represents an entire flow cytometry gating strategy, including instructions
    for compensation and transformation. Takes an optional, valid GatingML
    document as an input.
    """
    def __init__(self, gating_ml_file_path=None):
        self.gating_ns = None
        self.data_type_ns = None
        self.transform_ns = None
        self._cached_compensations = {}

        # keys are the object's ID (gate, xform, or matrix,
        # values are the object itself
        self.gates = {}
        self.transformations = {}
        self.comp_matrices = {}

        if gating_ml_file_path is not None:
            root_gml, gating_ns, dt_ns, xform_ns = _gml_utils.parse_gatingml_file(gating_ml_file_path)
            self.gating_ns = gating_ns
            self.data_type_ns = dt_ns
            self.transform_ns = xform_ns

            self.gates = _gml_utils.construct_gates(self, root_gml)
            self.transformations = _gml_utils.construct_transforms(root_gml, self.transform_ns, self.data_type_ns)
            self.comp_matrices = _gml_utils.construct_matrices(root_gml, self.transform_ns, self.data_type_ns)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{len(self.gates)} gates, {len(self.transformations)} transforms, '
            f'{len(self.comp_matrices)} compensations)'
        )

    def _build_hierarchy_tree(self, gates=None):
        nodes = {}

        root = anytree.Node('root')
        parent_gates = set()

        if gates is None:
            gates = self.gates

        for g_id, gate in gates.items():
            if gate.parent is not None:
                # record the set of parents so we can find the leaves later
                parent_gates.add(gate.parent)

                # we'll get children nodes after
                continue

            nodes[gate.id] = anytree.Node(
                gate.id,
                parent=root
            )

            if isinstance(gate, GMLQuadrantGate):
                for q_id, quad in gate.quadrants.items():
                    nodes[q_id] = anytree.Node(
                        q_id,
                        parent=nodes[gate.id]
                    )

        parent_gates.difference_update(set(nodes.keys()))
        # Now we have all the root-level nodes and the set of parents.
        # Process the parent gates
        parents_remaining = len(parent_gates)
        while parents_remaining > 0:
            discard = []
            for gate_id in parent_gates:
                gate = self.get_gate_by_reference(gate_id)
                parent_id = gate.parent
                if parent_id in nodes or parent_id is None:
                    nodes[gate_id] = anytree.Node(
                        gate_id,
                        parent=root if parent_id is None else nodes[parent_id]
                    )

                    if isinstance(gate, GMLQuadrantGate):
                        if gate.id not in nodes:
                            nodes[gate.id] = anytree.Node(
                                gate.id,
                                parent=root if parent_id is None else nodes[parent_id]
                            )
                        for q_id, quad in gate.quadrants.items():
                            nodes[q_id] = anytree.Node(
                                q_id,
                                parent=nodes[gate.id]
                            )
                    discard.append(gate_id)
            parent_gates.difference_update(set(discard))
            parents_remaining = len(parent_gates)

        # Process remaining leaves
        for g_id, gate in gates.items():
            if g_id in nodes:
                continue

            nodes[g_id] = anytree.Node(
                g_id,
                parent=nodes[gate.parent]
            )

            if isinstance(gate, GMLQuadrantGate):
                for q_id, quad in gate.quadrants.items():
                    nodes[q_id] = anytree.Node(
                        q_id,
                        parent=nodes[g_id]
                    )

        return root

    def get_gate_by_reference(self, gate_id):
        """
        For gates to lookup any reference gates in their parent gating
        strategy. It's not safe to just look at the gates dictionary as
        QuadrantGate IDs cannot be parents themselves, only their component
        Quadrant IDs can be parents.
        :return: Subclass of a Gate object
        """
        try:
            gate = self.gates[gate_id]
        except KeyError as e:
            # may be in a Quadrant gate
            gate = None
            for g_id, g in self.gates.items():
                if isinstance(g, GMLQuadrantGate):
                    if gate_id in g.quadrants:
                        gate = g
                        continue
            if gate is None:
                raise e

        return gate

    def get_gate_hierarchy(self, output='ascii'):
        """
        Show hierarchy of gates in multiple formats, including text,
        dictionary, or JSON,

        :param output: Determines format of hierarchy returned, either 'text',
            'dict', or 'JSON' (default is 'text')
        :return: either a text string or a dictionary
        """
        root = self._build_hierarchy_tree()

        tree = anytree.RenderTree(root, style=anytree.render.ContRoundStyle())

        if output == 'ascii':
            lines = []

            for row in tree:
                lines.append("%s%s" % (row.pre, row.node.name))

            return "\n".join(lines)
        elif output == 'json':
            exporter = anytree.exporter.JsonExporter()
            gs_json = exporter.export(root)

            return gs_json
        elif output == 'dict':
            exporter = anytree.exporter.DictExporter()
            gs_dict = exporter.export(root)

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
        root = self._build_hierarchy_tree()
        DotExporter(root).to_picture(output_file_path)

    def get_cached_compensation(self, sample, comp_ref):
        try:
            # return a copy of cached events in case downstream modifies them
            return self._cached_compensations[sample.original_filename][comp_ref].copy()
        except KeyError:
            return None

    def cache_compensated_events(self, sample, comp_ref, comp_events):
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
        if gate_id is None:
            gates = self.gates
        elif isinstance(gate_id, list):
            gates = {}
            for g_id in gate_id:
                gates[g_id] = self.gates[g_id]
        else:
            gates = {gate_id: self.gates[gate_id]}

        results = {}

        # anytree tree allows us to iterate from the root down to the leaves
        # in an order that follows the hierarchy, thereby avoiding duplicate
        # processing of parent gates
        root = self._build_hierarchy_tree(gates)
        tree = anytree.RenderTree(root)

        for item in tree:
            g_id = item.node.name
            if g_id == 'root':
                continue
            gate = self.get_gate_by_reference(g_id)
            if isinstance(gate, GMLQuadrantGate) and g_id in gate.quadrants:
                # This is a sub-gate, we'll process the sub-gates all at once
                # with the main QuadrantGate ID
                continue

            if verbose:
                print("%s: processing gate %s" % (sample, g_id))
            if gate.parent is not None and gate.parent in results:
                parent_results = results[gate.parent]
            else:
                parent_results = None
            results[g_id] = gate.apply(sample, parent_results)

        return GatingResults(results, sample_id=sample.original_filename)


class GatingResults(object):
    def __init__(self, results_dict, sample_id):
        self._raw_results = results_dict
        self.report = None
        self.sample_id = sample_id
        self._process_results()

    @staticmethod
    def _get_pd_result_dict(res_dict, gate_id):
        return {
            'sample': res_dict['sample'],
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

        for g_id, res in self._raw_results.items():
            if 'events' not in res:
                # it's a quad gate with sub-gates
                for sub_g_id, sub_res in res.items():
                    pd_dict = self._get_pd_result_dict(sub_res, sub_g_id)
                    pd_dict['quadrant_parent'] = g_id
                    pd_list.append(pd_dict)
            else:
                pd_list.append(self._get_pd_result_dict(res, g_id))

        df = pd.DataFrame(
            pd_list,
            columns=[
                'sample',
                'gate_id',
                'gate_type',
                'quadrant_parent',
                'parent',
                'count',
                'absolute_percent',
                'relative_percent'
            ]
        )

        self.report = df.set_index(['sample', 'gate_id']).sort_index()

    def get_gate_indices(self, gate_id):
        gate_series = self.report.loc[(self.sample_id, gate_id)]
        if isinstance(gate_series, pd.DataFrame):
            gate_series = gate_series.iloc[0]

        quad_parent = gate_series['quadrant_parent']

        if quad_parent is not None:
            return self._raw_results[quad_parent][gate_id]['events']
        else:
            return self._raw_results[gate_id]['events']

    def get_gate_count(self, gate_id):
        return self.report.loc[(self.sample_id, gate_id), 'count']

    def get_gate_absolute_percent(self, gate_id):
        return self.report.loc[(self.sample_id, gate_id), 'absolute_percent']

    def get_gate_relative_percent(self, gate_id):
        return self.report.loc[(self.sample_id, gate_id), 'relative_percent']
