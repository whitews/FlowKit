import numpy as np
from lxml import etree
from .transforms import gml_transforms
import anytree
from anytree.exporter import DotExporter
from flowkit.resources import gml_schema
from flowkit import utils
# noinspection PyUnresolvedReferences
from flowkit.models.gate import \
    QuadrantGate, \
    RectangleGate, \
    BooleanGate, \
    PolygonGate, \
    EllipsoidGate


GATE_TYPES = [
    'RectangleGate',
    'PolygonGate',
    'EllipsoidGate',
    'QuadrantGate',
    'BooleanGate'
]


class Matrix(object):
    def __init__(
        self,
        matrix_element,
        xform_namespace,
        data_type_namespace
    ):
        self.id = utils.find_attribute_value(matrix_element, xform_namespace, 'id')
        self.fluorochomes = []
        self.detectors = []
        self.matrix = []

        fluoro_el = matrix_element.find(
            '%s:fluorochromes' % xform_namespace,
            namespaces=matrix_element.nsmap
        )

        fcs_dim_els = fluoro_el.findall(
            '%s:fcs-dimension' % data_type_namespace,
            namespaces=matrix_element.nsmap
        )

        for dim_el in fcs_dim_els:
            label = utils.find_attribute_value(dim_el, data_type_namespace, 'name')

            if label is None:
                raise ValueError(
                    'Dimension name not found (line %d)' % dim_el.sourceline
                )
            self.fluorochomes.append(label)

        detectors_el = matrix_element.find(
            '%s:detectors' % xform_namespace,
            namespaces=matrix_element.nsmap
        )

        fcs_dim_els = detectors_el.findall(
            '%s:fcs-dimension' % data_type_namespace,
            namespaces=matrix_element.nsmap
        )

        for dim_el in fcs_dim_els:
            label = utils.find_attribute_value(dim_el, data_type_namespace, 'name')

            if label is None:
                raise ValueError(
                    'Dimension name not found (line %d)' % dim_el.sourceline
                )
            self.detectors.append(label)

        spectrum_els = matrix_element.findall(
            '%s:spectrum' % xform_namespace,
            namespaces=matrix_element.nsmap
        )

        for spectrum_el in spectrum_els:
            matrix_row = []

            coefficient_els = spectrum_el.findall(
                '%s:coefficient' % xform_namespace,
                namespaces=matrix_element.nsmap
            )

            for co_el in coefficient_els:
                value = utils.find_attribute_value(co_el, xform_namespace, 'value')
                if value is None:
                    raise ValueError(
                        'Matrix coefficient must have only 1 value (line %d)' % co_el.sourceline
                    )

                matrix_row.append(float(value))

            self.matrix.append(matrix_row)

        self.matrix = np.array(self.matrix)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.id}, dims: {len(self.fluorochomes)})'
        )

    def apply(self, sample):
        pass


class GatingStrategy(object):
    """
    Represents an entire flow cytometry gating strategy, including instructions
    for compensation and transformation. Takes an optional, valid GatingML
    document as an input.
    """
    def __init__(self, gating_ml_file_path):
        self.gml_schema = gml_schema

        xml_document = etree.parse(gating_ml_file_path)

        val = self.gml_schema.validate(xml_document)

        if not val:
            raise ValueError("Document is not valid GatingML")

        self.parser = etree.XMLParser(schema=self.gml_schema)

        self._gating_ns = None
        self._data_type_ns = None
        self._transform_ns = None

        root = xml_document.getroot()

        namespace_map = root.nsmap

        # find GatingML target namespace in the map
        for ns, url in namespace_map.items():
            if url == 'http://www.isac-net.org/std/Gating-ML/v2.0/gating':
                self._gating_ns = ns
            elif url == 'http://www.isac-net.org/std/Gating-ML/v2.0/datatypes':
                self._data_type_ns = ns
            elif url == 'http://www.isac-net.org/std/Gating-ML/v2.0/transformations':
                self._transform_ns = ns

        self._gate_types = [
            ':'.join([self._gating_ns, gt]) for gt in GATE_TYPES
        ]

        # keys will be gate ID, value is the gate object itself
        self.gates = {}

        for gt in self._gate_types:
            gt_gates = root.findall(gt, namespace_map)

            for gt_gate in gt_gates:
                constructor = globals()[gt.split(':')[1]]
                g = constructor(
                    gt_gate,
                    self._gating_ns,
                    self._data_type_ns,
                    self
                )

                if g.id in self.gates:
                    raise ValueError(
                        "Gate '%s' already exists. "
                        "Duplicate gate IDs are not allowed." % g.id
                    )
                self.gates[g.id] = g

        # look for transformations
        self.transformations = {}

        if self._transform_ns is not None:
            # types of transforms include:
            #   - ratio
            #   - log10
            #   - asinh
            #   - hyperlog
            #   - linear
            #   - logicle
            xform_els = root.findall(
                '%s:transformation' % self._transform_ns,
                namespaces=namespace_map
            )

            for xform_el in xform_els:
                xform = None

                # determine type of transformation
                fratio_els = xform_el.findall(
                    '%s:fratio' % self._transform_ns,
                    namespaces=namespace_map
                )

                if len(fratio_els) > 0:
                    xform = gml_transforms.RatioGMLTransform(
                        xform_el,
                        self._transform_ns,
                        self._data_type_ns
                    )

                flog_els = xform_el.findall(
                    '%s:flog' % self._transform_ns,
                    namespaces=namespace_map
                )

                if len(flog_els) > 0:
                    xform = gml_transforms.LogGMLTransform(
                        xform_el,
                        self._transform_ns
                    )

                fasinh_els = xform_el.findall(
                    '%s:fasinh' % self._transform_ns,
                    namespaces=namespace_map
                )

                if len(fasinh_els) > 0:
                    xform = gml_transforms.AsinhGMLTransform(
                        xform_el,
                        self._transform_ns
                    )

                hyperlog_els = xform_el.findall(
                    '%s:hyperlog' % self._transform_ns,
                    namespaces=namespace_map
                )

                if len(hyperlog_els) > 0:
                    xform = gml_transforms.HyperlogGMLTransform(
                        xform_el,
                        self._transform_ns
                    )

                flin_els = xform_el.findall(
                    '%s:flin' % self._transform_ns,
                    namespaces=namespace_map
                )

                if len(flin_els) > 0:
                    xform = gml_transforms.LinearGMLTransform(
                        xform_el,
                        self._transform_ns
                    )

                logicle_els = xform_el.findall(
                    '%s:logicle' % self._transform_ns,
                    namespaces=namespace_map
                )

                if len(logicle_els) > 0:
                    xform = gml_transforms.LogicleGMLTransform(
                        xform_el,
                        self._transform_ns
                    )

                if xform is not None:
                    self.transformations[xform.id] = xform

        # look for comp matrices
        self.comp_matrices = {}

        if self._transform_ns is not None:
            # comp matrices are defined by the 'spectrumMatrix' element
            matrix_els = root.findall(
                '%s:spectrumMatrix' % self._transform_ns,
                namespaces=namespace_map
            )

            for matrix_el in matrix_els:
                matrix = Matrix(
                    matrix_el,
                    self._transform_ns,
                    self._data_type_ns
                )

                self.comp_matrices[matrix.id] = matrix

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{len(self.gates)} gates, {len(self.transformations)} transforms, '
            f'{len(self.comp_matrices)} compensations)'
        )

    def _build_hierarchy_tree(self):
        nodes = {}

        root = anytree.Node('root')

        for g_id, gate in self.gates.items():
            if gate.parent is not None:
                # we'll get children nodes after
                continue

            nodes[gate.id] = anytree.Node(
                gate.id,
                parent=root
            )

            if isinstance(gate, QuadrantGate):
                for q_id, quad in gate.quadrants.items():
                    nodes[q_id] = anytree.Node(
                        q_id,
                        parent=nodes[gate.id]
                    )

        for g_id, gate in self.gates.items():
            if gate.parent is None:
                # at root level, we already got it
                continue

            nodes[gate.id] = anytree.Node(
                gate.id,
                parent=nodes[gate.parent]
            )

            if isinstance(gate, QuadrantGate):
                for q_id, quad in gate.quadrants.items():
                    nodes[q_id] = anytree.Node(
                        q_id,
                        parent=nodes[gate.id]
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
                if isinstance(g, QuadrantGate):
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

    def gate_sample(self, sample, gate_id=None):
        """
        Apply a gate to a sample, returning a dictionary where gate ID is the
        key, and the value contains the event indices for events in the Sample
        which are contained by the gate. If the gate has a parent gate, all
        gates in the hierarchy above will be included in the results. If 'gate'
        is None, then all gates will be evaluated.

        :param sample: an FCS Sample instance
        :param gate_id: A gate ID or list of gate IDs to evaluate on given
            Sample. If None, all gates will be evaluated
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

        for g_id, gate in gates.items():
            results[g_id] = gate.apply(sample)

        return results
