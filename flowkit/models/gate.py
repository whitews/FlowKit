import pkgutil
import os
# noinspection PyUnresolvedReferences
from lxml import etree, objectify

loader = pkgutil.get_loader('flowkit.resources')
resource_path = os.path.dirname(loader.path)
gating_ml_xsd = os.path.join(resource_path, 'Gating-ML.v2.0.xsd')


GATE_TYPES = [
    'RectangleGate'
]


class Dimension(object):
    def __init__(self, dim_element, gating_namespace, data_type_namespace):
        self.compensation_ref = str(
            dim_element.xpath(
                '@%s:compensation-ref' % gating_namespace,
                namespaces=dim_element.nsmap
            )[0]
        )

        self.min = None
        self.max = None

        # should be 0 or only 1 'min' attribute, but xpath returns a list, so...
        min_attribs = self.id = dim_element.xpath(
            '@%s:min' % gating_namespace,
            namespaces=dim_element.nsmap
        )

        if len(min_attribs) > 0:
            self.min = float(min_attribs[0])

        # ditto for 'max' attribute, 0 or 1 value
        max_attribs = self.id = dim_element.xpath(
            '@%s:max' % gating_namespace,
            namespaces=dim_element.nsmap
        )

        if len(max_attribs) > 0:
            self.max = float(max_attribs[0])

        # label be here
        fcs_dim_els = dim_element.find(
            '%s:fcs-dimension' % data_type_namespace,
            namespaces=dim_element.nsmap
        )

        label_attribs = fcs_dim_els.xpath(
            '@%s:name' % data_type_namespace,
            namespaces=dim_element.nsmap
        )

        if len(label_attribs) > 0:
            self.label = label_attribs[0]
        else:
            raise ValueError(
                'Dimension name not found (line %d)' % fcs_dim_els.sourceline
            )


class Gate(object):
    """
    Represents a single flow cytometry gate
    """
    def __init__(self, gate_element, gating_namespace, data_type_namespace):
        self.id = gate_element.xpath(
            '@%s:id' % gating_namespace,
            namespaces=gate_element.nsmap
        )[0]
        parent = gate_element.xpath(
            '@%s:parent' % gating_namespace,
            namespaces=gate_element.nsmap
        )
        if len(parent) == 0:
            self.parent = None
        else:
            self.parent = parent[0]

        dim_els = gate_element.findall(
            '%s:dimension' % gating_namespace,
            namespaces=gate_element.nsmap
        )

        self.dimensions = []

        for dim_el in dim_els:
            dim = Dimension(dim_el, gating_namespace, data_type_namespace)
            self.dimensions.append(dim)


class RectangleGate(Gate):
    """
    Represents a GatingML Rectangle Gate

    A RectangleGate can have one or more dimensions, and each dimension must
    specify at least one of a minimum or maximum value (or both). From the
    GatingML specification (sect. 5.1.1):

        Rectangular gates are used to express range gates (n = 1, i.e., one
        dimension), rectangle gates (n = 2, i.e., two dimensions), box regions
        (n = 3, i.e., three dimensions), and hyper-rectangular regions
        (n > 3, i.e., more than three dimensions).
    """
    def __init__(self, gate_element, gating_namespace, data_type_namespace):
        super().__init__(gate_element, gating_namespace, data_type_namespace)
        pass


class GatingStrategy(object):
    """
    Represents an entire flow cytometry gating strategy, including instructions
    for compensation and transformation. Takes an optional, valid GatingML
    document as an input.
    """
    def __init__(self, gating_ml_file_path):
        self.gml_schema = etree.XMLSchema(etree.parse(gating_ml_xsd))

        xml_document = etree.parse(gating_ml_file_path)

        val = self.gml_schema.validate(xml_document)

        print(val)

        self.parser = etree.XMLParser(schema=self.gml_schema)

        self._gating_ns = None
        self._data_type_ns = None

        root = xml_document.getroot()

        namespace_map = root.nsmap

        # find GatingML target namespace in the map
        for ns, url in namespace_map.items():
            if url == 'http://www.isac-net.org/std/Gating-ML/v2.0/gating':
                self._gating_ns = ns
            elif url == 'http://www.isac-net.org/std/Gating-ML/v2.0/datatypes':
                self._data_type_ns = ns

        self._gate_types = [
            ':'.join([self._gating_ns, gt]) for gt in GATE_TYPES
        ]

        self.gates = []

        for gt in self._gate_types:
            gt_gates = root.findall(gt, namespace_map)

            for gt_gate in gt_gates:
                constructor = globals()[gt.split(':')[1]]
                g = constructor(gt_gate, self._gating_ns, self._data_type_ns)

                self.gates.append(g)

    def gate_sample(self, sample, gate):
        pass
