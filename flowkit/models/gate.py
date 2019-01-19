import pkgutil
import os
from abc import ABC, abstractmethod
# noinspection PyUnresolvedReferences
from lxml import etree, objectify
import numpy as np
from flowkit import utils
import flowutils

loader = pkgutil.get_loader('flowkit.resources')
resource_path = os.path.dirname(loader.path)
gating_ml_xsd = os.path.join(resource_path, 'Gating-ML.v2.0.xsd')


GATE_TYPES = [
    'RectangleGate',
    'PolygonGate',
    'EllipsoidGate'
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


class Vertex(object):
    def __init__(self, vert_element, gating_namespace, data_type_namespace):
        self.coordinates = []

        coord_els = vert_element.findall(
            '%s:coordinate' % gating_namespace,
            namespaces=vert_element.nsmap
        )

        if len(coord_els) != 2:
            raise ValueError(
                'Vertex must contain 2 coordinate values (line %d)' % vert_element.sourceline
            )

        # should be 0 or only 1 'min' attribute, but xpath returns a list, so...
        for coord_el in coord_els:
            value_attribs = coord_el.xpath(
                '@%s:value' % data_type_namespace,
                namespaces=vert_element.nsmap
            )
            if len(value_attribs) != 1:
                raise ValueError(
                    'Vertex coordinate must have only 1 value (line %d)' % coord_el.sourceline
                )

            self.coordinates.append(float(value_attribs[0]))


class Gate(ABC):
    """
    Represents a single flow cytometry gate
    """
    def __init__(
            self,
            gate_element,
            gating_namespace,
            data_type_namespace,
            gating_strategy
    ):
        self.__parent__ = gating_strategy
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

    @abstractmethod
    def apply(self, sample):
        pass

    @staticmethod
    def compensate_sample(dim_comp_refs, sample):
        events = sample.get_raw_events()

        spill = None
        if len(dim_comp_refs) > 1:
            raise NotImplementedError(
                "Mixed compensation between individual channels is not "
                "implemented. Never seen it, but if you are reading this "
                "message, submit an issue to have it implemented."
            )
        elif len(dim_comp_refs) == 1 and 'FCS' in dim_comp_refs:
            meta = sample.get_metadata()
            if 'spill' not in meta or 'spillover' not in meta:
                pass
            elif 'spillover' in meta:  # preferred, per FCS standard
                spill = meta['spillover']
            elif 'spill' in meta:
                spill = meta['spill']
        else:
            # TODO: implement lookup in parent for specified comp-ref
            pass

        if spill is not None:
            events = events.copy()
            spill = utils.parse_compensation_matrix(
                spill,
                sample.pnn_labels,
                null_channels=sample.null_channels
            )
            indices = spill[0, :]  # headers are channel #'s
            indices = [int(i - 1) for i in indices]
            comp_matrix = spill[1:, :]  # just the matrix
            events = flowutils.compensate.compensate(
                events,
                comp_matrix,
                indices
            )

        return events


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
    def __init__(
            self,
            gate_element,
            gating_namespace,
            data_type_namespace,
            gating_strategy
    ):
        super().__init__(
            gate_element,
            gating_namespace,
            data_type_namespace,
            gating_strategy
        )
        pass

    def apply(self, sample):
        events = sample.get_raw_events()
        pnn_labels = sample.pnn_labels

        if events.shape[1] != len(pnn_labels):
            raise ValueError(
                "Number of FCS dimensions (%d) does not match label count (%d)"
                % (events.shape[1], len(pnn_labels))
            )

        dim_idx = []
        dim_min = []
        dim_max = []
        dim_comp_refs = set()

        for dim in self.dimensions:
            if dim.compensation_ref not in [None, 'uncompensated']:
                dim_comp_refs.add(dim.compensation_ref)

            if dim.min is None and dim.max is None:
                raise ValueError(
                    "Gate '%s' does not include a min or max value" % self.id
                )

            dim_idx.append(pnn_labels.index(dim.label))
            dim_min.append(dim.min)
            dim_max.append(dim.max)

        events = self.compensate_sample(dim_comp_refs, sample)

        results = np.ones(events.shape[0], dtype=np.bool)

        for i, d_idx in enumerate(dim_idx):
            if dim_min[i] is not None:
                results = np.bitwise_and(results, events[:, d_idx] >= dim_min[i])
            if dim_max[i] is not None:
                results = np.bitwise_and(results, events[:, d_idx] < dim_max[i])

        return results


class PolygonGate(Gate):
    """
    Represents a GatingML Polygon Gate

    A PolygonGate must have exactly 2 dimensions, and must specify at least
    three vertices. Polygons can have crossing boundaries, and interior regions
    are defined by the winding number method:
        https://en.wikipedia.org/wiki/Winding_number
    """
    def __init__(
            self,
            gate_element,
            gating_namespace,
            data_type_namespace,
            gating_strategy
    ):
        super().__init__(
            gate_element,
            gating_namespace,
            data_type_namespace,
            gating_strategy
        )
        vert_els = gate_element.findall(
            '%s:vertex' % gating_namespace,
            namespaces=gate_element.nsmap
        )

        self.vertices = []

        for vert_el in vert_els:
            vert = Vertex(vert_el, gating_namespace, data_type_namespace)
            self.vertices.append(vert)

    def apply(self, sample):
        events = sample.get_raw_events()
        pnn_labels = sample.pnn_labels

        if events.shape[1] != len(pnn_labels):
            raise ValueError(
                "Number of FCS dimensions (%d) does not match label count (%d)"
                % (events.shape[1], len(pnn_labels))
            )

        dim_idx = []
        dim_comp_refs = set()

        for dim in self.dimensions:
            if dim.compensation_ref not in [None, 'uncompensated']:
                dim_comp_refs.add(dim.compensation_ref)

            dim_idx.append(pnn_labels.index(dim.label))

        events = self.compensate_sample(dim_comp_refs, sample)

        path_verts = []

        for vert in self.vertices:
            path_verts.append(vert.coordinates)

        results = utils.points_in_polygon(path_verts, events[:, dim_idx])

        return results


class EllipsoidGate(Gate):
    """
    Represents a GatingML Ellipsoid Gate

    An EllipsoidGate must have at least 2 dimensions, and must specify a mean
    value (center of the ellipsoid), a covariance matrix, and a distance
    square (the square of the Mahalanobis distance).
    """
    def __init__(
            self,
            gate_element,
            gating_namespace,
            data_type_namespace,
            gating_strategy
    ):
        super().__init__(
            gate_element,
            gating_namespace,
            data_type_namespace,
            gating_strategy
        )

        # First, we'll get the center of the ellipse, contained in
        # a 'mean' element, that holds 2 'coordinate' elements
        mean_el = gate_element.find(
            '%s:mean' % gating_namespace,
            namespaces=gate_element.nsmap
        )

        self.coordinates = []

        coord_els = mean_el.findall(
            '%s:coordinate' % gating_namespace,
            namespaces=gate_element.nsmap
        )

        if len(coord_els) > 2:
            raise NotImplementedError(
                'Ellipsoids over 2 dimensions are not yet supported (line %d)' % gate_element.sourceline
            )
        elif len(coord_els) == 1:
            raise ValueError(
                'Ellipsoids must have at least 2 dimensions (line %d)' % gate_element.sourceline
            )

        for coord_el in coord_els:
            value_attribs = coord_el.xpath(
                '@%s:value' % data_type_namespace,
                namespaces=gate_element.nsmap
            )
            if len(value_attribs) != 1:
                raise ValueError(
                    'A coordinate must have only 1 value (line %d)' % coord_el.sourceline
                )

            self.coordinates.append(float(value_attribs[0]))

        # Next, we'll parse the covariance matrix, containing 2 'row'
        # elements, each containing 2 'entry' elements w/ value attributes
        covariance_el = gate_element.find(
            '%s:covarianceMatrix' % gating_namespace,
            namespaces=gate_element.nsmap
        )

        self.covariance_matrix = []

        covar_row_els = covariance_el.findall(
            '%s:row' % gating_namespace,
            namespaces=gate_element.nsmap
        )

        for row_el in covar_row_els:
            row_entry_els = row_el.findall(
                '%s:entry' % gating_namespace,
                namespaces=gate_element.nsmap
            )

            entry_vals = []
            for entry_el in row_entry_els:
                value_attribs = entry_el.xpath(
                    '@%s:value' % data_type_namespace,
                    namespaces=gate_element.nsmap
                )

                entry_vals.append(float(value_attribs[0]))

            if len(entry_vals) != 2:
                raise ValueError(
                    'A covariance row entry must have 2 values (line %d)' % row_el.sourceline
                )

            self.covariance_matrix.append(entry_vals)

        # Finally, get the distance square, which is a simple element w/
        # a single value attribute
        distance_square_el = gate_element.find(
            '%s:distanceSquare' % gating_namespace,
            namespaces=gate_element.nsmap
        )

        dist_square_value_attribs = distance_square_el.xpath(
            '@%s:value' % data_type_namespace,
            namespaces=gate_element.nsmap
        )

        self.distance_square = float(dist_square_value_attribs[0])

    def apply(self, sample):
        events = sample.get_raw_events()
        pnn_labels = sample.pnn_labels

        if events.shape[1] != len(pnn_labels):
            raise ValueError(
                "Number of FCS dimensions (%d) does not match label count (%d)"
                % (events.shape[1], len(pnn_labels))
            )

        dim_idx = []
        dim_comp_refs = set()

        for dim in self.dimensions:
            if dim.compensation_ref not in [None, 'uncompensated']:
                dim_comp_refs.add(dim.compensation_ref)

            dim_idx.append(pnn_labels.index(dim.label))

        events = self.compensate_sample(dim_comp_refs, sample)

        ellipse = utils.calculate_ellipse(
            self.coordinates[0],
            self.coordinates[1],
            self.covariance_matrix,
            n_std_dev=self.distance_square
        )

        results = utils.points_in_ellipse(ellipse, events[:, dim_idx])

        return results


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

        if not val:
            raise ValueError("Document is not valid GatingML")

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

    def gate_sample(self, sample, gate_id):
        """
        Apply a gate to a sample, returning a dictionary where gate ID is the
        key, and the value contains the event indices for events in the Sample
        which are contained by the gate. If the gate has a parent gate, all
        gates in the hierarchy above will be included in the results. If 'gate'
        is None, then all gates will be evaluated.

        :param sample: an FCS Sample instance
        :param gate_id: A gate ID to evaluate on given Sample. If None, all gates
            will be evaluated
        :return: Dictionary where keys are gate IDs, values are event indices
            in the given Sample which are contained by the gate
        """
        if gate_id is None:
            gates = self.gates
        else:
            gates = [self.gates[gate_id]]

        results = {}

        for gate in gates:
            results[gate_id] = gate.apply(sample)

        return results
