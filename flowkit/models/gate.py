from abc import ABC, abstractmethod
# noinspection PyUnresolvedReferences
from lxml import etree, objectify
import anytree
from anytree.exporter import DotExporter
import numpy as np
import flowutils
from flowkit import utils
from flowkit.resources import gml_schema
from .transforms import gml_transforms

GATE_TYPES = [
    'RectangleGate',
    'PolygonGate',
    'EllipsoidGate',
    'QuadrantGate',
    'BooleanGate'
]


class Dimension(object):
    def __init__(self, dim_element, gating_namespace, data_type_namespace):
        # check for presence of optional 'id' (present in quad gate dividers)
        self.id = utils.find_attribute_value(dim_element, gating_namespace, 'id')

        self.label = None
        self.compensation_ref = utils.find_attribute_value(dim_element, gating_namespace, 'compensation-ref')
        self.transformation_ref = utils.find_attribute_value(dim_element, gating_namespace, 'transformation-ref')
        self.new_dim_transformation_ref = None

        self.min = None
        self.max = None
        self.values = []  # quad gate dims can have multiple values

        # should be 0 or only 1 'min' attribute
        _min = utils.find_attribute_value(dim_element, gating_namespace, 'min')

        if _min is not None:
            self.min = float(_min)

        # ditto for 'max' attribute, 0 or 1 value
        _max = utils.find_attribute_value(dim_element, gating_namespace, 'max')

        if _max is not None:
            self.max = float(_max)

        # values in gating namespace, ok if not present
        value_els = dim_element.findall(
            '%s:value' % gating_namespace,
            namespaces=dim_element.nsmap
        )

        for value in value_els:
            self.values.append(float(value.text))

        # label be here
        fcs_dim_els = dim_element.find(
            '%s:fcs-dimension' % data_type_namespace,
            namespaces=dim_element.nsmap
        )

        # if no 'fcs-dimension' element is present, this might be a
        # 'new-dimension'  made from a transformation on other dims
        if fcs_dim_els is None:
            new_dim_el = dim_element.find(
                '%s:new-dimension' % data_type_namespace,
                namespaces=dim_element.nsmap
            )
            if new_dim_el is None:
                raise ValueError(
                    "Dimension invalid: neither fcs-dimension or new-dimension "
                    "tags found (line %d)" % dim_element.sourceline
                )

            # if we get here, there should be a 'transformation-ref' attribute
            xform_ref = utils.find_attribute_value(new_dim_el, data_type_namespace, 'transformation-ref')

            if xform_ref is not None:
                self.new_dim_transformation_ref = xform_ref
        else:
            self.label = utils.find_attribute_value(fcs_dim_els, data_type_namespace, 'name')
            if self.label is None:
                raise ValueError(
                    'Dimension name not found (line %d)' % fcs_dim_els.sourceline
                )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.id}, label: {self.label})'
        )


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

        # should be 0 or only 1 'min' attribute,
        for coord_el in coord_els:
            value = utils.find_attribute_value(coord_el, data_type_namespace, 'value')
            if value is None:
                raise ValueError(
                    'Vertex coordinate must have only 1 value (line %d)' % coord_el.sourceline
                )

            self.coordinates.append(float(value))

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.coordinates})'
        )


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
        self.id = utils.find_attribute_value(gate_element, gating_namespace, 'id')
        self.parent = utils.find_attribute_value(gate_element, gating_namespace, 'parent_id')

        # most gates specify dimensions in the 'dimension' tag,
        # but quad gates specify dimensions in the 'divider' tag
        dim_els = gate_element.findall(
            '%s:divider' % gating_namespace,
            namespaces=gate_element.nsmap
        )
        if len(dim_els) == 0:
            dim_els = gate_element.findall(
                '%s:dimension' % gating_namespace,
                namespaces=gate_element.nsmap
            )

        self.dimensions = []

        for dim_el in dim_els:
            dim = Dimension(dim_el, gating_namespace, data_type_namespace)
            self.dimensions.append(dim)

    def apply_parent_gate(self, sample, results):
        if self.parent is not None:
            parent_gate = self.__parent__.get_gate_by_reference(self.parent)

            parent_events = parent_gate.apply(sample)

            if isinstance(parent_gate, QuadrantGate):
                parent_events = parent_events[self.parent]

            results_and_parent = np.logical_and(parent_events, results)
        else:
            results_and_parent = results

        return results_and_parent

    @abstractmethod
    def apply(self, sample):
        pass

    def compensate_sample(self, dim_comp_refs, sample):
        events = sample.get_raw_events()
        events = events.copy()

        spill = None
        matrix = None

        if len(dim_comp_refs) == 0:
            pass
        elif len(dim_comp_refs) > 1:
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
            # lookup specified comp-ref in parent strategy
            matrix = self.__parent__.comp_matrices[list(dim_comp_refs)[0]]

        if spill is not None:
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
        elif matrix is not None:
            indices = [
                sample.get_channel_index(d) for d in matrix.detectors
            ]
            events = flowutils.compensate.compensate(
                events,
                matrix.matrix,
                indices
            )

        return events

    def preprocess_sample_events(self, sample):
        events = sample.get_raw_events()
        events = events.copy()
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
        new_dims = []
        dim_xform = []

        for dim in self.dimensions:
            if dim.compensation_ref not in [None, 'uncompensated']:
                dim_comp_refs.add(dim.compensation_ref)

            if dim.label is None:
                # dimension is a transform of other dimensions
                new_dims.append(dim)
                continue

            try:
                dim_idx.append(pnn_labels.index(dim.label))
                dim_min.append(dim.min)
                dim_max.append(dim.max)
                dim_xform.append(dim.transformation_ref)
            except ValueError:
                # for a referenced comp, the label may have been the
                # fluorochrome instead of the channel's PnN label. If so,
                # the referenced matrix object will also have the detector
                # names that will match
                matrix = self.__parent__.comp_matrices[dim.compensation_ref]
                matrix_dim_idx = matrix.fluorochomes.index(dim.label)
                detector = matrix.detectors[matrix_dim_idx]
                dim_idx.append(pnn_labels.index(detector))
                dim_min.append(dim.min)
                dim_max.append(dim.max)
                dim_xform.append(dim.transformation_ref)

        events = self.compensate_sample(dim_comp_refs, sample)

        for i, dim in enumerate(dim_idx):
            if dim_xform[i] is not None:
                xform = self.__parent__.transformations[dim_xform[i]]
                events[:, dim] = xform.apply(events[:, dim])

        return events, dim_idx, dim_min, dim_max, new_dims


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

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.id}, parent: {self.parent}, dims: {len(self.dimensions)})'
        )

    def apply(self, sample):
        events, dim_idx, dim_min, dim_max, new_dims = super().preprocess_sample_events(sample)

        results = np.ones(events.shape[0], dtype=np.bool)

        for i, d_idx in enumerate(dim_idx):
            if dim_min[i] is not None:
                results = np.bitwise_and(results, events[:, d_idx] >= dim_min[i])
            if dim_max[i] is not None:
                results = np.bitwise_and(results, events[:, d_idx] < dim_max[i])

        for new_dim in new_dims:
            # new dimensions are defined by transformations of other dims
            new_dim_xform = self.__parent__.transformations[new_dim.new_dim_transformation_ref]
            xform_events = new_dim_xform.apply(sample)

            if new_dim.transformation_ref is not None:
                xform = self.__parent__.transformations[new_dim.transformation_ref]
                xform_events = xform.apply(xform_events)

            if new_dim.min is not None:
                results = np.bitwise_and(results, xform_events >= new_dim.min)
            if new_dim.max is not None:
                results = np.bitwise_and(results, xform_events < new_dim.max)

        results = self.apply_parent_gate(sample, results)

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

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.id}, parent: {self.parent}, vertices: {len(self.vertices)})'
        )

    def apply(self, sample):
        events, dim_idx, dim_min, dim_max, new_dims = super().preprocess_sample_events(sample)
        path_verts = []

        for vert in self.vertices:
            path_verts.append(vert.coordinates)

        results = utils.points_in_polygon(path_verts, events[:, dim_idx])

        results = self.apply_parent_gate(sample, results)

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

        if len(coord_els) == 1:
            raise ValueError(
                'Ellipsoids must have at least 2 dimensions (line %d)' % gate_element.sourceline
            )

        for coord_el in coord_els:
            value = utils.find_attribute_value(coord_el, data_type_namespace, 'value')
            if value is None:
                raise ValueError(
                    'A coordinate must have only 1 value (line %d)' % coord_el.sourceline
                )

            self.coordinates.append(float(value))

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
                value = utils.find_attribute_value(entry_el, data_type_namespace, 'value')
                entry_vals.append(float(value))

            if len(entry_vals) != len(self.coordinates):
                raise ValueError(
                    'Covariance row entry value count must match # of dimensions (line %d)' % row_el.sourceline
                )

            self.covariance_matrix.append(entry_vals)

        # Finally, get the distance square, which is a simple element w/
        # a single value attribute
        distance_square_el = gate_element.find(
            '%s:distanceSquare' % gating_namespace,
            namespaces=gate_element.nsmap
        )

        dist_square_value = utils.find_attribute_value(distance_square_el, data_type_namespace, 'value')
        self.distance_square = float(dist_square_value)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.id}, parent: {self.parent}, coords: {self.coordinates})'
        )

    def apply(self, sample):
        events, dim_idx, dim_min, dim_max, new_dims = super().preprocess_sample_events(sample)

        results = utils.points_in_ellipsoid(
            self.covariance_matrix,
            self.coordinates,
            self.distance_square,
            events[:, dim_idx]
        )

        results = self.apply_parent_gate(sample, results)

        return results


class QuadrantGate(Gate):
    """
    Represents a GatingML Quadrant Gate

    A QuadrantGate must have at least 1 divider, and must specify the labels
    of the resulting quadrants the dividers produce. Quadrant gates are
    different from other gate types in that they are actually a collection of
    gates (quadrants), though even the term quadrant is misleading as they can
    divide a plane into more than 4 sections.

    Note: Only specific quadrants may be referenced as parent gates or as a
    component of a Boolean gate. If a QuadrantGate has a parent, then the
    parent gate is applicable to all quadrants in the QuadrantGate.
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

        # First, we'll check dimension count
        if len(self.dimensions) < 1:
            raise ValueError(
                'Quadrant gates must have at least 1 divider (line %d)' % gate_element.sourceline
            )

        # Next, we'll parse the Quadrant elements, each containing an
        # id attribute, and 1 or more 'position' elements. Each position
        # element has a 'divider-ref' and 'location' attribute.
        quadrant_els = gate_element.findall(
            '%s:Quadrant' % gating_namespace,
            namespaces=gate_element.nsmap
        )

        self.quadrants = {}

        for quadrant_el in quadrant_els:
            quad_id = utils.find_attribute_value(quadrant_el, gating_namespace, 'id')
            self.quadrants[quad_id] = []

            position_els = quadrant_el.findall(
                '%s:position' % gating_namespace,
                namespaces=gate_element.nsmap
            )

            for pos_el in position_els:
                divider_ref = utils.find_attribute_value(pos_el, gating_namespace, 'divider_ref')
                location = utils.find_attribute_value(pos_el, gating_namespace, 'location')

                divider = divider_ref
                location = float(location)
                q_min = None
                q_max = None
                dim_label = None

                for dim in self.dimensions:
                    if dim.id != divider:
                        continue
                    else:
                        dim_label = dim.label

                    for v in sorted(dim.values):
                        if v > location:
                            q_max = v

                            # once we have a max value, no need to
                            break
                        elif v <= location:
                            q_min = v

                if dim_label is None:
                    raise ValueError(
                        'Quadrant must define a divider reference (line %d)' % pos_el.sourceline
                    )

                self.quadrants[quad_id].append(
                    {
                        'divider': divider,
                        'dimension': dim_label,
                        'location': location,
                        'min': q_min,
                        'max': q_max
                    }
                )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.id}, parent: {self.parent}, quadrants: {len(self.quadrants)})'
        )

    def apply_parent_gate(self, sample, results):
        if self.parent is not None:
            parent_gate = self.__parent__.get_gate_by_reference(self.parent)

            parent_events = parent_gate.apply(sample)

            if isinstance(parent_gate, QuadrantGate):
                parent_events = parent_events[self.parent]

            results_and_parent = {}
            for q_id, q_result in results.items():
                results_and_parent[q_id] = np.logical_and(
                    parent_events,
                    q_result
                )
        else:
            results_and_parent = results

        return results_and_parent

    def apply(self, sample):
        events, dim_idx, dim_min, dim_max, new_dims = super().preprocess_sample_events(sample)

        results = {}

        for q_id, quadrant in self.quadrants.items():
            q_results = np.ones(events.shape[0], dtype=np.bool)

            # quadrant is a list of dicts containing quadrant bounds and
            # the referenced dimension
            for bound in quadrant:
                dim_idx = sample.pnn_labels.index(bound['dimension'])

                if bound['min'] is not None:
                    q_results = np.bitwise_and(
                        q_results,
                        events[:, dim_idx] >= bound['min']
                    )
                if bound['max'] is not None:
                    q_results = np.bitwise_and(
                        q_results,
                        events[:, dim_idx] < bound['max']
                    )

                results[q_id] = q_results

        results = self.apply_parent_gate(sample, results)

        return results


class BooleanGate(Gate):
    """
    Represents a GatingML Boolean Gate

    A BooleanGate performs the boolean operations AND, OR, or NOT on one or
    more other gates. Note, the boolean operation XOR is not supported in the
    GatingML specification but can be implemented using a combination of the
    supported operations.
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
        # boolean gates do not mix multiple operations, so there should be only
        # one of the following: 'and', 'or', or 'not'
        and_els = gate_element.findall(
            '%s:and' % gating_namespace,
            namespaces=gate_element.nsmap
        )
        or_els = gate_element.findall(
            '%s:or' % gating_namespace,
            namespaces=gate_element.nsmap
        )
        not_els = gate_element.findall(
            '%s:not' % gating_namespace,
            namespaces=gate_element.nsmap
        )

        if len(and_els) > 0:
            self.type = 'and'
            bool_op_el = and_els[0]
        elif len(or_els) > 0:
            self.type = 'or'
            bool_op_el = or_els[0]
        elif len(not_els) > 0:
            self.type = 'not'
            bool_op_el = not_els[0]
        else:
            raise ValueError(
                "Boolean gate must specify one of 'and', 'or', or 'not' (line %d)" % gate_element.sourceline
            )

        gate_ref_els = bool_op_el.findall(
            '%s:gateReference' % gating_namespace,
            namespaces=gate_element.nsmap
        )

        self.gate_refs = []

        for gate_ref_el in gate_ref_els:
            gate_ref = utils.find_attribute_value(gate_ref_el, gating_namespace, 'ref')
            if gate_ref is None:
                raise ValueError(
                    "Boolean gate reference must specify a 'ref' attribute (line %d)" % gate_ref_el.sourceline
                )

            use_complement = utils.find_attribute_value(gate_ref_el, gating_namespace, 'use-as-complement')
            if use_complement is not None:
                use_complement = use_complement == 'true'
            else:
                use_complement = False

            self.gate_refs.append(
                {
                    'ref': gate_ref,
                    'complement': use_complement
                }
            )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.id}, parent: {self.parent}, type: {self.type})'
        )

    def apply(self, sample):
        all_gate_results = []

        for gate_ref_dict in self.gate_refs:
            gate = self.__parent__.get_gate_by_reference(gate_ref_dict['ref'])
            gate_ref_results = gate.apply(sample)

            if isinstance(gate, QuadrantGate):
                gate_ref_results = gate_ref_results[gate_ref_dict['ref']]

            if gate_ref_dict['complement']:
                gate_ref_results = ~gate_ref_results

            all_gate_results.append(gate_ref_results)

        if self.type == 'and':
            results = np.logical_and.reduce(all_gate_results)
        elif self.type == 'or':
            results = np.logical_or.reduce(all_gate_results)
        elif self.type == 'not':
            # gml spec states only 1 reference is allowed for 'not' gate
            results = np.logical_not(all_gate_results[0])
        else:
            raise ValueError(
                "Boolean gate must specify one of 'and', 'or', or 'not'"
            )

        results = self.apply_parent_gate(sample, results)

        return results


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
