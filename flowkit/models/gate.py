from abc import ABC, abstractmethod
# noinspection PyUnresolvedReferences
from lxml import etree, objectify
import numpy as np
import flowutils
from flowkit import utils


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

    def apply_parent_gate(self, sample, results, parent_results):
        if self.parent is not None:
            parent_gate = self.__parent__.get_gate_by_reference(self.parent)
            parent_id = parent_gate.id

            if parent_results is not None:
                results_and_parent = np.logical_and(parent_results['events'], results)
                parent_count = parent_results['count']
            else:
                parent_result = parent_gate.apply(sample, parent_results)

                if isinstance(parent_gate, QuadrantGate):
                    parent_result = parent_result[self.parent]

                parent_count = parent_result['count']
                results_and_parent = np.logical_and(parent_result['events'], results)
        else:
            # no parent, so results are unchanged & parent count is total count
            parent_id = None
            parent_count = sample.event_count
            results_and_parent = results

        event_count = results_and_parent.sum()

        # TODO: check parent_count for div by zero
        final_results = {
            'sample': sample.original_filename,
            'events': results_and_parent,
            'count': event_count,
            'absolute_percent': (event_count / float(sample.event_count)) * 100.0,
            'relative_percent': (event_count / float(parent_count)) * 100.0,
            'parent': parent_id
        }

        return final_results

    @abstractmethod
    def apply(self, sample, parent_results):
        pass

    def compensate_sample(self, dim_comp_refs, sample):
        comp_matrix = None
        indices = None
        dim_comp_ref_count = len(dim_comp_refs)

        if dim_comp_ref_count == 0:
            events = sample.get_raw_events()
            return events.copy()
        elif dim_comp_ref_count > 1:
            raise NotImplementedError(
                "Mixed compensation between individual channels is not "
                "implemented. Never seen it, but if you are reading this "
                "message, submit an issue to have it implemented."
            )
        else:
            comp_ref = list(dim_comp_refs)[0]

        if comp_ref == 'FCS':
            meta = sample.get_metadata()
            spill = None
            if 'spill' not in meta or 'spillover' not in meta:
                pass
            elif 'spillover' in meta:  # preferred, per FCS standard
                spill = meta['spillover']
            elif 'spill' in meta:
                spill = meta['spill']

            if spill is not None:
                spill = utils.parse_compensation_matrix(
                    spill,
                    sample.pnn_labels,
                    null_channels=sample.null_channels
                )
                indices = spill[0, :]  # headers are channel #'s
                indices = [int(i - 1) for i in indices]
                comp_matrix = spill[1:, :]
        else:
            # lookup specified comp-ref in parent strategy
            matrix = self.__parent__.comp_matrices[comp_ref]
            indices = [
                sample.get_channel_index(d) for d in matrix.detectors
            ]
            comp_matrix = matrix.matrix

        events = self.__parent__.get_cached_compensation(
            sample,
            comp_ref
        )

        if events is None:
            events = sample.get_raw_events()
            events = events.copy()
        else:
            return events

        if comp_matrix is not None:
            events = flowutils.compensate.compensate(
                events,
                comp_matrix,
                indices
            )
            # cache the comp events
            self.__parent__.cache_compensated_events(
                sample,
                comp_ref,
                events
            )

        return events

    def preprocess_sample_events(self, sample):
        pnn_labels = sample.pnn_labels
        pns_labels = sample.pns_labels

        dim_idx = []
        dim_min = []
        dim_max = []
        dim_comp_refs = set()
        new_dims = []
        dim_xform = []

        for dim in self.dimensions:
            dim_comp = False
            if dim.compensation_ref not in [None, 'uncompensated']:
                dim_comp_refs.add(dim.compensation_ref)
                dim_comp = True

            if dim.label is None:
                # dimension is a transform of other dimensions
                new_dims.append(dim)
                continue

            if dim.label in pnn_labels:
                dim_idx.append(pnn_labels.index(dim.label))
            elif dim.label in pns_labels:
                dim_idx.append(pns_labels.index(dim.label))
            else:
                # for a referenced comp, the label may have been the
                # fluorochrome instead of the channel's PnN label. If so,
                # the referenced matrix object will also have the detector
                # names that will match
                if not dim_comp:
                    raise LookupError(
                        "%s is not found as a channel label or channel reference in %s" % (dim.label, sample)
                    )
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
                events[:, [dim]] = xform.apply(events[:, [dim]])

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

    def apply(self, sample, parent_results):
        events, dim_idx, dim_min, dim_max, new_dims = super().preprocess_sample_events(sample)

        results = np.ones(events.shape[0], dtype=np.bool)

        for i, d_idx in enumerate(dim_idx):
            if dim_min[i] is not None:
                results = np.bitwise_and(results, events[:, d_idx] >= dim_min[i])
            if dim_max[i] is not None:
                results = np.bitwise_and(results, events[:, d_idx] < dim_max[i])

        for new_dim in new_dims:
            # TODO: RatioTransforms aren't limited to rect gates, refactor to
            # allow other gate classes to handle new dimensions created from
            # ratio transforms. Also, the ratio transform's apply method is
            # different from other xforms in that it takes a sample argument
            # and not an events arguments

            # new dimensions are defined by transformations of other dims
            try:
                new_dim_xform = self.__parent__.transformations[new_dim.new_dim_transformation_ref]
            except KeyError:
                raise KeyError("New dimensions must provide a transformation")

            xform_events = new_dim_xform.apply(sample)

            if new_dim.transformation_ref is not None:
                xform = self.__parent__.transformations[new_dim.transformation_ref]
                xform_events = xform.apply(xform_events)

            if new_dim.min is not None:
                results = np.bitwise_and(results, xform_events >= new_dim.min)
            if new_dim.max is not None:
                results = np.bitwise_and(results, xform_events < new_dim.max)

        results = self.apply_parent_gate(sample, results, parent_results)

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

    def apply(self, sample, parent_results):
        events, dim_idx, dim_min, dim_max, new_dims = super().preprocess_sample_events(sample)
        path_verts = []

        for vert in self.vertices:
            path_verts.append(vert.coordinates)

        results = utils.points_in_polygon(np.array(path_verts, dtype='double'), events[:, dim_idx])

        results = self.apply_parent_gate(sample, results, parent_results)

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

    def apply(self, sample, parent_results):
        events, dim_idx, dim_min, dim_max, new_dims = super().preprocess_sample_events(sample)

        results = utils.points_in_ellipsoid(
            self.covariance_matrix,
            self.coordinates,
            self.distance_square,
            events[:, dim_idx]
        )

        results = self.apply_parent_gate(sample, results, parent_results)

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

    def apply_parent_gate(self, sample, results, parent_results):
        if self.parent is not None and parent_results is not None:
            parent_gate = self.__parent__.get_gate_by_reference(self.parent)
            parent_id = self.parent
            parent_events = parent_gate.apply(sample)

            if isinstance(parent_gate, QuadrantGate):
                parent_events = parent_events[self.parent]
                parent_count = parent_events.sum()
            else:
                parent_count = parent_events['events'].sum()

            results_and_parent = {}
            for q_id, q_result in results.items():
                results_and_parent[q_id] = np.logical_and(
                    parent_events,
                    q_result
                )
        else:
            if parent_results is not None:
                results_and_parent = np.logical_and(
                    parent_results,
                    results
                )
            else:
                results_and_parent = results
            parent_count = sample.event_count
            parent_id = None

        final_results = {}

        for q_id, q_result in results_and_parent.items():
            q_event_count = q_result.sum()

            final_results[q_id] = {
                'sample': sample.original_filename,
                'events': q_result,
                'count': q_event_count,
                'absolute_percent': (q_event_count / float(sample.event_count)) * 100.0,
                'relative_percent': (q_event_count / float(parent_count)) * 100.0,
                'parent': parent_id
            }

        return final_results

    def apply(self, sample, parent_results):
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

        results = self.apply_parent_gate(sample, results, parent_results)

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

    def apply(self, sample, parent_results):
        all_gate_results = []

        for gate_ref_dict in self.gate_refs:
            gate = self.__parent__.get_gate_by_reference(gate_ref_dict['ref'])
            gate_ref_results = gate.apply(sample, parent_results)

            if isinstance(gate, QuadrantGate):
                gate_ref_events = gate_ref_results[gate_ref_dict['ref']]['events']
            else:
                gate_ref_events = gate_ref_results['events']

            if gate_ref_dict['complement']:
                gate_ref_events = ~gate_ref_events

            all_gate_results.append(gate_ref_events)

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

        results = self.apply_parent_gate(sample, results, parent_results)

        return results
