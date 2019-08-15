import numpy as np
from .base_gate import Gate
from flowkit import _utils


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
            gate_id,
            parent_id,
            dimensions,
            gating_strategy,
    ):
        super().__init__(
            gate_id,
            parent_id,
            dimensions,
            gating_strategy
        )
        self.gate_type = "RectangleGate"

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
            # different from other transforms in that it takes a sample argument
            # and not an events arguments

            # new dimensions are defined by transformations of other dims
            try:
                new_dim_xform = self.__parent__.transformations[new_dim.ratio_ref]
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
            gate_id,
            parent_id,
            dimensions,
            vertices,
            gating_strategy,
    ):
        super().__init__(
            gate_id,
            parent_id,
            dimensions,
            gating_strategy
        )
        self.vertices = vertices
        self.gate_type = "PolygonGate"

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.id}, parent: {self.parent}, vertices: {len(self.vertices)})'
        )

    def apply(self, sample, parent_results):
        events, dim_idx, dim_min, dim_max, new_dims = super().preprocess_sample_events(sample)
        path_vertices = []

        for vert in self.vertices:
            path_vertices.append(vert.coordinates)

        results = _utils.points_in_polygon(np.array(path_vertices, dtype='double'), events[:, dim_idx])

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
            gate_id,
            parent_id,
            dimensions,
            coordinates,
            covariance_matrix,
            distance_square,
            gating_strategy
    ):
        super().__init__(
            gate_id,
            parent_id,
            dimensions,
            gating_strategy
        )
        self.gate_type = "EllipsoidGate"
        self.coordinates = coordinates
        self.covariance_matrix = covariance_matrix
        self.distance_square = distance_square

        if len(coordinates) == 1:
            raise ValueError(
                'Ellipsoids must have at least 2 dimensions'
            )

        if len(covariance_matrix) != len(self.coordinates):
            raise ValueError(
                'Covariance row entry value count must match # of dimensions'
            )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.id}, parent: {self.parent}, coords: {self.coordinates})'
        )

    def apply(self, sample, parent_results):
        events, dim_idx, dim_min, dim_max, new_dims = super().preprocess_sample_events(sample)

        results = _utils.points_in_ellipsoid(
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
            gate_id,
            parent_id,
            dimensions,
            quadrants,
            gating_strategy,
    ):
        super().__init__(
            gate_id,
            parent_id,
            dimensions,
            gating_strategy
        )
        self.gate_type = "QuadrantGate"

        # First, check dimension count
        if len(self.dimensions) < 1:
            raise ValueError('Quadrant gates must have at least 1 divider')

        # Parse quadrants
        for q_id, dividers in quadrants.items():
            # quadrants is a dictionary where keys are quad IDs, value is a list
            # of dicts, each containing keys:
            #  - divider
            #  - dimension: dimension label
            #  - location
            #  - min
            #  - max
            for divider in dividers:
                dim_label = None

                for dim in self.dimensions:
                    if dim.id != divider['divider']:
                        continue
                    else:
                        dim_label = dim.dimension_ref

                if dim_label is None:
                    raise ValueError(
                        'Quadrant must define a divider reference'
                    )

        self.quadrants = quadrants

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
                'parent': parent_id,
                'gate_type': self.gate_type
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
            gate_id,
            parent_id,
            dimensions,
            bool_type,
            gate_refs,
            gating_strategy
    ):
        super().__init__(
            gate_id,
            parent_id,
            dimensions,
            gating_strategy
        )
        self.gate_type = "BooleanGate"

        if bool_type not in ['and', 'or', 'not']:
            raise ValueError(
                "Boolean gate must specify one of 'and', 'or', or 'not'"
            )
        self.type = bool_type
        self.gate_refs = gate_refs

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
