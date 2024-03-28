"""
gates Module
"""
import numpy as np
from ._base_gate import Gate
from ..dimension import RatioDimension
from flowutils import gating


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
            gate_name,
            dimensions,
            use_complement=False
    ):
        """
        Create a RectangleGate instance.

        :param gate_name: text string for the name of the gate
        :param dimensions: list of Dimension instances used to define the gate boundaries
        :param use_complement: whether to use events inside or outside gate area, default is False (inside gate area)
        """
        super().__init__(
            gate_name,
            dimensions
        )
        self.gate_type = "RectangleGate"
        self.use_complement = use_complement

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.gate_name}, dims: {len(self.dimensions)})'
        )

    def apply(self, df_events):
        """
        Apply gate to events in given pandas DataFrame. The given DataFrame must have columns matching the
        Dimension IDs for the Dimension instances defined for the gate.

        :param df_events: pandas DataFrame with column labels matching Dimension IDs
        :return: NumPy array of boolean values for each event.  (True is inside gate)
        """
        results = np.ones(df_events.shape[0], dtype=bool)

        for i, dim in enumerate(self.dimensions):
            if isinstance(dim, RatioDimension):
                dim_id = dim.ratio_ref
            else:
                dim_id = dim.id

            if dim.min is not None:
                results = np.bitwise_and(results, df_events[dim_id].values >= dim.min)
            if dim.max is not None:
                results = np.bitwise_and(results, df_events[dim_id].values < dim.max)

        if self.use_complement:
            results = np.logical_not(results)

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
            gate_name,
            dimensions,
            vertices,
            use_complement=False
    ):
        """
        Create a PolygonGate instance.

        :param gate_name: text string for the name of the gate
        :param dimensions: list of Dimension instances
        :param vertices: list of 2-D coordinates used to define gate boundary
        :param use_complement: whether to use events inside or outside gate area, default is False (inside gate area)
        """
        super().__init__(
            gate_name,
            dimensions
        )
        self.vertices = vertices
        self.gate_type = "PolygonGate"
        self.use_complement = use_complement

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.gate_name}, vertices: {len(self.vertices)})'
        )

    def apply(self, df_events):
        """
        Apply gate to events in given pandas DataFrame. The given DataFrame must have columns matching the
        Dimension IDs for the Dimension instances defined for the gate.

        :param df_events: pandas DataFrame with column labels matching Dimension IDs
        :return: NumPy array of boolean values for each event  (True is inside gate)
        """
        dim_ids_ordered = []
        for i, dim in enumerate(self.dimensions):
            if isinstance(dim, RatioDimension):
                dim_ids_ordered.append(dim.ratio_ref)
            else:
                dim_ids_ordered.append(dim.id)

        results = gating.points_in_polygon(
            np.array(self.vertices, dtype=np.float64),
            df_events[dim_ids_ordered].values  # send a NumPy array and not a DataFrame
        )

        if self.use_complement:
            results = np.logical_not(results)

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
            gate_name,
            dimensions,
            coordinates,
            covariance_matrix,
            distance_square
    ):
        """
        Create an EllipsoidGate instance.

        :param gate_name: text string for the name of the gate
        :param dimensions: list of Dimension instances
        :param coordinates: center point of the ellipsoid for n-dimensions
        :param covariance_matrix: Covariance matrix for the ellipsoid shape (NxN array)
        :param distance_square: square of the Mahalanobis distance, controlling
            the size of the ellipsoid. The distance square parameter is conceptually
            similar to the number of standard deviations representing the boundary
            for an n-dimensional distribution of points.
        """
        super().__init__(
            gate_name,
            dimensions
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
            f'{self.gate_name}, coords: {self.coordinates})'
        )

    def apply(self, df_events):
        """
        Apply gate to events in given pandas DataFrame. The given DataFrame must have columns matching the
        Dimension IDs for the Dimension instances defined for the gate.

        :param df_events: pandas DataFrame with column labels matching Dimension IDs
        :return: NumPy array of boolean values for each event (True is inside gate)
        """
        dim_ids_ordered = []
        for i, dim in enumerate(self.dimensions):
            if isinstance(dim, RatioDimension):
                dim_ids_ordered.append(dim.ratio_ref)
            else:
                dim_ids_ordered.append(dim.id)

        results = gating.points_in_ellipsoid(
            self.covariance_matrix,
            self.coordinates,
            self.distance_square,
            df_events[dim_ids_ordered].values  # send a NumPy array and not a DataFrame
        )

        return results


class Quadrant(object):
    """
    Represents a single quadrant of a QuadrantGate.

    A quadrant is a rectangular section where the boundaries are specified by QuadrantDivider
    references.
    """
    def __init__(self, quadrant_id, divider_refs, divider_ranges):
        """
        Create a Quadrant instance.

        :param quadrant_id: text string to identify the quadrant
        :param divider_refs: list of text strings referencing QuadrantDivider instances
        :param divider_ranges: list of min/max pairs corresponding to boundaries of the
            given QuadrantDivider references.
        """
        self.id = quadrant_id

        div_count = len(divider_refs)

        if div_count != len(divider_ranges):
            raise ValueError("A min/max range must be specified for each divider reference")

        self.divider_refs = divider_refs
        self._divider_ranges = {}

        for i, div_range in enumerate(divider_ranges):
            if len(div_range) != 2:
                raise ValueError("Each divider range must have both a min & max value")

            self._divider_ranges[self.divider_refs[i]] = div_range

        if self._divider_ranges is None or len(self._divider_ranges) != div_count:
            raise ValueError("Failed to parse divider ranges")

    def get_divider_range(self, div_ref):
        """
        Returns the divider range values for the given QuadrantDivider ID reference.

        :param div_ref: QuadrantDivider ID string
        :return: min/max range values for the requested divider
        """
        return self._divider_ranges[div_ref]

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.id}, dividers: {len(self.divider_refs)})'
        )


class QuadrantGate(Gate):
    """
    Represents a GatingML Quadrant Gate.

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
            gate_name,
            dividers,
            quadrants
    ):
        """
        Create a QuadrantGate instance.

        :param gate_name: text string for the name of the gate
        :param dividers: a list of QuadrantDivider instances
        :param quadrants: a list of Quadrant instances
        """
        super().__init__(
            gate_name,
            dividers
        )
        self.gate_type = "QuadrantGate"

        # First, check dimension count
        if len(self.dimensions) < 1:
            raise ValueError('Quadrant gates must have at least 1 divider')

        # Parse quadrants
        for quadrant in quadrants:
            for divider_ref in quadrant.divider_refs:
                dim_id = None

                # self.dimensions in a QuadrantGate are dividers
                # make sure all divider IDs are referenced in the list of quadrants
                # and verify there is a dimension ID (for each quad)
                for dim in self.dimensions:
                    if dim.id != divider_ref:
                        continue
                    else:
                        dim_id = dim.dimension_ref

                if dim_id is None:
                    raise ValueError(
                        'Quadrant must define a divider reference'
                    )

        self.quadrants = {q.id: q for q in quadrants}

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.gate_name}, quadrants: {len(self.quadrants)})'
        )

    def apply(self, df_events):
        """
        Apply gate to events in given pandas DataFrame. The given DataFrame must have columns matching the
        QuadrantDivider dimension_ref for the QuadrantDivider instances defined for the gate.

        :param df_events: pandas DataFrame with column labels matching QuadrantDivider dimension_ref values.
        :return: A dictionary where each key is a Quadrant ID and the value is a NumPy array of boolean values
            for each event (True is inside gate).
        """
        results = {}

        for q_id, quadrant in self.quadrants.items():
            q_results = np.ones(df_events.shape[0], dtype=bool)

            dim_lut = {dim.id: dim.dimension_ref for dim in self.dimensions}

            # quadrant is a list of dicts containing quadrant bounds and
            # the referenced dimension
            for div_ref in quadrant.divider_refs:
                dim_ref = dim_lut[div_ref]
                div_ranges = quadrant.get_divider_range(div_ref)

                if div_ranges[0] is not None:
                    q_results = np.bitwise_and(
                        q_results,
                        df_events[dim_ref].values >= div_ranges[0]
                    )
                if div_ranges[1] is not None:
                    q_results = np.bitwise_and(
                        q_results,
                        df_events[dim_ref].values < div_ranges[1]
                    )

                results[quadrant.id] = q_results

        return results


class BooleanGate(Gate):
    """
    Represents a GatingML Boolean Gate.

    A BooleanGate performs the boolean operations AND, OR, or NOT on one or
    more other gates. Note, the boolean operation XOR is not supported in the
    GatingML specification but can be implemented using a combination of the
    supported operations.
    """
    def __init__(
            self,
            gate_name,
            bool_type,
            gate_refs
    ):
        """
        Create a BooleanGate instance.

        A gate reference is a dictionary containing keywords `ref`, `path`, and `complement`. The
        `ref` keyword is a text string referencing an existing gate ID. The `path` is a tuple of
        that gate's full gate path. The `complement` is a boolean flag specifying whether to use
        the boolean complement (i.e. "NOT").

        :param gate_name: text string for the name of the gate
        :param bool_type: string specifying boolean type. Accepted values: `and`, `or`, and `not` (case-insensitive)
        :param gate_refs: list of "gate reference" dictionaries (see above description of a gate reference)
        """
        super().__init__(
            gate_name,
            None
        )
        self.gate_type = "BooleanGate"

        bool_type = bool_type.lower()
        if bool_type not in ['and', 'or', 'not']:
            raise ValueError(
                "Boolean gate must specify one of 'and', 'or', or 'not'"
            )
        self.type = bool_type
        self.gate_refs = gate_refs

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.gate_name}, type: {self.type})'
        )

    def apply(self, df_events):
        """
        Apply gate to events in given pandas DataFrame. The given DataFrame must have columns matching the
        `ref` keys of the `gate_refs` list of dictionaries.

        :param df_events: pandas DataFrame with column labels matching gate_refs 'ref' keys
        :return: NumPy array of boolean values for each event (True is inside gate)
        """
        all_gate_results = []

        for gate_ref_dict in self.gate_refs:
            res_key = (gate_ref_dict['ref'], "/".join(gate_ref_dict['path']))
            gate_ref_events = df_events[res_key]

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

        return results
