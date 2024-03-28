"""
Module for GatingML gate classes

All GML gates are intended for internal use and exist to convert XML-based
GatingML elements to an intermediate Gate subclass. GML gates differ from
their parent class in that they retain a parent gate reference which is
used to assemble the gate tree. They also each provide a
`convert_to_parent_class` for converting them to their parent class for
public interaction in a GatingStrategy.
"""
from .. import gates
from ..._utils import xml_utils, xml_common


class GMLRectangleGate(gates.RectangleGate):
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
            use_complement=False
    ):
        gate_name, parent_gate_name, dimensions = xml_utils.parse_gate_element(
            gate_element,
            gating_namespace,
            data_type_namespace
        )
        self.parent = parent_gate_name
        self.use_complement = use_complement

        super().__init__(
            gate_name,
            dimensions,
            use_complement=self.use_complement
        )

    def convert_to_parent_class(self):
        """
        Convert to parent RectangleGate class.

        :return: RectangleGate
        """
        return gates.RectangleGate(self.gate_name, self.dimensions, use_complement=self.use_complement)


class GMLPolygonGate(gates.PolygonGate):
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
            use_complement=False
    ):
        gate_name, parent_gate_name, dimensions = xml_utils.parse_gate_element(
            gate_element,
            gating_namespace,
            data_type_namespace
        )
        self.parent = parent_gate_name

        vert_els = gate_element.findall(
            '%s:vertex' % gating_namespace,
            namespaces=gate_element.nsmap
        )

        vertices = []

        for vert_el in vert_els:
            vert = xml_utils.parse_vertex_element(vert_el, gating_namespace, data_type_namespace)
            vertices.append(vert)

        self.use_complement = use_complement

        super().__init__(
            gate_name,
            dimensions,
            vertices,
            use_complement=self.use_complement
        )

    def convert_to_parent_class(self):
        """
        Convert to parent PolygonGate class.

        :return: PolygonGate
        """
        return gates.PolygonGate(self.gate_name, self.dimensions, self.vertices, use_complement=self.use_complement)


class GMLEllipsoidGate(gates.EllipsoidGate):
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
            data_type_namespace
    ):
        gate_name, parent_gate_name, dimensions = xml_utils.parse_gate_element(
            gate_element,
            gating_namespace,
            data_type_namespace
        )
        self.parent = parent_gate_name

        # First, we'll get the center of the ellipse, contained in
        # a 'mean' element, that holds 2 'coordinate' elements
        mean_el = gate_element.find(
            '%s:mean' % gating_namespace,
            namespaces=gate_element.nsmap
        )

        coordinates = []

        coord_els = mean_el.findall(
            '%s:coordinate' % gating_namespace,
            namespaces=gate_element.nsmap
        )

        if len(coord_els) == 1:
            raise ValueError(
                'Ellipsoids must have at least 2 dimensions (line %d)' % gate_element.sourceline
            )

        for coord_el in coord_els:
            value = xml_common.find_attribute_value(coord_el, data_type_namespace, 'value')
            if value is None:
                raise ValueError(
                    'A coordinate must have only 1 value (line %d)' % coord_el.sourceline
                )

            coordinates.append(float(value))

        # Next, we'll parse the covariance matrix, containing 2 'row'
        # elements, each containing 2 'entry' elements w/ value attributes
        covariance_el = gate_element.find(
            '%s:covarianceMatrix' % gating_namespace,
            namespaces=gate_element.nsmap
        )

        covariance_matrix = []

        covariance_row_els = covariance_el.findall(
            '%s:row' % gating_namespace,
            namespaces=gate_element.nsmap
        )

        for row_el in covariance_row_els:
            row_entry_els = row_el.findall(
                '%s:entry' % gating_namespace,
                namespaces=gate_element.nsmap
            )

            entry_values = []
            for entry_el in row_entry_els:
                value = xml_common.find_attribute_value(entry_el, data_type_namespace, 'value')
                entry_values.append(float(value))

            if len(entry_values) != len(coordinates):
                raise ValueError(
                    'Covariance row entry value count must match # of dimensions (line %d)' % row_el.sourceline
                )

            covariance_matrix.append(entry_values)

        # Finally, get the distance square, which is a simple element w/
        # a single value attribute
        distance_square_el = gate_element.find(
            '%s:distanceSquare' % gating_namespace,
            namespaces=gate_element.nsmap
        )

        dist_square_value = xml_common.find_attribute_value(distance_square_el, data_type_namespace, 'value')
        distance_square = float(dist_square_value)

        super().__init__(
            gate_name,
            dimensions,
            coordinates,
            covariance_matrix,
            distance_square
        )

    def convert_to_parent_class(self):
        """
        Convert to parent EllipsoidGate class.

        :return: EllipsoidGate
        """
        return gates.EllipsoidGate(
            self.gate_name, self.dimensions, self.coordinates, self.covariance_matrix, self.distance_square
        )


class GMLQuadrantGate(gates.QuadrantGate):
    """
    Represents a GatingML Quadrant Gate

    A QuadrantGate must have at least 1 divider, and must specify the IDs
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
            data_type_namespace
    ):
        gate_name, parent_gate_name, dividers = xml_utils.parse_gate_element(
            gate_element,
            gating_namespace,
            data_type_namespace
        )
        self.parent = parent_gate_name

        # First, we'll check dimension count
        if len(dividers) < 1:
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

        quadrants = []

        for quadrant_el in quadrant_els:
            quad_id = xml_common.find_attribute_value(quadrant_el, gating_namespace, 'id')

            position_els = quadrant_el.findall(
                '%s:position' % gating_namespace,
                namespaces=gate_element.nsmap
            )

            divider_refs = []
            divider_ranges = []

            for pos_el in position_els:
                divider_ref = xml_common.find_attribute_value(pos_el, gating_namespace, 'divider_ref')
                location = xml_common.find_attribute_value(pos_el, gating_namespace, 'location')
                location = float(location)
                q_min = None
                q_max = None
                dim_id = None

                for div in dividers:
                    if div.id != divider_ref:
                        continue
                    else:
                        dim_id = div.dimension_ref

                    for v in sorted(div.values):
                        if v > location:
                            q_max = v

                            # once we have a max value, no need to
                            break
                        elif v <= location:
                            q_min = v

                if dim_id is None:
                    raise ValueError(
                        'Quadrant must define a divider reference (line %d)' % pos_el.sourceline
                    )

                divider_refs.append(divider_ref)
                divider_ranges.append((q_min, q_max))

            quadrants.append(
                gates.Quadrant(
                    quad_id,
                    divider_refs=divider_refs,
                    divider_ranges=divider_ranges
                )
            )

        super().__init__(
            gate_name,
            dividers,
            quadrants
        )

    def convert_to_parent_class(self):
        """
        Convert to parent QuadrantGate class.

        :return: QuadrantGate
        """
        return gates.QuadrantGate(self.gate_name, self.dimensions, self.quadrants.values())


class GMLBooleanGate(gates.BooleanGate):
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
            data_type_namespace
    ):
        gate_name, parent_gate_name, dimensions = xml_utils.parse_gate_element(
            gate_element,
            gating_namespace,
            data_type_namespace
        )
        self.parent = parent_gate_name
        
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
            bool_type = 'and'
            bool_op_el = and_els[0]
        elif len(or_els) > 0:
            bool_type = 'or'
            bool_op_el = or_els[0]
        elif len(not_els) > 0:
            bool_type = 'not'
            bool_op_el = not_els[0]
        else:
            raise ValueError(
                "Boolean gate must specify one of 'and', 'or', or 'not' (line %d)" % gate_element.sourceline
            )

        gate_ref_els = bool_op_el.findall(
            '%s:gateReference' % gating_namespace,
            namespaces=gate_element.nsmap
        )

        gate_refs = []

        for gate_ref_el in gate_ref_els:
            gate_ref = xml_common.find_attribute_value(gate_ref_el, gating_namespace, 'ref')
            if gate_ref is None:
                raise ValueError(
                    "Boolean gate reference must specify a 'ref' attribute (line %d)" % gate_ref_el.sourceline
                )

            use_complement = xml_common.find_attribute_value(gate_ref_el, gating_namespace, 'use-as-complement')
            if use_complement is not None:
                use_complement = use_complement == 'true'
            else:
                use_complement = False

            # TODO: see if 'gate_refs' list of dictionaries can be to something more easily documented
            gate_refs.append(
                {
                    'ref': gate_ref,
                    'complement': use_complement
                }
            )

        super().__init__(
            gate_name,
            bool_type,
            gate_refs
        )

    def convert_to_parent_class(self):
        """
        Convert to parent BooleanGate class.

        :return: BooleanGate
        """
        return gates.BooleanGate(self.gate_name, self.type, self.gate_refs)
