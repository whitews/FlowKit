"""
Module containing FlowJo compatible versions of gate classes
"""
import copy
import numpy as np
from ._base_gate import Gate
from ..gates import PolygonGate
from ..transforms import WSPBiexTransform
from ..._utils import xml_utils
from ...exceptions import FlowJoWSPParsingError


def _rotate_point_around_point(point, cov_mat, center_point=(0, 0)):
    """
    Rotates given point around center_point

    :param point: Coordinates of point to rotate
    :param cov_mat: Covariance matrix for the rotation
    :param center_point: Coordinates of the reference rotation point. Default is the origin (0, 0)

    :return: Rotated point coordinates
    """
    point_translated = np.array([point[0] - center_point[0], point[1] - center_point[1]])
    point_rot = np.dot(point_translated, cov_mat)
    point_untranslated = point_rot + center_point

    return point_untranslated


class WSPEllipsoidGate(Gate):
    """
    Represents a FlowJo workspace Ellipsoid Gate

    FlowJo ellipsoids are odd in that they specify the foci and 4 edge points on the
    ellipse (not the main vertices or co-vertices). They also store coordinates in
    the FlowJo "display space" of 256 x 256. The display space is binned after the
    transformation has been applied, so the multiplier to scale the coordinates is
    found by dividing the max transformed dimension range by 256.
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

        # First, check the dimensions for transformation references
        # There shouldn't be one, as FJ uses the dimension ID to lookup transforms.
        for dim in dimensions:
            if dim.transformation_ref is not None:
                raise NotImplementedError("FlowJo ellipse with transformations is not yet supported")

        # Get the foci points of the ellipse, contained in
        # a 'foci' element, that holds 2 'vertex' elements
        self.foci = self._parse_foci_elements(gate_element, gating_namespace, data_type_namespace)
        self.edge_vertices = self._parse_edge_elements(gate_element, gating_namespace, data_type_namespace)

        # save use_complement arg for later conversion to PolygonGate
        self.use_complement = use_complement

        super().__init__(
            gate_name,
            dimensions
        )

    @staticmethod
    def _parse_coordinate_elements(parent_element, gating_namespace, data_type_namespace):
        coord_els = parent_element.findall(
            '%s:coordinate' % gating_namespace,
            namespaces=parent_element.nsmap
        )

        coordinates = []

        for coord_el in coord_els:
            value = xml_utils.find_attribute_value(coord_el, data_type_namespace, 'value')
            if value is None:
                raise FlowJoWSPParsingError(
                    'A coordinate must have only 1 value (line %d)' % coord_el.sourceline
                )

            coordinates.append(float(value))

        return coordinates

    def _parse_foci_elements(self, gate_element, gating_namespace, data_type_namespace):
        foci_el = gate_element.find(
            '%s:foci' % gating_namespace,
            namespaces=gate_element.nsmap
        )

        foci = []

        foci_vertex_els = foci_el.findall(
            '%s:vertex' % gating_namespace,
            namespaces=gate_element.nsmap
        )

        if len(foci_vertex_els) <= 1:
            raise FlowJoWSPParsingError(
                'Ellipsoids must have at least 2 dimensions (line %d)' % gate_element.sourceline
            )

        for foci_vertex_el in foci_vertex_els:
            coordinates = self._parse_coordinate_elements(foci_vertex_el, gating_namespace, data_type_namespace)
            foci.append(np.array(coordinates))

        return np.array(foci)

    def _parse_edge_elements(self, gate_element, gating_namespace, data_type_namespace):
        # Next, we'll parse the edge vertices
        edge_el = gate_element.find(
            '%s:edge' % gating_namespace,
            namespaces=gate_element.nsmap
        )

        edge_vertices = []

        edge_vertex_els = edge_el.findall(
            '%s:vertex' % gating_namespace,
            namespaces=gate_element.nsmap
        )

        if len(edge_vertex_els) < 4:
            raise FlowJoWSPParsingError(
                'FlowJo ellipsoids must have 4 edge points (line %d)' % gate_element.sourceline
            )

        for edge_vertex_el in edge_vertex_els:
            coordinates = self._parse_coordinate_elements(edge_vertex_el, gating_namespace, data_type_namespace)
            edge_vertices.append(np.array(coordinates))

        return np.array(edge_vertices)

    def convert_to_polygon_gate(self, transforms, n_vertices=128):
        """
        Converts this WSPEllipsoidGate to a regular PolygonGate that is compatible for analysis in FlowKit
        :param transforms: list of transforms to use for gate dimensions (must be in same order as self.dimensions)
        :param n_vertices: number of polygon vertices to approximate the ellipse
        :return: PolygonGate instance
        """
        # FlowJo stores ellipsoid vertex values differently from any other gate.
        # They are stored in the binned "display space", so range from 0.0 - 256.0.
        # The binned space is linear over the transform range.
        #
        # To convert to a polygon:
        #     1. Determine center & rotation angle from foci
        #     2. Translate foci & edge vertices such that center is at origin
        #     3. Rotate foci & edge vertices such that major/minor axes are || to x/y axes
        #     4. Determine major axis orientation (x vs y-axis)
        #     5. Use foci & major axis to determine minor axis (2nd FJ point is unreliable)
        #     6. Generate new x, y points from ellipse definition for set of angles
        #     7. Rotate & translate coordinates back to original orientation
        #     8. Scale any dimensions using biex transform
        #     9. Create PolygonGate from the new set of coordinates
        # Find center of ellipse
        foci = copy.deepcopy(self.foci) / 256.0
        center = (foci[0] + foci[1]) / 2.0

        # Determine rotation of ellipse
        slope = (foci[1][1] - foci[0][1]) / (foci[1][0] - foci[0][0])
        theta_rad = np.arctan(slope)
        cos, sin = np.cos(theta_rad), np.sin(theta_rad)
        r = np.array(((cos, -sin), (sin, cos)))

        # Translate foci & edge vertices to the origin
        foci_origin = foci - center
        edge_vertices_origin = (copy.deepcopy(self.edge_vertices) / 256.0) - center

        # According to FlowJo devs, edge vertices are ordered as:
        #     1st & 2nd points are major axis
        #     3rd & 4th points are minor axis
        # Rotate edge vertices
        # Only need are one major & one minor point since the other is symmetric
        foci_rotated = _rotate_point_around_point(foci_origin[0], r)
        rv1 = _rotate_point_around_point(edge_vertices_origin[0], r)
        rv3 = _rotate_point_around_point(edge_vertices_origin[2], r)

        # However, I don't trust that the 1st point is always the major
        # axis or if it is always on x or y, so we'll make sure.
        # Use absolute values & find max
        rv1 = np.abs(rv1)
        rv3 = np.abs(rv3)
        rv1_max_pos = rv1.argmax()
        rv3_max_pos = rv3.argmax()

        if rv1_max_pos == rv3_max_pos:
            raise FlowJoWSPParsingError(
                "Cannot determine major axis of FlowJo ellipse gate '%s'" % self.gate_name
            )

        rv1_max_val = rv1[rv1_max_pos]
        rv3_max_val = rv3[rv3_max_pos]

        if rv1_max_val >= rv3_max_val:
            # rv1 is major axis (even if a circle)
            a = rv1_max_val
        else:
            # rv3 is major axis
            a = rv3_max_val

        # Also, calculate b from foci and found 'a', since the
        # minor vertex stored by FlowJo seems off
        b = np.sqrt(np.abs((foci_rotated[0]) ** 2 - (a ** 2)))

        # Calculate set of angles for getting points on ellipse
        angles = [2 * np.pi * (i / n_vertices) for i in range(n_vertices)]

        # Calculate x, y coordinates for each of the angles
        # x = a * cos(θ)
        # y = b * sin(θ)
        if rv1_max_pos == 0:
            # major axis is the x-axis
            x = a * np.cos(angles)
            y = b * np.sin(angles)
        else:
            # minor axis is the x-axis
            x = b * np.cos(angles)
            y = a * np.sin(angles)

        # rotate ellipse to the original orientation, then translate
        inv_r = np.linalg.inv(r)
        xy = np.vstack([x, y]).T

        # this will be the final set of polygon vertices
        xy_rot_trans = np.dot(xy, inv_r) + center

        # the final complication is the different scaling of biex transforms
        for i, xform in enumerate(transforms):
            if isinstance(xform, WSPBiexTransform):
                # biex transform is always scaled from 0-4096
                xform_range = 4096.0
            else:
                # all others are scaled from 0-1
                xform_range = 1.0

            xy_rot_trans[:, i] *= xform_range

        return PolygonGate(self.gate_name, self.dimensions, xy_rot_trans, use_complement=self.use_complement)

    def apply(self, df_events):
        """
        WSPEllipsoid gate is not intended to be used externally & does not support apply()
        :param df_events:
        """
        pass
