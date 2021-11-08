"""
Module containing FlowJo compatible versions of gate classes
"""
import copy
import numpy as np
from ._base_gate import Gate
from ..gates import EllipsoidGate
from ..._utils import gate_utils, xml_utils


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
            data_type_namespace
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

        super().__init__(
            gate_name,
            parent_gate_name,
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
                raise ValueError(
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
            raise ValueError(
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
            raise ValueError(
                'FlowJo ellipsoids must have 4 edge points (line %d)' % gate_element.sourceline
            )

        for edge_vertex_el in edge_vertex_els:
            coordinates = self._parse_coordinate_elements(edge_vertex_el, gating_namespace, data_type_namespace)
            edge_vertices.append(np.array(coordinates))

        return np.array(edge_vertices)

    def convert_to_ellipsoid_gate(self, transforms):
        """
        Converts this WSPEllipsoidGate to a regular EllipsoidGate that is compatible for analysis in FlowKit
        :param transforms: list of transforms to use for gate dimensions (must be in same order as self.dimensions)
        :return: EllipsoidGate instance
        """
        foci = copy.deepcopy(self.foci)
        edge_vertices = copy.deepcopy(self.edge_vertices)

        if transforms is not None:
            for coord in foci:
                for i, value in enumerate(coord):
                    # first inverse transform from the FJ "display space"
                    untransformed_coord = transforms[i].inverse(np.array([float(value)]) / 256.0)[0]

                    # now, we're back in untransformed data space, and can apply the transform
                    coord[i] = transforms[i].apply(np.array([untransformed_coord]))[0]

            for coord in edge_vertices:
                for i, value in enumerate(coord):
                    # first inverse transform from the FJ "display space"
                    untransformed_coord = transforms[i].inverse(np.array([float(value)]) / 256.0)[0]

                    # now, we're back in untransformed data space, and can apply the transform
                    coord[i] = transforms[i].apply(np.array([[untransformed_coord]]))[0]

        center = (foci[0] + foci[1]) / 2.0

        # need to rotate points on ellipse
        slope = (foci[1][1] - foci[0][1]) / (foci[1][0] - foci[0][0])
        theta_rad = np.arctan(slope)
        cos, sin = np.cos(theta_rad), np.sin(theta_rad)
        r = np.array(((cos, -sin), (sin, cos)))

        rv1 = gate_utils.rotate_point_around_point(edge_vertices[0], r, center)
        rv3 = gate_utils.rotate_point_around_point(edge_vertices[2], r, center)

        # (((x - cx) ** 2) / a ** 2) + (((y - cy) ** 2) / b ** 2) = 1
        # let:
        #     m = 1 / a^2
        #     n = 1 / b^2
        #     xd2 = (x - cx)^2
        #     yd2 = (y - cy)^2
        #
        # then:
        # xd2 * m + yd2 * n = a
        tv1 = rv1 - center
        tv3 = rv3 - center

        a = [tv1 ** 2, tv3 ** 2]
        b = [1., 1.]

        m, n = np.linalg.solve(a, b)
        w = 2 * np.sqrt(1. / abs(m))
        h = 2 * np.sqrt(1. / abs(n))

        # calculate covariance matrix
        w_h_array = np.array([w, h])
        eig_values = (w_h_array / 2.) ** 2

        inv_r = np.linalg.inv(r)
        diag_mat = np.diag(eig_values)
        cov_mat = r.dot(diag_mat).dot(inv_r)
        distance_square = 1.

        return EllipsoidGate(self.gate_name, self.parent, self.dimensions, center, cov_mat, distance_square)

    def apply(self, df_events):
        """
        WSPEllipsoid gate is not intended to be used externally & does not support apply()
        :param df_events:
        """
        pass
