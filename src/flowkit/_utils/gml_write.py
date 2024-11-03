"""
Utility functions for writing GatingML 2.0 documents
"""
import numpy as np
from lxml import etree
from .._models.dimension import RatioDimension, QuadrantDivider
# noinspection PyProtectedMember
from .._models.transforms import _transforms
# noinspection PyProtectedMember
from .._models.gates._gates import \
    BooleanGate, \
    EllipsoidGate, \
    QuadrantGate, \
    PolygonGate, \
    RectangleGate
from ..exceptions import QuadrantReferenceError


def _add_matrix_to_gml(root, matrix_id, matrix, ns_map):
    xform_ml = etree.SubElement(root, "{%s}spectrumMatrix" % ns_map['transforms'])
    xform_ml.set('{%s}id' % ns_map['transforms'], matrix_id)

    fluoros_ml = etree.SubElement(xform_ml, "{%s}fluorochromes" % ns_map['transforms'])

    for fluoro in matrix.fluorochromes:
        fluoro_ml = etree.SubElement(fluoros_ml, '{%s}fcs-dimension' % ns_map['data-type'])
        fluoro_ml.set('{%s}name' % ns_map['data-type'], fluoro)

    detectors_ml = etree.SubElement(xform_ml, "{%s}detectors" % ns_map['transforms'])

    for detector in matrix.detectors:
        detector_ml = etree.SubElement(detectors_ml, '{%s}fcs-dimension' % ns_map['data-type'])
        detector_ml.set('{%s}name' % ns_map['data-type'], detector)

    for row in matrix.matrix:
        row_ml = etree.SubElement(xform_ml, "{%s}spectrum" % ns_map['transforms'])
        for val in row:
            coefficient_ml = etree.SubElement(row_ml, "{%s}coefficient" % ns_map['transforms'])
            coefficient_ml.set('{%s}value' % ns_map['transforms'], str(val))


def _add_transform_to_gml(root, transform_id, transform, ns_map):
    xform_ml = etree.SubElement(root, "{%s}transformation" % ns_map['transforms'])
    xform_ml.set('{%s}id' % ns_map['transforms'], transform_id)

    if isinstance(transform, _transforms.RatioTransform):
        ratio_ml = etree.SubElement(xform_ml, "{%s}fratio" % ns_map['transforms'])
        ratio_ml.set('{%s}A' % ns_map['transforms'], str(transform.param_a))
        ratio_ml.set('{%s}B' % ns_map['transforms'], str(transform.param_b))
        ratio_ml.set('{%s}C' % ns_map['transforms'], str(transform.param_c))

        for dim in transform.dimensions:
            fcs_dim_ml = etree.SubElement(ratio_ml, '{%s}fcs-dimension' % ns_map['data-type'])
            fcs_dim_ml.set('{%s}name' % ns_map['data-type'], dim)
    elif isinstance(transform, _transforms.LogTransform):
        log_ml = etree.SubElement(xform_ml, "{%s}flog" % ns_map['transforms'])
        log_ml.set('{%s}T' % ns_map['transforms'], str(transform.param_t))
        log_ml.set('{%s}M' % ns_map['transforms'], str(transform.param_m))
    elif isinstance(transform, _transforms.AsinhTransform):
        asinh_ml = etree.SubElement(xform_ml, "{%s}fasinh" % ns_map['transforms'])
        asinh_ml.set('{%s}T' % ns_map['transforms'], str(transform.param_t))
        asinh_ml.set('{%s}M' % ns_map['transforms'], str(transform.param_m))
        asinh_ml.set('{%s}A' % ns_map['transforms'], str(transform.param_a))
    elif isinstance(transform, _transforms.LogicleTransform):
        logicle_ml = etree.SubElement(xform_ml, "{%s}logicle" % ns_map['transforms'])
        logicle_ml.set('{%s}T' % ns_map['transforms'], str(transform.param_t))
        logicle_ml.set('{%s}W' % ns_map['transforms'], str(transform.param_w))
        logicle_ml.set('{%s}M' % ns_map['transforms'], str(transform.param_m))
        logicle_ml.set('{%s}A' % ns_map['transforms'], str(transform.param_a))
    elif isinstance(transform, _transforms.HyperlogTransform):
        hyperlog_ml = etree.SubElement(xform_ml, "{%s}hyperlog" % ns_map['transforms'])
        hyperlog_ml.set('{%s}T' % ns_map['transforms'], str(transform.param_t))
        hyperlog_ml.set('{%s}W' % ns_map['transforms'], str(transform.param_w))
        hyperlog_ml.set('{%s}M' % ns_map['transforms'], str(transform.param_m))
        hyperlog_ml.set('{%s}A' % ns_map['transforms'], str(transform.param_a))
    elif isinstance(transform, _transforms.LinearTransform):
        lin_ml = etree.SubElement(xform_ml, "{%s}flin" % ns_map['transforms'])
        lin_ml.set('{%s}T' % ns_map['transforms'], str(transform.param_t))
        lin_ml.set('{%s}A' % ns_map['transforms'], str(transform.param_a))


def _add_gate_to_gml(root, gate, ns_map):
    if isinstance(gate, RectangleGate):
        gate_ml = etree.SubElement(root, "{%s}RectangleGate" % ns_map['gating'])
    elif isinstance(gate, PolygonGate):
        gate_ml = etree.SubElement(root, "{%s}PolygonGate" % ns_map['gating'])

        for v in gate.vertices:
            vert_ml = etree.SubElement(gate_ml, '{%s}vertex' % ns_map['gating'])
            for coord in v:
                coord_ml = etree.SubElement(vert_ml, '{%s}coordinate' % ns_map['gating'])
                coord_ml.set('{%s}value' % ns_map['data-type'], str(coord))
    elif isinstance(gate, BooleanGate):
        gate_ml = etree.SubElement(root, "{%s}BooleanGate" % ns_map['gating'])

        if gate.type == 'and':
            bool_type_ml = etree.SubElement(gate_ml, '{%s}and' % ns_map['gating'])
        elif gate.type == 'or':
            bool_type_ml = etree.SubElement(gate_ml, '{%s}or' % ns_map['gating'])
        elif gate.type == 'not':
            bool_type_ml = etree.SubElement(gate_ml, '{%s}not' % ns_map['gating'])
        else:
            raise ValueError("Boolean gate type '%s' is not valid" % gate.type)

        for gate_ref in gate.gate_refs:
            gate_ref_ml = etree.SubElement(bool_type_ml, '{%s}gateReference' % ns_map['gating'])
            gate_ref_ml.set('{%s}ref' % ns_map['gating'], gate_ref['ref'])
            if gate_ref['complement']:
                gate_ref_ml.set('{%s}use-as-complement' % ns_map['gating'], "true")

    elif isinstance(gate, EllipsoidGate):
        gate_ml = etree.SubElement(root, "{%s}EllipsoidGate" % ns_map['gating'])
        mean_ml = etree.SubElement(gate_ml, '{%s}mean' % ns_map['gating'])
        cov_ml = etree.SubElement(gate_ml, '{%s}covarianceMatrix' % ns_map['gating'])
        dist_square_ml = etree.SubElement(gate_ml, '{%s}distanceSquare' % ns_map['gating'])
        dist_square_ml.set('{%s}value' % ns_map['data-type'], str(gate.distance_square))

        for c in gate.coordinates:
            coord_ml = etree.SubElement(mean_ml, '{%s}coordinate' % ns_map['gating'])
            coord_ml.set('{%s}value' % ns_map['data-type'], str(c))

        for row in gate.covariance_matrix:
            row_ml = etree.SubElement(cov_ml, '{%s}row' % ns_map['gating'])

            for val in row:
                entry_ml = etree.SubElement(row_ml, '{%s}entry' % ns_map['gating'])
                entry_ml.set('{%s}value' % ns_map['data-type'], str(val))
    elif isinstance(gate, QuadrantGate):
        gate_ml = etree.SubElement(root, "{%s}QuadrantGate" % ns_map['gating'])

        for q_id, quadrant in gate.quadrants.items():
            quad_ml = etree.SubElement(gate_ml, '{%s}Quadrant' % ns_map['gating'])
            quad_ml.set('{%s}id' % ns_map['gating'], q_id)

            for div_ref in quadrant.divider_refs:
                pos_ml = etree.SubElement(quad_ml, '{%s}position' % ns_map['gating'])
                pos_ml.set('{%s}divider_ref' % ns_map['gating'], div_ref)

                div_ranges = quadrant.get_divider_range(div_ref)
                if div_ranges[0] is None:
                    loc_coord = div_ranges[1] / 2.0
                elif div_ranges[1] is None:
                    loc_coord = div_ranges[0] * 2.0
                else:
                    loc_coord = np.mean(div_ranges)

                pos_ml.set('{%s}location' % ns_map['gating'], str(loc_coord))
    else:
        raise ValueError("Gate %s is not a valid GatingML 2.0 element" % gate.gate_name)

    gate_ml.set('{%s}id' % ns_map['gating'], gate.gate_name)

    for i, dim in enumerate(gate.dimensions):
        dim_type = 'dim'

        if isinstance(dim, QuadrantDivider):
            dim_ml = etree.Element('{%s}divider' % ns_map['gating'])
            dim_ml.set('{%s}id' % ns_map['gating'], dim.id)
            dim_type = 'quad'
        elif isinstance(dim, RatioDimension):
            dim_ml = etree.Element('{%s}dimension' % ns_map['gating'])
            dim_type = 'ratio'
        else:
            dim_ml = etree.Element('{%s}dimension' % ns_map['gating'])

        gate_ml.insert(i, dim_ml)

        if dim.compensation_ref is not None:
            dim_ml.set('{%s}compensation-ref' % ns_map['gating'], dim.compensation_ref)
        if dim.transformation_ref is not None:
            dim_ml.set('{%s}transformation-ref' % ns_map['gating'], dim.transformation_ref)

        if dim_type != 'quad':
            if dim.min is not None:
                dim_ml.set('{%s}min' % ns_map['gating'], str(dim.min))
            if dim.max is not None:
                dim_ml.set('{%s}max' % ns_map['gating'], str(dim.max))

        if dim_type == 'ratio':
            new_dim_el = etree.SubElement(dim_ml, '{%s}new-dimension' % ns_map['data-type'])
            new_dim_el.set('{%s}transformation-ref' % ns_map['data-type'], dim.ratio_ref)
        else:
            fcs_dim_ml = etree.SubElement(dim_ml, '{%s}fcs-dimension' % ns_map['data-type'])
            if dim_type == 'dim':
                fcs_dim_ml.set('{%s}name' % ns_map['data-type'], dim.id)
            elif dim_type == 'quad':
                fcs_dim_ml.set('{%s}name' % ns_map['data-type'], dim.dimension_ref)
                for val in dim.values:
                    value_ml = etree.SubElement(dim_ml, '{%s}value' % ns_map['gating'])
                    value_ml.text = str(val)

    return gate_ml


def _add_gates_from_gate_dict(gating_strategy, gate_dict, ns_map, parent_ml, sample_id=None):
    # the gate_dict will have keys 'name' and 'children'. top-level 'name' value is 'root'
    for child in gate_dict['children']:
        gate_id = child['name']

        try:
            gate = gating_strategy.get_gate(gate_id, sample_id=sample_id)
        except QuadrantReferenceError:
            # single quadrants will be handled in the owning quadrant gate
            gate = None

        if gate is not None:
            child_ml = _add_gate_to_gml(parent_ml, gate, ns_map)

            if gate_dict['name'] != 'root':
                # this is a recursion, add the parent reference
                child_ml.set('{%s}parent_id' % ns_map['gating'], gate_dict['name'])

        if 'children' in child:  # and not isinstance(gate, QuadrantGate):
            _add_gates_from_gate_dict(gating_strategy, child, ns_map, parent_ml, sample_id=sample_id)


def export_gatingml(gating_strategy, file_handle, sample_id=None):
    """
    Exports a valid GatingML 2.0 document from given GatingStrategy instance.
    Specify the sample ID to use that sample's custom gates in the exported
    file, otherwise the template gates will be exported.

    :param gating_strategy: A GatingStrategy instance
    :param file_handle: File handle for exported GatingML 2.0 document
    :param sample_id: an optional text string representing a Sample instance
    :return: None
    """
    ns_g = "http://www.isac-net.org/std/Gating-ML/v2.0/gating"
    ns_dt = "http://www.isac-net.org/std/Gating-ML/v2.0/datatypes"
    ns_xform = "http://www.isac-net.org/std/Gating-ML/v2.0/transformations"
    ns_map = {
        'gating': ns_g,
        'data-type': ns_dt,
        'transforms': ns_xform
    }

    root = etree.Element('{%s}Gating-ML' % ns_g, nsmap=ns_map)

    # process gating strategy transformations
    for xform_id, xform in gating_strategy.transformations.items():
        _add_transform_to_gml(root, xform_id, xform, ns_map)

    # process gating strategy compensation matrices
    for matrix_id, matrix in gating_strategy.comp_matrices.items():
        _add_matrix_to_gml(root, matrix_id, matrix, ns_map)

    # get gate hierarchy as a dictionary
    gate_dict = gating_strategy.get_gate_hierarchy('dict')

    # recursively convert all gates to GatingML
    _add_gates_from_gate_dict(gating_strategy, gate_dict, ns_map, root, sample_id=sample_id)

    et = etree.ElementTree(root)

    et.write(file_handle, encoding="utf-8", xml_declaration=True, pretty_print=True)
