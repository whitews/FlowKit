"""
Utility functions for parsing & exporting gating related XML documents
"""
import numpy as np
from lxml import etree
import anytree
from .._resources import gml_schema
from .._models.dimension import Dimension, RatioDimension, QuadrantDivider
from .._models.vertex import Vertex
from .._models.gating_strategy import GatingStrategy
from .._models.transforms import _transforms, _gml_transforms
from .._models.transforms._matrix import Matrix
from .._models.gates._gml_gates import \
    GMLBooleanGate, \
    GMLEllipsoidGate, \
    GMLQuadrantGate, \
    GMLPolygonGate, \
    GMLRectangleGate
from .._models.gates._gates import \
    BooleanGate, \
    EllipsoidGate, \
    Quadrant, \
    QuadrantGate, \
    PolygonGate, \
    RectangleGate


# map GatingML gate keys to our GML gate classes
gate_constructor_lut = {
    'RectangleGate': GMLRectangleGate,
    'PolygonGate': GMLPolygonGate,
    'EllipsoidGate': GMLEllipsoidGate,
    'QuadrantGate': GMLQuadrantGate,
    'BooleanGate': GMLBooleanGate
}


def _build_hierarchy_tree(gates_dict):
    nodes = {}

    root = anytree.Node('root')
    parent_gates = set()

    for g_id, gate in gates_dict.items():
        if gate.parent is not None:
            # record the set of parents so we can find the leaves later
            parent_gates.add(gate.parent)

            # we'll get children nodes after
            continue

        # use g_id for lookup (for uid), gate.id for Node name
        nodes[g_id] = anytree.Node(
            g_id,
            parent=root,
            gate=gate
        )

        if isinstance(gate, QuadrantGate):
            for q_id, quad in gate.quadrants.items():
                nodes[q_id] = anytree.Node(
                    q_id,
                    parent=nodes[gate.id],
                    gate=quad
                )

    parent_gates.difference_update(set(nodes.keys()))
    # Now we have all the root-level nodes and the set of parents.
    # Process the parent gates
    parents_remaining = len(parent_gates)
    while parents_remaining > 0:
        discard = []
        for gate_id in parent_gates:
            gate = gates_dict[gate_id]
            parent_id = gate.parent

            if parent_id in nodes or parent_id is None:
                nodes[gate_id] = anytree.Node(
                    gate_id,
                    parent=root if parent_id is None else nodes[parent_id],
                    gate=gate
                )

                if isinstance(gate, QuadrantGate):
                    for q_id, quad in gate.quadrants.items():
                        nodes[q_id] = anytree.Node(
                            q_id,
                            parent=nodes[gate.id],
                            gate=quad
                        )
                discard.append(gate_id)
        parent_gates.difference_update(set(discard))
        parents_remaining = len(parent_gates)

    # Process remaining leaves
    for g_id, gate in gates_dict.items():
        if g_id in nodes:
            continue

        nodes[g_id] = anytree.Node(
            g_id,
            parent=nodes[gate.parent],
            gate=gate
        )

        # TODO: this conditional isn't covered in tests
        if isinstance(gate, QuadrantGate):
            for q_id, quad in gate.quadrants.items():
                nodes[q_id] = anytree.Node(
                    q_id,
                    parent=nodes[g_id],
                    gate=quad
                )

    return root


def parse_gating_xml(xml_file_or_path):
    doc_type, root_gml, gating_ns, data_type_ns, xform_ns = _get_xml_type(xml_file_or_path)

    gating_strategy = GatingStrategy()

    if doc_type == 'gatingml':
        gates = _construct_gates(root_gml, gating_ns, data_type_ns)
        transformations = _construct_transforms(root_gml, xform_ns, data_type_ns)
        comp_matrices = _construct_matrices(root_gml, xform_ns, data_type_ns)
    elif doc_type == 'flowjo':
        raise ValueError("File is a FlowJo workspace, use parse_wsp or Session.import_flowjo_workspace.")
    else:
        raise ValueError("Gating file format is not supported.")

    for c_id, c in comp_matrices.items():
        gating_strategy.add_comp_matrix(c)
    for t_id, t in transformations.items():
        gating_strategy.add_transform(t)

    root = _build_hierarchy_tree(gates)

    for n in root.descendants:
        if isinstance(n.gate, Quadrant):
            continue
        gating_strategy.add_gate(n.gate)

    return gating_strategy


def _get_xml_type(xml_file_or_path):
    xml_document = etree.parse(xml_file_or_path)

    val = gml_schema.validate(xml_document)
    root = xml_document.getroot()

    if val:
        doc_type = 'gatingml'
    else:
        # Try parsing as a FlowJo workspace
        if 'flowJoVersion' in root.attrib:
            if int(root.attrib['flowJoVersion'].split('.')[0]) >= 10:
                doc_type = 'flowjo'
            else:
                raise ValueError("FlowKit only supports FlowJo workspaces for version 10 or higher.")
        else:
            raise ValueError("File is neither GatingML 2.0 compliant nor a FlowJo workspace.")

    gating_ns = None
    data_type_ns = None
    transform_ns = None

    # find GatingML target namespace in the map
    for ns, url in root.nsmap.items():
        if url == 'http://www.isac-net.org/std/Gating-ML/v2.0/gating':
            gating_ns = ns
        elif url == 'http://www.isac-net.org/std/Gating-ML/v2.0/datatypes':
            data_type_ns = ns
        elif url == 'http://www.isac-net.org/std/Gating-ML/v2.0/transformations':
            transform_ns = ns

    if gating_ns is None:
        raise ValueError("GatingML namespace reference is missing from GatingML file")

    return doc_type, root, gating_ns, data_type_ns, transform_ns


def _construct_gates(root_gml, gating_ns, data_type_ns):
    gates_dict = {}

    for gate_str, gate_class in gate_constructor_lut.items():
        gt_gates = root_gml.findall(
            ':'.join([gating_ns, gate_str]),
            root_gml.nsmap
        )

        for gt_gate in gt_gates:
            g = gate_class(
                gt_gate,
                gating_ns,
                data_type_ns
            )

            if g.id in gates_dict:
                raise ValueError(
                    "Gate '%s' already exists. "
                    "Duplicate gate IDs are not allowed." % g.id
                )
            gates_dict[g.id] = g

    return gates_dict


def _construct_transforms(root_gml, transform_ns, data_type_ns):
    transformations = {}

    if transform_ns is not None:
        # types of transforms include:
        #   - ratio
        #   - log10
        #   - asinh
        #   - hyperlog
        #   - linear
        #   - logicle
        xform_els = root_gml.findall(
            '%s:transformation' % transform_ns,
            namespaces=root_gml.nsmap
        )

        for xform_el in xform_els:
            xform = None

            # determine type of transformation
            fratio_els = xform_el.findall(
                '%s:fratio' % transform_ns,
                namespaces=root_gml.nsmap
            )

            if len(fratio_els) > 0:
                xform = _gml_transforms.RatioGMLTransform(
                    xform_el,
                    transform_ns,
                    data_type_ns
                )

            flog_els = xform_el.findall(
                '%s:flog' % transform_ns,
                namespaces=root_gml.nsmap
            )

            if len(flog_els) > 0:
                xform = _gml_transforms.LogGMLTransform(
                    xform_el,
                    transform_ns
                )

            fasinh_els = xform_el.findall(
                '%s:fasinh' % transform_ns,
                namespaces=root_gml.nsmap
            )

            if len(fasinh_els) > 0:
                xform = _gml_transforms.AsinhGMLTransform(
                    xform_el,
                    transform_ns
                )

            hyperlog_els = xform_el.findall(
                '%s:hyperlog' % transform_ns,
                namespaces=root_gml.nsmap
            )

            if len(hyperlog_els) > 0:
                xform = _gml_transforms.HyperlogGMLTransform(
                    xform_el,
                    transform_ns
                )

            flin_els = xform_el.findall(
                '%s:flin' % transform_ns,
                namespaces=root_gml.nsmap
            )

            if len(flin_els) > 0:
                xform = _gml_transforms.LinearGMLTransform(
                    xform_el,
                    transform_ns
                )

            logicle_els = xform_el.findall(
                '%s:logicle' % transform_ns,
                namespaces=root_gml.nsmap
            )

            if len(logicle_els) > 0:
                xform = _gml_transforms.LogicleGMLTransform(
                    xform_el,
                    transform_ns
                )

            if xform is not None:
                transformations[xform.id] = xform

    return transformations


def _construct_matrices(root_gml, transform_ns, data_type_ns):
    comp_matrices = {}

    if transform_ns is not None:
        # comp matrices are defined by the 'spectrumMatrix' element
        matrix_els = root_gml.findall(
            '%s:spectrumMatrix' % transform_ns,
            namespaces=root_gml.nsmap
        )

        for matrix_el in matrix_els:
            matrix = parse_matrix_element(
                matrix_el,
                transform_ns,
                data_type_ns
            )

            comp_matrices[matrix.id] = matrix

    return comp_matrices


def find_attribute_value(xml_el, namespace, attribute_name):
    attribs = xml_el.xpath(
        '@%s:%s' % (namespace, attribute_name),
        namespaces=xml_el.nsmap
    )

    if len(attribs) > 1:
        raise ValueError(
            "Multiple %s attributes found (line %d)" % (
                attribute_name, xml_el.sourceline
            )
        )
    elif len(attribs) == 0:
        return None

    return attribs[0]


def parse_gate_element(
        gate_element,
        gating_namespace,
        data_type_namespace
):
    gate_id = find_attribute_value(gate_element, gating_namespace, 'id')
    parent_id = find_attribute_value(gate_element, gating_namespace, 'parent_id')

    # most gates specify dimensions in the 'dimension' tag,
    # but quad gates specify dimensions in the 'divider' tag
    div_els = gate_element.findall(
        '%s:divider' % gating_namespace,
        namespaces=gate_element.nsmap
    )

    dimensions = []  # may actually be a list of dividers

    if len(div_els) == 0:
        dim_els = gate_element.findall(
            '%s:dimension' % gating_namespace,
            namespaces=gate_element.nsmap
        )

        dimensions = []

        for dim_el in dim_els:
            dim = parse_dimension_element(dim_el, gating_namespace, data_type_namespace)
            dimensions.append(dim)
    else:
        for div_el in div_els:
            dim = parse_divider_element(div_el, gating_namespace, data_type_namespace)
            dimensions.append(dim)

    return gate_id, parent_id, dimensions


def parse_dimension_element(
        dim_element,
        gating_namespace,
        data_type_namespace
):
    compensation_ref = find_attribute_value(dim_element, gating_namespace, 'compensation-ref')
    transformation_ref = find_attribute_value(dim_element, gating_namespace, 'transformation-ref')

    range_min = None
    range_max = None

    # should be 0 or only 1 'min' attribute (same for 'max')
    _min = find_attribute_value(dim_element, gating_namespace, 'min')
    _max = find_attribute_value(dim_element, gating_namespace, 'max')

    if _min is not None:
        range_min = float(_min)
    if _max is not None:
        range_max = float(_max)

    # label be here
    fcs_dim_el = dim_element.find(
        '%s:fcs-dimension' % data_type_namespace,
        namespaces=dim_element.nsmap
    )

    # if no 'fcs-dimension' element is present, this might be a
    # 'new-dimension'  made from a transformation on other dims
    if fcs_dim_el is None:
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
        ratio_xform_ref = find_attribute_value(new_dim_el, data_type_namespace, 'transformation-ref')

        if ratio_xform_ref is None:
            raise ValueError(
                "New dimensions must provid a transform reference (line %d)" % dim_element.sourceline
            )
        dimension = RatioDimension(
            ratio_xform_ref,
            compensation_ref,
            transformation_ref,
            range_min=range_min,
            range_max=range_max
        )
    else:
        label = find_attribute_value(fcs_dim_el, data_type_namespace, 'name')
        if label is None:
            raise ValueError(
                'Dimension name not found (line %d)' % fcs_dim_el.sourceline
            )

        dimension = Dimension(
            label,
            compensation_ref,
            transformation_ref,
            range_min=range_min,
            range_max=range_max
        )

    return dimension


def parse_divider_element(divider_element, gating_namespace, data_type_namespace):
    # Get'id' (present in quad gate dividers)
    dimension_id = find_attribute_value(divider_element, gating_namespace, 'id')

    compensation_ref = find_attribute_value(divider_element, gating_namespace, 'compensation-ref')
    transformation_ref = find_attribute_value(divider_element, gating_namespace, 'transformation-ref')

    # label be here
    fcs_dim_el = divider_element.find(
        '%s:fcs-dimension' % data_type_namespace,
        namespaces=divider_element.nsmap
    )

    label = find_attribute_value(fcs_dim_el, data_type_namespace, 'name')
    if label is None:
        raise ValueError(
            'Divider dimension name not found (line %d)' % fcs_dim_el.sourceline
        )

    values = []  # quad gate dims can have multiple values

    # values in gating namespace, ok if not present
    value_els = divider_element.findall(
        '%s:value' % gating_namespace,
        namespaces=divider_element.nsmap
    )

    for value in value_els:
        values.append(float(value.text))

    divider = QuadrantDivider(dimension_id, label, compensation_ref, values, transformation_ref)

    return divider


def parse_vertex_element(vert_element, gating_namespace, data_type_namespace):
    coordinates = []

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
        value = find_attribute_value(coord_el, data_type_namespace, 'value')
        if value is None:
            raise ValueError(
                'Vertex coordinate must have only 1 value (line %d)' % coord_el.sourceline
            )

        coordinates.append(float(value))

    return Vertex(coordinates)


def parse_matrix_element(
    matrix_element,
    xform_namespace,
    data_type_namespace
):
    matrix_id = find_attribute_value(matrix_element, xform_namespace, 'id')
    fluorochomes = []
    detectors = []
    matrix = []

    fluoro_el = matrix_element.find(
        '%s:fluorochromes' % xform_namespace,
        namespaces=matrix_element.nsmap
    )

    fcs_dim_els = fluoro_el.findall(
        '%s:fcs-dimension' % data_type_namespace,
        namespaces=matrix_element.nsmap
    )

    for dim_el in fcs_dim_els:
        label = find_attribute_value(dim_el, data_type_namespace, 'name')

        if label is None:
            raise ValueError(
                'Dimension name not found (line %d)' % dim_el.sourceline
            )
        fluorochomes.append(label)

    detectors_el = matrix_element.find(
        '%s:detectors' % xform_namespace,
        namespaces=matrix_element.nsmap
    )

    fcs_dim_els = detectors_el.findall(
        '%s:fcs-dimension' % data_type_namespace,
        namespaces=matrix_element.nsmap
    )

    for dim_el in fcs_dim_els:
        label = find_attribute_value(dim_el, data_type_namespace, 'name')

        if label is None:
            raise ValueError(
                'Dimension name not found (line %d)' % dim_el.sourceline
            )
        detectors.append(label)

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
            value = find_attribute_value(co_el, xform_namespace, 'value')
            if value is None:
                raise ValueError(
                    'Matrix coefficient must have only 1 value (line %d)' % co_el.sourceline
                )

            matrix_row.append(float(value))

        matrix.append(matrix_row)

    matrix = np.array(matrix)

    return Matrix(matrix_id, matrix, detectors, fluorochomes)


def _add_matrix_to_gml(root, matrix, ns_map):
    xform_ml = etree.SubElement(root, "{%s}spectrumMatrix" % ns_map['transforms'])
    xform_ml.set('{%s}id' % ns_map['transforms'], matrix.id)

    fluoros_ml = etree.SubElement(xform_ml, "{%s}fluorochromes" % ns_map['transforms'])

    for fluoro in matrix.fluorochomes:
        fluoro_ml = etree.SubElement(fluoros_ml, '{%s}fcs-dimension' % ns_map['data-type'])
        fluoro_ml.set('{%s}name' % ns_map['data-type'], fluoro)

    detectors_ml = etree.SubElement(xform_ml, "{%s}detectors" % ns_map['transforms'])

    for detector in matrix.detectors:
        detector_ml = etree.SubElement(detectors_ml, '{%s}fcs-dimension' % ns_map['data-type'])
        detector_ml.set('{%s}name' % ns_map['data-type'], detector)

    for row in matrix.matrix:
        row_ml = etree.SubElement(xform_ml, "{%s}spectrum" % ns_map['transforms'])
        for val in row:
            coeff_ml = etree.SubElement(row_ml, "{%s}coefficient" % ns_map['transforms'])
            coeff_ml.set('{%s}value' % ns_map['transforms'], str(val))


def _add_transform_to_gml(root, transform, ns_map):
    xform_ml = etree.SubElement(root, "{%s}transformation" % ns_map['transforms'])
    xform_ml.set('{%s}id' % ns_map['transforms'], transform.id)

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
        hlog_ml = etree.SubElement(xform_ml, "{%s}hyperlog" % ns_map['transforms'])
        hlog_ml.set('{%s}T' % ns_map['transforms'], str(transform.param_t))
        hlog_ml.set('{%s}W' % ns_map['transforms'], str(transform.param_w))
        hlog_ml.set('{%s}M' % ns_map['transforms'], str(transform.param_m))
        hlog_ml.set('{%s}A' % ns_map['transforms'], str(transform.param_a))
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
            for c in v.coordinates:
                coord_ml = etree.SubElement(vert_ml, '{%s}coordinate' % ns_map['gating'])
                coord_ml.set('{%s}value' % ns_map['data-type'], str(c))
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
        raise ValueError("Gate %s is not a valid GatingML 2.0 element" % gate.id)

    gate_ml.set('{%s}id' % ns_map['gating'], gate.id)

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
                fcs_dim_ml.set('{%s}name' % ns_map['data-type'], dim.label)
            elif dim_type == 'quad':
                fcs_dim_ml.set('{%s}name' % ns_map['data-type'], dim.dimension_ref)
                for val in dim.values:
                    value_ml = etree.SubElement(dim_ml, '{%s}value' % ns_map['gating'])
                    value_ml.text = str(val)

    return gate_ml


def _add_gates_from_gate_dict(gating_strategy, gate_dict, ns_map, parent_ml):
    # the gate_dict will have keys 'name' and 'children'. top-level 'name' value is 'root'
    for child in gate_dict['children']:
        gate_id = child['name']
        skip = False

        gate = gating_strategy.get_gate(gate_id)
        if isinstance(gate, Quadrant):
            # single quadrants will be handled in the owning quadrant gate
            skip = True

        if not skip:
            child_ml = _add_gate_to_gml(parent_ml, gate, ns_map)

            if gate_dict['name'] != 'root':
                # this is a recursion, add the parent reference
                child_ml.set('{%s}parent_id' % ns_map['gating'], gate_dict['name'])

        if 'children' in child:  # and not isinstance(gate, QuadrantGate):
            _add_gates_from_gate_dict(gating_strategy, child, ns_map, parent_ml)


def export_gatingml(gating_strategy, file_handle):
    """
    Exports a valid GatingML 2.0 document from given GatingStrategy instance
    :param gating_strategy: A GatingStrategy instance
    :param file_handle: File handle for exported GatingML 2.0 document
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
        _add_transform_to_gml(root, xform, ns_map)

    # process gating strategy compensation matrices
    for matrix_id, matrix in gating_strategy.comp_matrices.items():
        _add_matrix_to_gml(root, matrix, ns_map)

    # get gate hierarchy as a dictionary
    gate_dict = gating_strategy.get_gate_hierarchy('dict')

    # recursively convert all gates to GatingML
    _add_gates_from_gate_dict(gating_strategy, gate_dict, ns_map, root)

    et = etree.ElementTree(root)

    et.write(file_handle, encoding="utf-8", xml_declaration=True, pretty_print=True)
