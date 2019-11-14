import numpy as np
import os
from lxml import etree
from ._resources import gml_schema
from ._models.dimension import Dimension, RatioDimension, QuadrantDivider
from ._models.vertex import Vertex
from flowkit import Matrix
from ._models.transforms import gml_transforms
from ._models.gates.gml_gates import \
    GMLBooleanGate, \
    GMLEllipsoidGate, \
    GMLQuadrantGate, \
    GMLPolygonGate, \
    GMLRectangleGate
from ._models.gates.gates import \
    BooleanGate, \
    EllipsoidGate, \
    QuadrantGate, \
    PolygonGate, \
    RectangleGate


def parse_gatingml_file(gating_ml_file_path):
    xml_document = etree.parse(gating_ml_file_path)

    val = gml_schema.validate(xml_document)

    if not val:
        raise ValueError("Document is not valid GatingML")

    root = xml_document.getroot()

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

    return root, gating_ns, data_type_ns, transform_ns


def construct_gates(gating_strategy, root_gml):
    # map GatingML gate keys to our GML gate classes
    gate_constructor_lut = {
        'RectangleGate': GMLRectangleGate,
        'PolygonGate': GMLPolygonGate,
        'EllipsoidGate': GMLEllipsoidGate,
        'QuadrantGate': GMLQuadrantGate,
        'BooleanGate': GMLBooleanGate
    }

    gates_dict = {}

    for gate_str, gate_class in gate_constructor_lut.items():
        gt_gates = root_gml.findall(
            ':'.join([gating_strategy.gating_ns, gate_str]),
            root_gml.nsmap
        )

        for gt_gate in gt_gates:
            g = gate_class(
                gt_gate,
                gating_strategy.gating_ns,
                gating_strategy.data_type_ns
            )

            if g.id in gates_dict:
                raise ValueError(
                    "Gate '%s' already exists. "
                    "Duplicate gate IDs are not allowed." % g.id
                )
            gates_dict[g.id] = g

    return gates_dict


def construct_transforms(root_gml, transform_ns, data_type_ns):
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
                xform = gml_transforms.RatioGMLTransform(
                    xform_el,
                    transform_ns,
                    data_type_ns
                )

            flog_els = xform_el.findall(
                '%s:flog' % transform_ns,
                namespaces=root_gml.nsmap
            )

            if len(flog_els) > 0:
                xform = gml_transforms.LogGMLTransform(
                    xform_el,
                    transform_ns
                )

            fasinh_els = xform_el.findall(
                '%s:fasinh' % transform_ns,
                namespaces=root_gml.nsmap
            )

            if len(fasinh_els) > 0:
                xform = gml_transforms.AsinhGMLTransform(
                    xform_el,
                    transform_ns
                )

            hyperlog_els = xform_el.findall(
                '%s:hyperlog' % transform_ns,
                namespaces=root_gml.nsmap
            )

            if len(hyperlog_els) > 0:
                xform = gml_transforms.HyperlogGMLTransform(
                    xform_el,
                    transform_ns
                )

            flin_els = xform_el.findall(
                '%s:flin' % transform_ns,
                namespaces=root_gml.nsmap
            )

            if len(flin_els) > 0:
                xform = gml_transforms.LinearGMLTransform(
                    xform_el,
                    transform_ns
                )

            logicle_els = xform_el.findall(
                '%s:logicle' % transform_ns,
                namespaces=root_gml.nsmap
            )

            if len(logicle_els) > 0:
                xform = gml_transforms.LogicleGMLTransform(
                    xform_el,
                    transform_ns
                )

            if xform is not None:
                transformations[xform.id] = xform

    return transformations


def construct_matrices(root_gml, transform_ns, data_type_ns):
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

    return Matrix(matrix_id, fluorochomes, detectors, matrix)


def add_gate_to_gml(root, gate, ns_map):
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
    elif isinstance(gate, EllipsoidGate):
        gate_ml = etree.SubElement(root, "{%s}EllipsoidGate" % ns_map['gating'])
    elif isinstance(gate, QuadrantGate):
        gate_ml = etree.SubElement(root, "{%s}QuadrantGate" % ns_map['gating'])
    else:
        raise(ValueError, "gate is not a valid GatingML 2.0 element")

    gate_ml.set('{%s}id' % ns_map['gating'], gate.id)

    for i, dim in enumerate(gate.dimensions):
        dim_ml = etree.Element('{%s}dimension' % ns_map['gating'])
        gate_ml.insert(i, dim_ml)
        if dim.compensation_ref is not None:
            dim_ml.set('{%s}compensation-ref' % ns_map['gating'], dim.compensation_ref)
        if dim.transformation_ref is not None:
            dim_ml.set('{%s}transformation-ref' % ns_map['gating'], dim.transformation_ref)
        if dim.min is not None:
            dim_ml.set('{%s}min' % ns_map['gating'], str(dim.min))
        if dim.max is not None:
            dim_ml.set('{%s}max' % ns_map['gating'], str(dim.max))

        fcs_dim_ml = etree.SubElement(dim_ml, '{%s}fcs-dimension' % ns_map['data-type'])
        fcs_dim_ml.set('{%s}name' % ns_map['data-type'], dim.label)

    return gate_ml


def export_gatingml(gating_strategy, file_name, base_dir=None):
    """
    Exports a valid GatingML 2.0 document from given GatingStrategy instance
    :param gating_strategy: A GatingStrategy instance
    :param file_name: File name for exported GatingML 2.0 document
    :param base_dir: Optional parent directory for exported document. Default is current directory.
    :return: None
    """
    ns_g = "http://www.isac-net.org/std/Gating-ML/v2.0/gating"
    ns_dt = "http://www.isac-net.org/std/Gating-ML/v2.0/datatypes"
    ns_map = {
        'gating': ns_g,
        'data-type': ns_dt
    }

    root = etree.Element('{%s}Gating-ML' % ns_g, nsmap=ns_map)

    # get gate hierarchy as a dictionary
    gate_dict = gating_strategy.get_gate_hierarchy('dict')

    # the gate_dict will have keys 'name' and 'children'. 'name' value is 'root'
    for child in gate_dict['children']:
        # all the top-level children have no parents, so we'll process them first
        gate_id = child['name']
        gate = gating_strategy.gates[gate_id]
        add_gate_to_gml(root, gate, ns_map)

    et = etree.ElementTree(root)

    if base_dir is not None:
        file_path = os.path.join(base_dir, file_name)
    else:
        file_path = file_name

    with open(file_path, 'wb') as f:
        et.write(f, encoding="utf-8", xml_declaration=True, pretty_print=True)
