"""
Utility functions for parsing GatingML 2.0 documents
"""
import numpy as np
import networkx as nx
from .xml_common import _get_xml_type, find_attribute_value
from .._models.dimension import Dimension, RatioDimension, QuadrantDivider
from .._models.gating_strategy import GatingStrategy
from .._models import transforms
# noinspection PyProtectedMember
from .._models.transforms._matrix import Matrix
# noinspection PyProtectedMember
from .._models.gates._gml_gates import \
    GMLBooleanGate, \
    GMLEllipsoidGate, \
    GMLQuadrantGate, \
    GMLPolygonGate, \
    GMLRectangleGate
# noinspection PyProtectedMember
from .._models.gates._gates import BooleanGate, QuadrantGate

# map GatingML gate keys to our GML gate classes
gate_constructor_lut = {
    'RectangleGate': GMLRectangleGate,
    'PolygonGate': GMLPolygonGate,
    'EllipsoidGate': GMLEllipsoidGate,
    'QuadrantGate': GMLQuadrantGate,
    'BooleanGate': GMLBooleanGate
}


def parse_gating_xml(xml_file_or_path):
    """
    Parse a GatingML 20 document and return as a GatingStrategy.

    :param xml_file_or_path: file handle or file path to a GatingML 2.0 document
    :return: GatingStrategy instance
    """
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
        gating_strategy.add_comp_matrix(c_id, c)
    for t_id, t in transformations.items():
        gating_strategy.add_transform(t_id, t)

    deps = []
    quadrants = []
    bool_edges = []

    for g_id, gate in gates.items():
        # GML gates have a parent reference & their gate names are
        # required to be unique, so we can use them to assemble the tree
        if gate.parent is None:
            parent = 'root'
        else:
            parent = gate.parent

        deps.append((parent, g_id))

        if isinstance(gate, QuadrantGate):
            for q_id in gate.quadrants:
                deps.append((g_id, q_id))
                quadrants.append(q_id)

        if isinstance(gate, BooleanGate):
            for g_ref in gate.gate_refs:
                deps.append((g_ref['ref'], g_id))

                bool_edges.append((g_ref['ref'], g_id))

    dag = nx.DiGraph(deps)

    is_acyclic = nx.is_directed_acyclic_graph(dag)

    if not is_acyclic:
        raise ValueError("The given GatingML 2.0 file is invalid, cyclic gate dependencies are not allowed.")

    process_order = list(nx.algorithms.topological_sort(dag))

    for q_id in quadrants:
        process_order.remove(q_id)

    # remove boolean edges to create a true ancestor graph
    dag.remove_edges_from(bool_edges)

    for g_id in process_order:
        # skip 'root' node
        if g_id == 'root':
            continue
        gate = gates[g_id]

        # For Boolean gates we need to add gate paths to the
        # referenced gates via 'gate_path' key in the gate_refs dict
        if isinstance(gate, BooleanGate):
            bool_gate_refs = gate.gate_refs
            for gate_ref in bool_gate_refs:
                # since we're parsing GML, all gate IDs must be unique
                # so safe to lookup in our graph
                gate_ref_path = list(nx.all_simple_paths(dag, 'root', gate_ref['ref']))[0]
                gate_ref['path'] = tuple(gate_ref_path[:-1])  # don't repeat the gate name

        # need to get the gate path
        # again, since GML gate IDs must be unique, safe to lookup from graph
        gate_path = tuple(nx.shortest_path(dag, 'root', g_id))[:-1]

        # Convert GML gates to their superclass & add to gating strategy
        gating_strategy.add_gate(gate.convert_to_parent_class(), gate_path)

    return gating_strategy


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

            if g.gate_name in gates_dict:
                raise ValueError(
                    "Gate '%s' already exists. "
                    "Duplicate gate names are not allowed." % g.gate_name
                )
            gates_dict[g.gate_name] = g

    return gates_dict


def _construct_transforms(root_gml, transform_ns, data_type_ns):
    transformations = {}

    if transform_ns is not None:
        # GML uses a 'transformation' wrapper tag before we can tell what kind of xform it is
        xform_els = root_gml.findall(
            '%s:transformation' % transform_ns,
            namespaces=root_gml.nsmap
        )

        for xform_el in xform_els:
            xform_id, xform = _parse_transformation_element(xform_el, transform_ns, data_type_ns)

            if xform is not None:
                transformations[xform_id] = xform

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
            matrix_id, matrix = _parse_matrix_element(
                matrix_el,
                transform_ns,
                data_type_ns
            )

            comp_matrices[matrix_id] = matrix

    return comp_matrices


def parse_gate_element(
        gate_element,
        gating_namespace,
        data_type_namespace
):
    """
    This class parses a GatingML-2.0 compatible gate XML element and extracts the gate ID,
     parent gate ID, and dimensions.

    :param gate_element: gate XML element from a GatingML-2.0 document
    :param gating_namespace: XML namespace for gating elements/attributes
    :param data_type_namespace: XML namespace for data type elements/attributes
    """
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
            dim = _parse_dimension_element(dim_el, gating_namespace, data_type_namespace)
            dimensions.append(dim)
    else:
        for div_el in div_els:
            dim = _parse_divider_element(div_el, gating_namespace, data_type_namespace)
            dimensions.append(dim)

    return gate_id, parent_id, dimensions


def _parse_dimension_element(
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

    # ID be here
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
                "New dimensions must provide a transform reference (line %d)" % dim_element.sourceline
            )
        dimension = RatioDimension(
            ratio_xform_ref,
            compensation_ref,
            transformation_ref,
            range_min=range_min,
            range_max=range_max
        )
    else:
        dim_id = find_attribute_value(fcs_dim_el, data_type_namespace, 'name')
        if dim_id is None:
            raise ValueError(
                'Dimension name not found (line %d)' % fcs_dim_el.sourceline
            )

        dimension = Dimension(
            dim_id,
            compensation_ref,
            transformation_ref,
            range_min=range_min,
            range_max=range_max
        )

    return dimension


def _parse_divider_element(divider_element, gating_namespace, data_type_namespace):
    # Get 'id' (present in quad gate dividers)
    divider_id = find_attribute_value(divider_element, gating_namespace, 'id')

    compensation_ref = find_attribute_value(divider_element, gating_namespace, 'compensation-ref')
    transformation_ref = find_attribute_value(divider_element, gating_namespace, 'transformation-ref')

    # ID be here
    fcs_dim_el = divider_element.find(
        '%s:fcs-dimension' % data_type_namespace,
        namespaces=divider_element.nsmap
    )

    dim_id = find_attribute_value(fcs_dim_el, data_type_namespace, 'name')
    if dim_id is None:
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

    divider = QuadrantDivider(divider_id, dim_id, compensation_ref, values, transformation_ref)

    return divider


def parse_vertex_element(vertex_element, gating_namespace, data_type_namespace):
    """
    This class parses a GatingML-2.0 compatible vertex XML element and returns a list of coordinates.

    :param vertex_element: vertex XML element from a GatingML-2.0 document
    :param gating_namespace: XML namespace for gating elements/attributes
    :param data_type_namespace: XML namespace for data type elements/attributes
    """
    coordinates = []

    coord_els = vertex_element.findall(
        '%s:coordinate' % gating_namespace,
        namespaces=vertex_element.nsmap
    )

    if len(coord_els) != 2:
        raise ValueError(
            'Vertex must contain 2 coordinate values (line %d)' % vertex_element.sourceline
        )

    # should be 0 or only 1 'min' attribute,
    for coord_el in coord_els:
        value = find_attribute_value(coord_el, data_type_namespace, 'value')
        if value is None:
            raise ValueError(
                'Vertex coordinate must have only 1 value (line %d)' % coord_el.sourceline
            )

        coordinates.append(float(value))

    return coordinates


def _parse_matrix_element(
    matrix_element,
    xform_namespace,
    data_type_namespace
):
    matrix_id = find_attribute_value(matrix_element, xform_namespace, 'id')
    fluorochromes = []
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
        fluorochromes.append(label)

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

    return matrix_id, Matrix(matrix, detectors, fluorochromes)


def _parse_fratio_element(fratio_element, transform_namespace, data_type_namespace):
    # f ratio transform has 3 parameters: A, B, and C
    # these are attributes of the 'fratio' element
    param_a = find_attribute_value(fratio_element, transform_namespace, 'A')
    param_b = find_attribute_value(fratio_element, transform_namespace, 'B')
    param_c = find_attribute_value(fratio_element, transform_namespace, 'C')

    if None in [param_a, param_b, param_c]:
        raise ValueError(
            "Ratio transform must provide an 'A', a 'B', and a 'C' "
            "attribute (line %d)" % fratio_element.sourceline
        )

    # convert from string to float
    param_a = float(param_a)
    param_b = float(param_b)
    param_c = float(param_c)

    fcs_dim_els = fratio_element.findall(
        '%s:fcs-dimension' % data_type_namespace,
        namespaces=fratio_element.nsmap
    )

    dim_ids = []

    for dim_el in fcs_dim_els:
        dim_id = find_attribute_value(dim_el, data_type_namespace, 'name')

        if dim_id is None:
            raise ValueError(
                'Dimension name not found (line %d)' % dim_el.sourceline
            )
        dim_ids.append(dim_id)

    xform = transforms.RatioTransform(dim_ids, param_a, param_b, param_c)

    return xform


def _parse_flog_element(flog_element, transform_namespace):
    # f log transform has 2 parameters: T and M
    # these are attributes of the 'flog' element
    param_t = find_attribute_value(flog_element, transform_namespace, 'T')
    param_m = find_attribute_value(flog_element, transform_namespace, 'M')

    if None in [param_t, param_m]:
        raise ValueError(
            "Log transform must provide an 'T' attribute (line %d)" % flog_element.sourceline
        )

    # convert string to float
    param_t = float(param_t)
    param_m = float(param_m)

    xform = transforms.LogTransform(param_t, param_m)

    return xform


def _parse_fasinh_element(fasinh_element, transform_namespace):
    # f asinh transform has 3 parameters: T, M, and A
    # these are attributes of the 'fasinh' element
    param_t = find_attribute_value(fasinh_element, transform_namespace, 'T')
    param_m = find_attribute_value(fasinh_element, transform_namespace, 'M')
    param_a = find_attribute_value(fasinh_element, transform_namespace, 'A')

    if None in [param_t, param_m, param_a]:
        raise ValueError(
            "Asinh transform must provide 'T', 'M', and 'A' attributes (line %d)" % fasinh_element.sourceline
        )

    # convert string to float
    param_t = float(param_t)
    param_m = float(param_m)
    param_a = float(param_a)

    xform = transforms.AsinhTransform(param_t, param_m, param_a)

    return xform


def _parse_hyperlog_element(hyperlog_element, transform_namespace):
    # hyperlog transform has 4 parameters: T, W, M, and A
    # these are attributes of the 'hyperlog' element
    param_t = find_attribute_value(hyperlog_element, transform_namespace, 'T')
    param_w = find_attribute_value(hyperlog_element, transform_namespace, 'W')
    param_m = find_attribute_value(hyperlog_element, transform_namespace, 'M')
    param_a = find_attribute_value(hyperlog_element, transform_namespace, 'A')

    if None in [param_t, param_w, param_m, param_a]:
        raise ValueError(
            "Hyperlog transform must provide 'T', 'W', 'M', and 'A' "
            "attributes (line %d)" % hyperlog_element.sourceline
        )

    # convert string to float
    param_t = float(param_t)
    param_w = float(param_w)
    param_m = float(param_m)
    param_a = float(param_a)

    xform = transforms.HyperlogTransform(param_t, param_w, param_m, param_a)

    return xform


def _parse_flin_element(flin_element, transform_namespace):
    # f linear transform has 2 parameters: T and A
    # these are attributes of the 'flin' element
    param_t = find_attribute_value(flin_element, transform_namespace, 'T')
    param_a = find_attribute_value(flin_element, transform_namespace, 'A')

    if None in [param_t, param_a]:
        raise ValueError(
            "Linear transform must provide 'T' and 'A' attributes (line %d)" % flin_element.sourceline
        )

    # convert string to float
    param_t = float(param_t)
    param_a = float(param_a)

    xform = transforms.LinearTransform(param_t, param_a)

    return xform


def _parse_logicle_element(logicle_element, transform_namespace):
    # logicle transform has 4 parameters: T, W, M, and A
    # these are attributes of the 'logicle' element
    param_t = find_attribute_value(logicle_element, transform_namespace, 'T')
    param_w = find_attribute_value(logicle_element, transform_namespace, 'W')
    param_m = find_attribute_value(logicle_element, transform_namespace, 'M')
    param_a = find_attribute_value(logicle_element, transform_namespace, 'A')

    if None in [param_t, param_w, param_m, param_a]:
        raise ValueError(
            "Logicle transform must provide 'T', 'W', 'M', and 'A' "
            "attributes (line %d)" % logicle_element.sourceline
        )

    # convert string to float
    param_t = float(param_t)
    param_w = float(param_w)
    param_m = float(param_m)
    param_a = float(param_a)

    xform = transforms.LogicleTransform(param_t, param_w, param_m, param_a)

    return xform


def _parse_transformation_element(transformation_element, transform_namespace, data_type_namespace):
    xform = None

    # 'transformation' wrapper tag has the transform ID, so grab that & then parse
    # the child tags. There should only be 1 child that will reveal the xform type.
    xform_id = find_attribute_value(transformation_element, transform_namespace, 'id')
    xform_child_els = transformation_element.getchildren()

    # there should only ever be 1 child tag, but we'll loop
    for xform_child_el in xform_child_els:
        # Transform type tags include:
        #   - fratio
        #   - flog
        #   - fasinh
        #   - hyperlog
        #   - flin
        #   - logicle
        xform_type = xform_child_el.tag.partition('}')[-1]

        if xform_type == 'fratio':
            xform = _parse_fratio_element(
                xform_child_el, transform_namespace, data_type_namespace
            )
        elif xform_type == 'flog':
            xform = _parse_flog_element(xform_child_el, transform_namespace)
        elif xform_type == 'fasinh':
            xform = _parse_fasinh_element(xform_child_el, transform_namespace)
        elif xform_type == 'hyperlog':
            xform = _parse_hyperlog_element(xform_child_el, transform_namespace)
        elif xform_type == 'flin':
            xform = _parse_flin_element(xform_child_el, transform_namespace)
        elif xform_type == 'logicle':
            xform = _parse_logicle_element(xform_child_el, transform_namespace)

    return xform_id, xform
