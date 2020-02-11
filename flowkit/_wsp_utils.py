import numpy as np
from flowkit import _gml_utils
from ._models.transforms import transforms
from ._models.gates.gml_gates import \
    GMLPolygonGate, \
    GMLRectangleGate

# map GatingML gate keys to our GML gate classes
gate_constructor_lut = {
    'RectangleGate': GMLRectangleGate,
    'PolygonGate': GMLPolygonGate,
    # 'EllipsoidGate': WSPEllipsoidGate,
    # 'QuadrantGate': WSPQuadrantGate,
    # 'BooleanGate': WSPBooleanGate
}


def parse_wsp(gating_strategy, root_xml):
    # first, find Groups & SampleList elements
    ns_map = root_xml.nsmap
    groups_el = root_xml.find('Groups', ns_map)
    sample_list_el = root_xml.find('SampleList', ns_map)

    # next, find which samples belong to which groups
    group_node_els = groups_el.findall('GroupNode', ns_map)
    sample_els = sample_list_el.findall('Sample', ns_map)

    # Note, sample IDs are strings
    group_sample_lut = {}

    for group_node_el in group_node_els:
        group_el = group_node_el.find('Group', ns_map)
        group_sample_refs_el = group_el.find('SampleRefs', ns_map)

        if group_sample_refs_el is None:
            # TODO: handle 'Compensation' group if necessary
            continue
        group_sample_els = group_sample_refs_el.findall('SampleRef', ns_map)

        sample_ids = []

        for sample_el in group_sample_els:
            sample_ids.append(sample_el.attrib['sampleID'])

        group_sample_lut[group_node_el.attrib['name']] = sample_ids

    samples = {}

    for sample_el in sample_els:
        xforms_el = sample_el.find('Transformations', ns_map)

        sample_node_el = sample_el.find('SampleNode', ns_map)

        sample_id = sample_node_el.attrib['sampleID']
        sample_name = sample_node_el.attrib['name']

        # It appears there is only a single set of xforms per sample, one for each channel.
        # And, the xforms have no IDs. We'll extract it and give it IDs based on ???
        sample_xform_lut = parse_wsp_transforms(xforms_el, gating_strategy.transform_ns, gating_strategy.data_type_ns)

        # parse spilloverMatrix elements
        sample_comp = parse_wsp_compensation(sample_el, gating_strategy.transform_ns, gating_strategy.data_type_ns)

        # FlowJo WSP gates are nested so we'll have to do a recursive search from the root Sub-populations node
        sample_root_sub_pop_el = sample_node_el.find('Subpopulations', ns_map)
        # TODO: FJ WSP gates are stored in non-transformed space. After parsing the XML the values need
        #       to be converted to the compensated & transformed space
        sample_gates = recurse_sub_populations(sample_root_sub_pop_el, gating_strategy, ns_map)

        samples[sample_id] = {
            'name': sample_name,
            'gates': sample_gates,
            'transform_lut': sample_xform_lut,
            'compensation': sample_comp
        }

    gates_dict = {}

    return gates_dict


def recurse_sub_populations(sub_pop_el, gating_strategy, ns_map):
    gates = []

    pop_els = sub_pop_el.findall('Population', ns_map)
    for pop_el in pop_els:
        pop_name = pop_el.attrib['name']
        owning_group = pop_el.attrib['owningGroup']
        gate_el = pop_el.find('Gate', ns_map)

        gate_child_els = gate_el.getchildren()

        if len(gate_child_els) != 1:
            raise ValueError("Gate element must have only 1 child element")

        gate_child_el = gate_child_els[0]

        # determine gate type
        # TODO: this string parsing seems fragile, may need to be shored up
        gate_type = gate_child_el.tag.partition('}')[-1]
        gate_class = gate_constructor_lut[gate_type]

        g = gate_class(
            gate_child_el,
            gating_strategy.gating_ns,
            gating_strategy.data_type_ns
        )

        gates.append(
            {
                'node': pop_name,
                'owning_group': owning_group,
                'gate': g
            }
        )

        sub_pop_els = pop_el.findall('Subpopulations', ns_map)
        for el in sub_pop_els:
            gates.extend(recurse_sub_populations(el, gating_strategy, ns_map))

    return gates


def parse_wsp_transforms(transforms_el, transform_ns, data_type_ns):
    # get all children and then determine the tag based on the xform type (linear, fasinh, etc.)
    xform_els = transforms_el.getchildren()

    # there should be one transform per channel, use the channel names to create a LUT
    xforms_lut = {}

    for xform_el in xform_els:
        xform_type = xform_el.tag.partition('}')[-1]

        param_el = xform_el.find(
            '%s:parameter' % data_type_ns,
            namespaces=xform_el.nsmap
        )
        param_name = _gml_utils.find_attribute_value(param_el, data_type_ns, 'name')

        # FlowKit only supports linear, log, and logicle transformations in FlowJo WSP files.
        # All other bi-ex transforms implemented by FlowJo are undocumented and not reproducible
        if xform_type == 'linear':
            min_range = _gml_utils.find_attribute_value(xform_el, transform_ns, 'minRange')
            max_range = _gml_utils.find_attribute_value(xform_el, transform_ns, 'maxRange')
            xforms_lut[param_name] = transforms.LinearTransform(
                param_name,
                param_t=float(max_range),
                param_a=float(min_range)
            )
        elif xform_type == 'log':
            # TODO: implement log transform
            pass
        elif xform_type == 'logicle':
            # logicle transform has 4 parameters: T, W, M, and A
            # these are attributes of the 'logicle' element
            param_t = _gml_utils.find_attribute_value(xform_el, transform_ns, 'T')
            param_w = _gml_utils.find_attribute_value(xform_el, transform_ns, 'W')
            param_m = _gml_utils.find_attribute_value(xform_el, transform_ns, 'M')
            param_a = _gml_utils.find_attribute_value(xform_el, transform_ns, 'A')
            xforms_lut[param_name] = transforms.LogicleTransform(
                param_name,
                param_t=float(param_t),
                param_w=float(param_w),
                param_m=float(param_m),
                param_a=float(param_a)
            )
        else:
            error_msg = "FlowJo transform type %s is undocumented and not supported in FlowKit. " % xform_type
            error_msg += "Please edit the workspace in FlowJo and save all channel transformations as either " \
                "linear, log, or logicle"

            raise ValueError(error_msg)

    return xforms_lut


def parse_wsp_compensation(sample_el, transform_ns, data_type_ns):
    # find spilloverMatrix elements, not sure if there should be just a single matrix or multiple
    # going with a single one now since there do not appear to be comp references in the WSP gate elements
    matrix_els = sample_el.findall(
        '%s:spilloverMatrix' % transform_ns,
        namespaces=sample_el.nsmap
    )

    if len(matrix_els) > 1:
        raise ValueError("Multiple spillover matrices per sample are not supported.")
    elif len(matrix_els) == 0:
        return None

    matrix_el = matrix_els[0]

    matrix_id = _gml_utils.find_attribute_value(matrix_el, transform_ns, 'id')
    matrix_prefix = matrix_el.attrib['prefix']
    matrix_suffix = matrix_el.attrib['suffix']

    detectors = []
    matrix = []

    params_els = matrix_el.find('%s:parameters' % data_type_ns, namespaces=matrix_el.nsmap)
    param_els = params_els.findall('%s:parameter' % data_type_ns, namespaces=matrix_el.nsmap)
    for param_el in param_els:
        param_name = _gml_utils.find_attribute_value(param_el, data_type_ns, 'name')
        detectors.append(param_name)

    spill_els = matrix_el.findall(
        '%s:spillover' % transform_ns,
        namespaces=matrix_el.nsmap
    )

    for spill_el in spill_els:
        matrix_row = []

        coefficient_els = spill_el.findall(
            '%s:coefficient' % transform_ns,
            namespaces=spill_el.nsmap
        )

        for co_el in coefficient_els:
            value = _gml_utils.find_attribute_value(co_el, transform_ns, 'value')
            if value is None:
                raise ValueError(
                    'Matrix coefficient must have only 1 value (line %d)' % co_el.sourceline
                )

            matrix_row.append(float(value))

        matrix.append(matrix_row)

    matrix = np.array(matrix)

    matrix_dict = {
        'matrix_id': matrix_id,
        'prefix': matrix_prefix,
        'suffix': matrix_suffix,
        'detectors': detectors,
        'matrix': matrix
    }

    return matrix_dict
