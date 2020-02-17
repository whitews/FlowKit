import copy
import numpy as np
from flowkit import _gml_utils, Matrix
from ._models.dimension import Dimension
from ._models.transforms import transforms
from ._models.gates.gates import \
    PolygonGate, \
    RectangleGate
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


def parse_wsp(root_xml, gating_ns, transform_ns, data_type_ns):
    # first, find SampleList elements
    ns_map = root_xml.nsmap
    sample_list_el = root_xml.find('SampleList', ns_map)
    sample_els = sample_list_el.findall('Sample', ns_map)

    wsp_dict = {}

    for sample_el in sample_els:
        transforms_el = sample_el.find('Transformations', ns_map)

        sample_node_el = sample_el.find('SampleNode', ns_map)

        sample_id = sample_node_el.attrib['sampleID']
        sample_name = sample_node_el.attrib['name']

        # It appears there is only a single set of xforms per sample, one for each channel.
        # And, the xforms have no IDs. We'll extract it and give it IDs based on ???
        sample_xform_lut = parse_wsp_transforms(transforms_el, transform_ns, data_type_ns)

        # parse spilloverMatrix elements
        sample_comp = parse_wsp_compensation(sample_el, transform_ns, data_type_ns)

        # FlowJo WSP gates are nested so we'll have to do a recursive search from the root Sub-populations node
        sample_root_sub_pop_el = sample_node_el.find('Subpopulations', ns_map)

        # FJ WSP gates are stored in non-transformed space. After parsing the XML the values need
        # to be converted to the compensated & transformed space. Also, the recurse_sub_populations
        # function replaces the non-human readable IDs in the XML with population names
        sample_gates = recurse_sub_populations(
            sample_root_sub_pop_el,
            None,  # starting at root, so no parent ID
            gating_ns,
            data_type_ns
        )

        for sample_gate in sample_gates:
            group = sample_gate['owning_group']
            gate = sample_gate['gate']

            if group not in wsp_dict:
                wsp_dict[group] = {}
            if sample_name not in wsp_dict[group]:
                detectors = sample_comp['detectors']
                matrix = Matrix(
                    sample_comp['matrix_name'],
                    fluorochromes=['' for d in detectors],
                    detectors=detectors,
                    matrix=sample_comp['matrix']
                )

                wsp_dict[group][sample_name] = {
                    'gates': [],
                    'transform_lut': sample_xform_lut,
                    'compensation': matrix
                }

            gate = convert_wsp_gate(gate, sample_comp, sample_xform_lut)
            wsp_dict[group][sample_name]['gates'].append(gate)

    return wsp_dict


def recurse_sub_populations(sub_pop_el, parent_id, gating_ns, data_type_ns):
    gates = []
    ns_map = sub_pop_el.nsmap

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
            gating_ns,
            data_type_ns
        )

        # replace ID and parent ID with population names
        g.id = pop_name
        g.parent = parent_id

        gates.append(
            {
                'owning_group': owning_group,
                'gate': g
            }
        )

        sub_pop_els = pop_el.findall('Subpopulations', ns_map)
        for el in sub_pop_els:
            gates.extend(recurse_sub_populations(el, pop_name, gating_ns, data_type_ns))

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

    # we'll ignore the non-human readable matrix ID and use the name instead
    matrix_name = matrix_el.attrib['name']
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
        'matrix_name': matrix_name,
        'prefix': matrix_prefix,
        'suffix': matrix_suffix,
        'detectors': detectors,
        'matrix': matrix
    }

    return matrix_dict


def convert_wsp_gate(wsp_gate, comp_matrix, xform_lut):
    new_dims = []
    xforms = []

    for dim in wsp_gate.dimensions:
        dim_label = dim.label.lstrip(comp_matrix['prefix'])
        dim_label = dim_label.rstrip(comp_matrix['suffix'])

        comp_ref = None
        xform_id = None
        new_dim_min = None
        new_dim_max = None

        if dim_label in comp_matrix['detectors']:
            comp_ref = comp_matrix['matrix_name']

        if dim_label in xform_lut:
            xform = xform_lut[dim_label]
            xforms.append(xform)  # need these later for vertices, coordinates, etc.
            xform_id = xform.id
            if dim.min is not None:
                new_dim_min = xform.apply(np.array([[float(dim.min)]]))

            if dim.max is not None:
                new_dim_max = xform.apply(np.array([[float(dim.max)]]))
        else:
            xforms.append(None)

        new_dim = Dimension(dim_label, comp_ref, xform_id, range_min=new_dim_min, range_max=new_dim_max)
        new_dims.append(new_dim)

    if isinstance(wsp_gate, GMLPolygonGate):
        # convert vertices using saved xforms
        vertices = copy.deepcopy(wsp_gate.vertices)
        for v in vertices:
            for i, c in enumerate(v.coordinates):
                if xforms[i] is not None:
                    v.coordinates[i] = xforms[i].apply(np.array([[float(c)]]))

        # TODO: support more than just PolygonGate
        gate = PolygonGate(wsp_gate.id, wsp_gate.parent, new_dims, vertices)
    else:
        raise NotImplemented("Only polygon gates for FlowJo workspaces are currently supported.")

    return gate
