"""
Utility functions related to FlowJo Workspace files (.wsp)
"""
import copy
import re
import numpy as np
from .xml_utils import find_attribute_value, _get_xml_type, _build_hierarchy_tree
from .._models.dimension import Dimension
from .._models.transforms._matrix import Matrix
from .._models.transforms import _transforms, _wsp_transforms
from .._models.gates._gates import PolygonGate
from .._models.gates._gml_gates import \
    GMLBooleanGate, \
    GMLQuadrantGate, \
    GMLPolygonGate, \
    GMLRectangleGate
from .._models.gates._wsp_gates import WSPEllipsoidGate

wsp_gate_constructor_lut = {
    'RectangleGate': GMLRectangleGate,
    'PolygonGate': GMLPolygonGate,
    'EllipsoidGate': WSPEllipsoidGate,
    'QuadrantGate': GMLQuadrantGate,
    'BooleanGate': GMLBooleanGate
}


def _parse_wsp_compensation(sample_el, transform_ns, data_type_ns):
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
        param_name = find_attribute_value(param_el, data_type_ns, 'name')
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
            value = find_attribute_value(co_el, transform_ns, 'value')
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


def _parse_wsp_transforms(transforms_el, transform_ns, data_type_ns):
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
        param_name = find_attribute_value(param_el, data_type_ns, 'name')

        # FlowKit only supports linear, log, and logicle transformations in FlowJo WSP files.
        # All other bi-ex transforms implemented by FlowJo are undocumented and not reproducible
        if xform_type == 'linear':
            min_range = find_attribute_value(xform_el, transform_ns, 'minRange')
            max_range = find_attribute_value(xform_el, transform_ns, 'maxRange')
            xforms_lut[param_name] = _transforms.LinearTransform(
                param_name,
                param_t=float(max_range),
                param_a=float(min_range)
            )
        elif xform_type == 'log':
            offset = find_attribute_value(xform_el, transform_ns, 'offset')
            decades = find_attribute_value(xform_el, transform_ns, 'decades')
            xforms_lut[param_name] = _wsp_transforms.WSPLogTransform(
                param_name,
                offset=float(offset),
                decades=float(decades)
            )
        elif xform_type == 'logicle':
            # logicle transform has 4 parameters: T, W, M, and A
            # these are attributes of the 'logicle' element
            param_t = find_attribute_value(xform_el, transform_ns, 'T')
            param_w = find_attribute_value(xform_el, transform_ns, 'W')
            param_m = find_attribute_value(xform_el, transform_ns, 'M')
            param_a = find_attribute_value(xform_el, transform_ns, 'A')
            xforms_lut[param_name] = _transforms.LogicleTransform(
                param_name,
                param_t=float(param_t),
                param_w=float(param_w),
                param_m=float(param_m),
                param_a=float(param_a)
            )
        elif xform_type == 'biex':
            # biex transform has 5 parameters, but only 2 are really used
            # these are attributes of the 'biex' element
            param_neg = find_attribute_value(xform_el, transform_ns, 'neg')
            param_width = find_attribute_value(xform_el, transform_ns, 'width')
            # These next 3 exist but are not used.
            # param_length = find_attribute_value(xform_el, transform_ns, 'length')
            # param_maxRange = find_attribute_value(xform_el, transform_ns, 'maxRange')
            # param_pos = find_attribute_value(xform_el, transform_ns, 'pos')
            xforms_lut[param_name] = _wsp_transforms.WSPBiexTransform(
                param_name,
                negative=float(param_neg),
                width=float(param_width)
            )
        else:
            error_msg = "FlowJo transform type '%s' is undocumented and not supported in FlowKit. " % xform_type
            error_msg += "Please edit the workspace in FlowJo and save all channel transformations as either " \
                "linear, log, biex, or logicle"

            raise ValueError(error_msg)

    return xforms_lut


def _convert_wsp_gate(wsp_gate, comp_matrix, xform_lut):
    new_dims = []
    xforms = []

    for dim in wsp_gate.dimensions:
        if comp_matrix is not None:
            pre = comp_matrix['prefix']
            suf = comp_matrix['suffix']
            dim_label = dim.label

            if dim_label.startswith(pre):
                dim_label = re.sub(r'^%s' % pre, '', dim_label)
            if dim_label.endswith(suf):
                dim_label = re.sub(r'%s$' % suf, '', dim_label)

            if dim_label in comp_matrix['detectors']:
                comp_ref = comp_matrix['matrix_name']
            else:
                comp_ref = None
        else:
            dim_label = dim.label
            comp_ref = None

        xform_id = None
        new_dim_min = None
        new_dim_max = None

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
                    v.coordinates[i] = xforms[i].apply(np.array([[float(c)]]))[0][0]

        gate = PolygonGate(wsp_gate.id, wsp_gate.parent, new_dims, vertices)
        gate.uid = wsp_gate.uid
        gate.parent_uid = wsp_gate.parent_uid
    elif isinstance(wsp_gate, GMLRectangleGate):
        gate = wsp_gate
        gate.dimensions = new_dims
    elif isinstance(wsp_gate, WSPEllipsoidGate):
        gate = wsp_gate
    else:
        raise NotImplemented("%s gates for FlowJo workspaces are not currently supported." % type(wsp_gate).__name__)

    return gate


def _recurse_wsp_sub_populations(sub_pop_el, parent_id, gating_ns, data_type_ns):
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
        gate_class = wsp_gate_constructor_lut[gate_type]

        g = gate_class(
            gate_child_el,
            gating_ns,
            data_type_ns
        )

        # replace ID and parent ID with population names, but save the originals
        # so we can re-create the correct hierarchy later.
        # NOTE: Some wsp files have the ID as an attribute of the GatingML-style sub-element,
        #       though it isn't always the case. However, it seems the ID is reliably included
        #       in the parent "Gate" element, saved here in the 'gate_el' variable.
        #       Likewise for parent_id
        gate_id = find_attribute_value(gate_el, gating_ns, 'id')
        g.uid = gate_id
        g.parent_uid = find_attribute_value(gate_el, gating_ns, 'parent_id')
        g.id = pop_name
        g.parent = parent_id

        if g.id == 'TIME':
            print('asdf')

        gates.append(
            {
                'owning_group': owning_group,
                'gate': g
            }
        )

        sub_pop_els = pop_el.findall('Subpopulations', ns_map)
        for el in sub_pop_els:
            gates.extend(_recurse_wsp_sub_populations(el, pop_name, gating_ns, data_type_ns))

    return gates


def parse_wsp(workspace_file_or_path):
    doc_type, root_xml, gating_ns, data_type_ns, transform_ns = _get_xml_type(workspace_file_or_path)

    # first, find SampleList elements
    ns_map = root_xml.nsmap
    groups_el = root_xml.find('Groups', ns_map)
    group_node_els = groups_el.findall('GroupNode', ns_map)
    sample_list_el = root_xml.find('SampleList', ns_map)
    sample_els = sample_list_el.findall('Sample', ns_map)

    for group_node_el in group_node_els:
        # TODO: parse compensation to use for default 'All Samples' group
        pass

    wsp_dict = {}

    for sample_el in sample_els:
        transforms_el = sample_el.find('Transformations', ns_map)
        sample_node_el = sample_el.find('SampleNode', ns_map)
        sample_name = sample_node_el.attrib['name']

        # It appears there is only a single set of xforms per sample, one for each channel.
        # And, the xforms have no IDs. We'll extract it and give it IDs based on ???
        sample_xform_lut = _parse_wsp_transforms(transforms_el, transform_ns, data_type_ns)

        # parse spilloverMatrix elements
        sample_comp = _parse_wsp_compensation(sample_el, transform_ns, data_type_ns)

        # FlowJo WSP gates are nested so we'll have to do a recursive search from the root Sub-populations node
        sample_root_sub_pop_el = sample_node_el.find('Subpopulations', ns_map)

        # FJ WSP gates are stored in non-transformed space. After parsing the XML the values need
        # to be converted to the compensated & transformed space. Also, the recurse_sub_populations
        # function replaces the non-human readable IDs in the XML with population names
        sample_gates = _recurse_wsp_sub_populations(
            sample_root_sub_pop_el,
            None,  # starting at root, so no parent ID
            gating_ns,
            data_type_ns
        )

        for sample_gate in sample_gates:
            if sample_gate['owning_group'] == '':
                group = "All Samples"
            else:
                group = sample_gate['owning_group']
            gate = sample_gate['gate']

            if group not in wsp_dict:
                wsp_dict[group] = {}
            if sample_name not in wsp_dict[group]:
                if sample_comp is None:
                    matrix = None
                else:
                    detectors = sample_comp['detectors']
                    matrix = Matrix(
                        sample_comp['matrix_name'],
                        sample_comp['matrix'],
                        detectors=detectors,
                        fluorochromes=['' for _ in detectors]
                    )

                wsp_dict[group][sample_name] = {
                    'gates': {},
                    'transforms': list(sample_xform_lut.values()),
                    'compensation': matrix
                }

            gate = _convert_wsp_gate(gate, sample_comp, sample_xform_lut)
            # TODO: this will still overwrite re-used gate IDs on different branches
            wsp_dict[group][sample_name]['gates'][gate.uid] = gate

    # finally, convert 'gates' value to tree hierarchy
    for group_id, group_dict in wsp_dict.items():
        for sample_name, sample_dict in group_dict.items():
            tree = _build_hierarchy_tree(sample_dict['gates'], use_uid=True)
            for d in tree.descendants:
                d.name = d.gate.id
            sample_dict['gates'] = tree

    return wsp_dict
