"""
Utility functions related to FlowJo Workspace files (.wsp)
"""
import copy
import datetime
import re
import numpy as np
from lxml import etree
from .sample_utils import FCS_STANDARD_KEYWORDS
from .xml_utils import find_attribute_value, _get_xml_type
from .._models.dimension import Dimension
# noinspection PyProtectedMember
from .._models.transforms._matrix import Matrix
# noinspection PyProtectedMember
from .._models.transforms import _transforms, _wsp_transforms
# noinspection PyProtectedMember
from .._models.gates._gates import PolygonGate, RectangleGate
# noinspection PyProtectedMember
from .._models.gates._gml_gates import \
    GMLBooleanGate, \
    GMLQuadrantGate, \
    GMLPolygonGate, \
    GMLRectangleGate
# noinspection PyProtectedMember
from .._models.gates._wsp_gates import WSPEllipsoidGate

wsp_gate_constructor_lut = {
    'RectangleGate': GMLRectangleGate,
    'PolygonGate': GMLPolygonGate,
    'EllipsoidGate': WSPEllipsoidGate,
    'QuadrantGate': GMLQuadrantGate,
    'BooleanGate': GMLBooleanGate
}


def _parse_wsp_compensation(sample_el, transform_ns, data_type_ns):
    # Find spilloverMatrix elements, not sure if there should be just a
    # single matrix or multiple. Going with a single one now since there
    # do not appear to be comp references in the WSP gate elements.
    matrix_els = sample_el.findall(
        '%s:spilloverMatrix' % transform_ns,
        namespaces=sample_el.nsmap
    )

    matrix_els_cnt = len(matrix_els)

    if matrix_els_cnt > 1:
        raise ValueError("Multiple spillover matrices per sample are not supported.")
    elif matrix_els_cnt == 0:
        return None

    matrix_el = matrix_els[0]

    # we'll ignore the non-human readable matrix ID and use the name instead
    matrix_name = matrix_el.attrib['name']
    matrix_prefix = matrix_el.attrib['prefix']
    matrix_suffix = matrix_el.attrib['suffix']

    detectors = []
    matrix_array = []

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

        matrix_array.append(matrix_row)

    matrix_array = np.array(matrix_array)
    matrix = Matrix(
        matrix_name,
        matrix_array,
        detectors=detectors,
        fluorochromes=['' for _ in detectors]
    )

    matrix_dict = {
        'matrix_name': matrix_name,
        'prefix': matrix_prefix,
        'suffix': matrix_suffix,
        'detectors': detectors,
        'matrix_array': matrix_array,
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
            param_length = find_attribute_value(xform_el, transform_ns, 'length')
            param_max_range = find_attribute_value(xform_el, transform_ns, 'maxRange')
            param_pos = find_attribute_value(xform_el, transform_ns, 'pos')
            param_pos = round(float(param_pos), 2)

            if param_length != '256':
                raise ValueError("FlowJo biex 'length' parameter value of %s is not supported." % param_length)

            xforms_lut[param_name] = _wsp_transforms.WSPBiexTransform(
                param_name,
                negative=float(param_neg),
                width=float(param_width),
                positive=float(param_pos),
                max_value=float(param_max_range)
            )
        elif xform_type == 'fasinh':
            # FlowJo's implementation of fasinh is slightly different from GML,
            # and uses an additional 'length' scale factor. However, this scaling
            # doesn't seem to affect the results and we can use the regular
            # GML version of asinh. The xform_el also contains other
            # unnecessary parameters: 'length', 'maxRange', and 'W'
            param_t = find_attribute_value(xform_el, transform_ns, 'T')
            param_a = find_attribute_value(xform_el, transform_ns, 'A')
            param_m = find_attribute_value(xform_el, transform_ns, 'M')
            xforms_lut[param_name] = _transforms.AsinhTransform(
                param_name,
                param_t=float(param_t),
                param_m=float(param_m),
                param_a=float(param_a)
            )
        else:
            error_msg = "FlowJo transform type '%s' is undocumented and not supported in FlowKit. " % xform_type
            error_msg += "Please edit the workspace in FlowJo and save all channel transformations as either " \
                "linear, log, biex, logicle, or ArcSinh"

            raise ValueError(error_msg)

    return xforms_lut


def _convert_wsp_gate(wsp_gate, comp_matrix, xform_lut, ignore_transforms=False):
    new_dims = []
    xforms = []

    for dim in wsp_gate.dimensions:
        if comp_matrix is not None:
            pre = comp_matrix['prefix']
            suf = comp_matrix['suffix']
            dim_id = dim.id

            if dim_id.startswith(pre):
                dim_id = re.sub(r'^%s' % pre, '', dim_id)
            if dim_id.endswith(suf):
                dim_id = re.sub(r'%s$' % suf, '', dim_id)

            if dim_id in comp_matrix['detectors']:
                comp_ref = comp_matrix['matrix_name']
            else:
                comp_ref = None
        else:
            dim_id = dim.id
            comp_ref = None

        xform_id = None
        new_dim_min = None
        new_dim_max = None

        if dim_id in xform_lut and not ignore_transforms:
            xform = xform_lut[dim_id]
            xforms.append(xform)  # need these later for vertices, coordinates, etc.
            xform_id = xform.id
            if dim.min is not None:
                new_dim_min = xform.apply(np.array([[float(dim.min)]]))

            if dim.max is not None:
                new_dim_max = xform.apply(np.array([[float(dim.max)]]))
        else:
            xforms.append(None)
            if dim.min is not None:
                new_dim_min = float(dim.min)

            if dim.max is not None:
                new_dim_max = float(dim.max)

        new_dim = Dimension(
            dim_id,
            comp_ref,
            xform_id,
            range_min=new_dim_min,
            range_max=new_dim_max
        )
        new_dims.append(new_dim)

    if isinstance(wsp_gate, GMLPolygonGate):
        # convert vertices using saved xforms
        vertices = copy.deepcopy(wsp_gate.vertices)
        for v in vertices:
            for i, c in enumerate(v.coordinates):
                if xforms[i] is not None:
                    v.coordinates[i] = xforms[i].apply(np.array([[float(c)]]))[0][0]

        gate = PolygonGate(wsp_gate.gate_name, wsp_gate.parent, new_dims, vertices)
    elif isinstance(wsp_gate, GMLRectangleGate):
        gate = copy.deepcopy(wsp_gate)
        gate.dimensions = new_dims
    elif isinstance(wsp_gate, WSPEllipsoidGate):
        gate = wsp_gate.convert_to_ellipsoid_gate(xforms)
        gate.dimensions = new_dims
    else:
        raise NotImplemented(
            "%s gates for FlowJo workspaces are not currently supported." % type(wsp_gate).__name__
        )

    return gate


def _recurse_wsp_sub_populations(sub_pop_el, gate_path, gating_ns, data_type_ns):
    gates = []
    ns_map = sub_pop_el.nsmap

    if gate_path is None:
        # here we'll create the gate path as a list b/c we will append to it for recursion
        # however, when it is finally stored in the list of gate dicts we will convert to tuple
        gate_path = ['root']
        parent_id = None
    else:
        parent_id = gate_path[-1]

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
        g.gate_name = pop_name
        g.parent = parent_id

        gates.append(
            {
                'owning_group': owning_group,
                'gate': g,
                'gate_path': tuple(gate_path)  # converting to tuple!
            }
        )

        sub_pop_els = pop_el.findall('Subpopulations', ns_map)
        child_gate_path = copy.copy(gate_path)
        child_gate_path.append(pop_name)
        for el in sub_pop_els:
            gates.extend(_recurse_wsp_sub_populations(el, child_gate_path, gating_ns, data_type_ns))

    return gates


def _parse_wsp_groups(group_node_els, ns_map, gating_ns, data_type_ns):
    wsp_groups = {}

    for group_node_el in group_node_els:
        group_name = group_node_el.attrib['name']
        group_samples = []
        group_gates = []

        # Group membership for samples is found in the 'Group' branch.
        # The SampleRefs element will contain a SampleRef element for
        # each sample assigned to the group.
        group_el = group_node_el.find('Group', ns_map)
        if group_el is not None:
            group_sample_refs_el = group_el.find('SampleRefs', ns_map)

            if group_sample_refs_el is not None:
                group_sample_ref_els = group_sample_refs_el.findall('SampleRef', ns_map)
                for s_ref_el in group_sample_ref_els:
                    group_s_id = s_ref_el.attrib['sampleID']
                    group_samples.append(group_s_id)

        group_root_sub_pop_el = group_node_el.find('Subpopulations', ns_map)

        # ignore groups with no sub-populations
        if group_root_sub_pop_el is not None:
            group_gates = _recurse_wsp_sub_populations(
                group_root_sub_pop_el,
                None,  # starting at root, so no gate path
                gating_ns,
                data_type_ns
            )

        wsp_groups[group_name] = {
            'gates': group_gates,
            'samples': group_samples
        }

    return wsp_groups


def _parse_wsp_samples(sample_els, ns_map, gating_ns, transform_ns, data_type_ns):
    wsp_samples = {}

    for sample_el in sample_els:
        transforms_el = sample_el.find('Transformations', ns_map)
        sample_node_el = sample_el.find('SampleNode', ns_map)
        sample_name = sample_node_el.attrib['name']
        sample_id = sample_node_el.attrib['sampleID']

        # It appears there is only a single set of xforms per sample, one for each channel.
        # And, the xforms have no IDs. We'll extract it and give it IDs based on ???
        sample_xform_lut = _parse_wsp_transforms(transforms_el, transform_ns, data_type_ns)

        # parse spilloverMatrix elements
        sample_comp = _parse_wsp_compensation(sample_el, transform_ns, data_type_ns)

        # FlowJo WSP gates are nested so we'll have to do a recursive search from the root
        # Sub-populations node
        sample_root_sub_pop_el = sample_node_el.find('Subpopulations', ns_map)

        if sample_root_sub_pop_el is None:
            continue

        sample_gates = _recurse_wsp_sub_populations(
            sample_root_sub_pop_el,
            None,  # starting at root, so no gate path
            gating_ns,
            data_type_ns
        )

        # sample gate LUT will store everything we need to convert sample gates,
        # including any custom gates (ones with empty string owning groups).
        wsp_samples[sample_id] = {
            'sample_name': sample_name,
            'custom_gate_ids': set(),
            'custom_gates': [],
            'transforms': sample_xform_lut,
            'comp': sample_comp
        }

        for sample_gate in sample_gates:
            if sample_gate['owning_group'] == '':
                # If the owning group is an empty string, it is a custom gate for that sample
                # that is potentially used in another group. However, it appears that if a
                # sample has a custom gate then that custom gate cannot be further customized.
                # Since there is only a single custom gate per gate name per sample, then we
                # can create a LUT of custom gates per sample
                wsp_samples[sample_id]['custom_gate_ids'].add(sample_gate['gate'].gate_name)
                wsp_samples[sample_id]['custom_gates'].append(
                    {
                        'gate': sample_gate['gate'],
                        'gate_path': sample_gate['gate_path']
                    }
                )

    return wsp_samples


def parse_wsp(workspace_file_or_path, ignore_transforms=False):
    """
    Converts a FlowJo 10 workspace file (.wsp) into a nested Python dictionary with the following structure:

        wsp_dict[group][sample_name] = {
            'gates': {gate_id: gate_instance, ...},
            'transforms': transforms_list,
            'compensation': matrix
        }

    :param workspace_file_or_path: A FlowJo .wsp file or file path
    :param ignore_transforms: Some FlowJo 10 transforms are incompatible with FlowKit. This boolean argument
        allows parsing workspace files with unsupported transforms without raising errors. Default is False.
    :return: dict
    """
    doc_type, root_xml, gating_ns, data_type_ns, transform_ns = _get_xml_type(workspace_file_or_path)

    # first, find 1st level elements:
    #     - Groups -> GroupNode
    #     - SampleList -> Sample
    ns_map = root_xml.nsmap
    groups_el = root_xml.find('Groups', ns_map)
    group_node_els = groups_el.findall('GroupNode', ns_map)
    sample_list_el = root_xml.find('SampleList', ns_map)
    sample_els = sample_list_el.findall('Sample', ns_map)

    # Find all group gates before looking a the custom sample gates.
    # Custom sample gates are defined in the SampleList branch and
    # are indicated by having an empty string for the owningGroup.
    # Only 1 custom sample gate can exist per sample, regardless
    # of sample group. So, if a group contains a gate with a name
    # that matches a custom sample gate name, then the custom one
    # overrides the group gate.
    wsp_groups = _parse_wsp_groups(group_node_els, ns_map, gating_ns, data_type_ns)

    # Now parse the Sample elements in SampleList looking for custom sample gates
    wsp_samples = _parse_wsp_samples(sample_els, ns_map, gating_ns, transform_ns, data_type_ns)

    # Now we can assemble the complete gating strategies but the gates
    # need to be converted to the transformed space according to the
    # correct specified transforms. This is because FlowJo gates are
    # stored in the untransformed space.
    wsp_dict = {}

    for group_id, group_dict in wsp_groups.items():
        wsp_dict[group_id] = {}

        for group_sample_id in group_dict['samples']:
            sample_dict = wsp_samples[group_sample_id]
            sample_name = sample_dict['sample_name']

            group_sample_gate_names = []
            group_sample_gates = []

            for group_gate in group_dict['gates']:
                group_gate_name = group_gate['gate'].gate_name

                tmp_gate = copy.deepcopy(group_gate['gate'])

                if group_gate_name in sample_dict['custom_gate_ids']:
                    group_gate_path = group_gate['gate_path']
                    for sample_gate_dict in sample_dict['custom_gates']:
                        # noinspection PyTypeChecker
                        tmp_sample_gate = sample_gate_dict['gate']
                        # noinspection PyTypeChecker
                        tmp_sample_gate_path = sample_gate_dict['gate_path']
                        # noinspection PyUnresolvedReferences
                        if group_gate_path == tmp_sample_gate_path and tmp_sample_gate.gate_name == group_gate_name:
                            # found a match, overwrite tmp_gate
                            tmp_gate = tmp_sample_gate

                tmp_gate = _convert_wsp_gate(
                    tmp_gate,
                    sample_dict['comp'],
                    sample_dict['transforms'],
                    ignore_transforms=ignore_transforms
                )

                group_sample_gate_names.append(group_gate_name)
                group_sample_gates.append(
                    {
                        'gate': tmp_gate,
                        'gate_path': group_gate['gate_path']
                    }
                )

            # Now, we need to check if there were only custom sample gates
            # and no group gates. In this case the above would never have
            # found the custom sample gates, but we don't want to replicate
            # them.
            for sample_gate_dict in sample_dict['custom_gates']:
                # noinspection PyTypeChecker
                sample_gate = sample_gate_dict['gate']
                # noinspection PyTypeChecker
                sample_gate_path = sample_gate_dict['gate_path']

                # noinspection PyUnresolvedReferences
                if sample_gate.gate_name not in group_sample_gate_names:
                    # noinspection PyTypeChecker
                    tmp_gate = _convert_wsp_gate(
                        sample_gate,
                        sample_dict['comp'],
                        sample_dict['transforms'],
                        ignore_transforms=ignore_transforms
                    )
                    group_sample_gates.append(
                        {
                            'gate': tmp_gate,
                            'gate_path': sample_gate_path
                        }
                    )

            if sample_dict['comp'] is not None:
                matrix = sample_dict['comp']['matrix']
            else:
                matrix = None

            wsp_dict[group_id][sample_name] = {
                'gates': group_sample_gates,
                'transforms': list(sample_dict['transforms'].values()),
                'compensation': matrix
            }

    return wsp_dict


def _add_matrix_to_wsp(parent_el, prefix, matrix, ns_map):
    matrix_el = etree.SubElement(parent_el, "{%s}spilloverMatrix" % ns_map['transforms'])
    matrix_el.set('spectral', "0")
    matrix_el.set('prefix', prefix)
    matrix_el.set('name', matrix.id)
    matrix_el.set('version', "FlowJo-10.7.1")
    # TODO: need to set id?
    matrix_el.set('suffix', "")

    params_el = etree.SubElement(matrix_el, "{%s}parameters" % ns_map['data-type'])

    # add parameter labels & their prefix
    for param_label in matrix.detectors:
        param_el = etree.SubElement(params_el, "{%s}parameter" % ns_map['data-type'])
        param_el.set('{%s}name' % ns_map['data-type'], param_label)
        param_el.set('userProvidedCompInfix', "".join([prefix, param_label]))

    # add the matrix array values
    matrix_array = matrix.matrix
    for i_row, param_label in enumerate(matrix.detectors):
        spillover_el = etree.SubElement(matrix_el, "{%s}spillover" % ns_map['transforms'])
        spillover_el.set('{%s}parameter' % ns_map['data-type'], param_label)
        spillover_el.set('userProvidedCompInfix', "".join([prefix, param_label]))

        for i_col, param_label2 in enumerate(matrix.detectors):
            matrix_value = matrix_array[i_row, i_col]

            coef_el = etree.SubElement(spillover_el, "{%s}coefficient" % ns_map['transforms'])
            coef_el.set('{%s}parameter' % ns_map['data-type'], param_label2)
            coef_el.set('{%s}value' % ns_map['transforms'], str(matrix_value))


def _add_transform_to_wsp(parent_el, parameter_label, transform, ns_map):
    if isinstance(transform, _transforms.LinearTransform):
        xform_el = etree.SubElement(parent_el, "{%s}linear" % ns_map['transforms'])
        xform_el.set('{%s}minRange' % ns_map['transforms'], str(transform.param_a))
        xform_el.set('{%s}maxRange' % ns_map['transforms'], str(transform.param_t))
    elif isinstance(transform, _transforms.LogicleTransform):
        xform_el = etree.SubElement(parent_el, "{%s}logicle" % ns_map['transforms'])
        xform_el.set('{%s}length' % ns_map['transforms'], "256")
        xform_el.set('{%s}T' % ns_map['transforms'], str(transform.param_t))
        xform_el.set('{%s}A' % ns_map['transforms'], str(transform.param_a))
        xform_el.set('{%s}W' % ns_map['transforms'], str(transform.param_w))
        xform_el.set('{%s}M' % ns_map['transforms'], str(transform.param_m))
    else:
        raise NotImplementedError("Transform type %s is not yet supported" % type(transform))

    param_el = etree.SubElement(xform_el, "{%s}parameter" % ns_map['data-type'])
    param_el.set('{%s}name' % ns_map['data-type'], str(parameter_label))


def _add_sample_keywords_to_wsp(parent_el, sample):
    # setup some regex vars for detecting dynamic FCS standard keywords
    # i.e. PnN, PnS, etc.
    regex_pnx = re.compile(r"^p(\d+)[bdefglnoprst]$", re.IGNORECASE)

    # start by adding special FlowJo keyword for the FCS version
    kw_el = etree.SubElement(parent_el, "Keyword")
    kw_el.set('name', "FJ_FCS_VERSION")
    kw_el.set('value', str(sample.version))

    for kw, val in sample.metadata.items():
        fcs_kw = False
        if kw in FCS_STANDARD_KEYWORDS:
            fcs_kw = True
        else:
            match = regex_pnx.match(kw)
            if match:
                fcs_kw = True

        kw_el = etree.SubElement(parent_el, "Keyword")
        if fcs_kw:
            kw_el.set('name', "$%s" % kw.upper())
        else:
            kw_el.set('name', kw.upper())

        kw_el.set('value', val)


def _add_polygon_gate(parent_el, gate, fj_gate_id, fj_parent_gate_id, gating_strategy, comp_prefix, ns_map):
    gate_instance_el = etree.SubElement(parent_el, "{%s}PolygonGate" % ns_map['gating'])
    gate_instance_el.set('quadID', '-1')
    gate_instance_el.set('gateResolution', '256')
    gate_instance_el.set('{%s}id' % ns_map['gating'], "ID%s" % fj_gate_id)
    if fj_parent_gate_id is not None:
        gate_instance_el.set('{%s}parent_id' % ns_map['gating'], "ID%s" % fj_parent_gate_id)

    xform_refs = []
    for dim in gate.dimensions:
        # use comp prefix for label except for scatter and time channels
        if dim.id[:4] in ['FSC-', 'SSC-', 'Time']:
            dim_id = dim.id
        else:
            dim_id = comp_prefix + dim.id

        dim_el = etree.SubElement(gate_instance_el, "{%s}dimension" % ns_map['gating'])
        fcs_dim_el = etree.SubElement(dim_el, "{%s}fcs-dimension" % ns_map['data-type'])
        fcs_dim_el.set("{%s}name" % ns_map['data-type'], dim_id)

        xform_refs.append(dim.transformation_ref)

    for vertex in gate.vertices:
        vertex_el = etree.SubElement(gate_instance_el, "{%s}vertex" % ns_map['gating'])
        for dim_idx, coord in enumerate(vertex.coordinates):
            if xform_refs[dim_idx] is not None:
                xform = gating_strategy.get_transform(xform_refs[dim_idx])
                inv_coord = xform.inverse(np.array([[coord]]))[0, 0]
            else:
                inv_coord = coord

            coord_el = etree.SubElement(vertex_el, "{%s}coordinate" % ns_map['gating'])
            coord_el.set("{%s}value" % ns_map['data-type'], str(inv_coord))


def _add_rectangle_gate(parent_el, gate, fj_gate_id, fj_parent_gate_id, gating_strategy, comp_prefix, ns_map):
    gate_instance_el = etree.SubElement(parent_el, "{%s}RectangleGate" % ns_map['gating'])
    gate_instance_el.set('percentX', '0')
    gate_instance_el.set('percentY', '0')
    gate_instance_el.set('{%s}id' % ns_map['gating'], "ID%s" % fj_gate_id)
    if fj_parent_gate_id is not None:
        gate_instance_el.set('{%s}parent_id' % ns_map['gating'], "ID%s" % fj_parent_gate_id)

    for dim in gate.dimensions:
        # use comp prefix for label except for scatter and time channels
        if dim.id[:4] in ['FSC-', 'SSC-', 'Time']:
            dim_id = dim.id
        else:
            dim_id = comp_prefix + dim.id

        dim_el = etree.SubElement(gate_instance_el, "{%s}dimension" % ns_map['gating'])
        fcs_dim_el = etree.SubElement(dim_el, "{%s}fcs-dimension" % ns_map['data-type'])
        fcs_dim_el.set("{%s}name" % ns_map['data-type'], dim_id)

        xform_ref = dim.transformation_ref

        if xform_ref is not None:
            xform = gating_strategy.get_transform(xform_ref)
            dim_min = xform.inverse(np.array([[dim.min]]))[0, 0]
            dim_max = xform.inverse(np.array([[dim.max]]))[0, 0]
        else:
            dim_min = dim.min
            dim_max = dim.max

        dim_el.set('{%s}min' % ns_map['gating'], str(dim_min))
        dim_el.set('{%s}max' % ns_map['gating'], str(dim_max))


def _add_group_node_to_wsp(parent_el, group_name, sample_id_list):
    group_node_el = etree.SubElement(parent_el, "GroupNode")
    group_node_el.set('name', group_name)
    group_node_el.set('annotation', "")
    group_node_el.set('owningGroup', group_name)

    group_el = etree.SubElement(group_node_el, "Group")
    group_el.set('name', group_name)

    sample_refs_el = etree.SubElement(group_el, "SampleRefs")

    for sample_id in sample_id_list:
        sample_ref_el = etree.SubElement(sample_refs_el, "SampleRef")
        sample_ref_el.set('sampleID', sample_id)


def _recurse_add_sub_populations(parent_el, gate_id, gate_path, gating_strategy, gate_fj_id_lut, comp_prefix, ns_map):
    # first, add given gate to parent XML element inside it's own Population element
    pop_el = etree.SubElement(parent_el, "Population")
    pop_el.set('name', gate_id)
    pop_el.set('annotation', "")
    pop_el.set('owningGroup', "")

    # Lookup the gate's FJ ID
    fj_id = gate_fj_id_lut[(gate_id, tuple(gate_path))]

    # Determine if gate has a parent gate & lookup it's FJ ID
    if len(gate_path) > 1:
        # gate has a true parent gate
        parent_gate_id = gate_path[-1]
        parent_gate_path = gate_path[:-1]
        parent_fj_id = gate_fj_id_lut[(parent_gate_id, tuple(parent_gate_path))]
    else:
        parent_fj_id = None

    gate_el = etree.SubElement(pop_el, 'Gate')
    gate_el.set('{%s}id' % ns_map['gating'], "ID%s" % fj_id)

    if parent_fj_id is not None:
        gate_el.set('{%s}parent_id' % ns_map['gating'], "ID%s" % parent_fj_id)

    # Get the gate instance to determine the gate class
    gate = gating_strategy.get_gate(gate_id, gate_path)

    if isinstance(gate, PolygonGate):
        _add_polygon_gate(gate_el, gate, fj_id, parent_fj_id, gating_strategy, comp_prefix, ns_map)
    elif isinstance(gate, RectangleGate):
        _add_rectangle_gate(gate_el, gate, fj_id, parent_fj_id, gating_strategy, comp_prefix, ns_map)
    else:
        raise NotImplementedError("Exporting %s gates is not yet implemented" % str(gate.__class__))

    # If there are child gates, create a new Sub-pop element and recurse
    child_gates = gating_strategy.get_child_gates(gate_id, gate_path)
    if len(child_gates) > 0:
        sub_pops_el = etree.SubElement(pop_el, "Subpopulations")

        # child gate path will be the parent's gate path plus the parent ID
        child_gate_path = copy.deepcopy(gate_path)
        child_gate_path = child_gate_path + (gate_id,)
        for child_gate in child_gates:
            _recurse_add_sub_populations(
                sub_pops_el,
                child_gate.gate_name,
                child_gate_path,
                gating_strategy,
                gate_fj_id_lut,
                comp_prefix,
                ns_map
            )


def _add_sample_node_to_wsp(parent_el, sample_name, sample_id, gating_strategy, comp_prefix_lut, ns_map):
    sample_node_el = etree.SubElement(parent_el, "SampleNode")
    sample_node_el.set('name', sample_name)
    sample_node_el.set('annotation', "")
    sample_node_el.set('owningGroup', "")
    sample_node_el.set('sampleID', sample_id)

    sub_pops_el = etree.SubElement(sample_node_el, "Subpopulations")

    gate_fj_id = 1
    gate_fj_id_lut = {}
    for gate_id, gate_path in gating_strategy.get_gate_ids():
        gate_fj_id_lut[(gate_id, tuple(gate_path))] = str(sample_id) + str(gate_fj_id)
        gate_fj_id += 1

    root_gates = gating_strategy.get_root_gates()

    # need to find a matching compensation to add the correct comp prefix to parameter labels
    comp_ids = gating_strategy.comp_matrices.keys()

    # there really shouldn't be more than 1 compensation matrix and we only support 1 for now
    comp_prefix = ''
    for comp_id in comp_ids:
        if comp_id in comp_prefix_lut:
            comp_prefix = comp_prefix_lut[comp_id]
            break

    for gate in root_gates:
        _recurse_add_sub_populations(
            sub_pops_el,
            gate.gate_name,
            ('root',),
            gating_strategy,
            gate_fj_id_lut,
            comp_prefix,
            ns_map
        )


def export_flowjo_wsp(group_gating_strategies, group_name, samples, file_handle):
    """
    Exports a FlowJo 10 workspace file (.wsp) from the given GatingStrategy instance
    :param group_gating_strategies: dictionary of Session sample group GatingStrategy instances
    :param group_name: text string label for sample group
    :param samples: list of Sample instances associated with the sample group
    :param file_handle: File handle for exported FlowJo workspace file
    :return: None
    """
    # < Workspace
    # version = "20.0"
    # modDate = "Thu Aug 20 12:19:04 EDT 2020"
    # flowJoVersion = "10.6.2"
    # curGroup = "All Samples"
    # xmlns: xsi = "http://www.w3.org/2001/XMLSchema-instance"
    # xmlns: gating = "http://www.isac-net.org/std/Gating-ML/v2.0/gating"
    # xmlns: transforms = "http://www.isac-net.org/std/Gating-ML/v2.0/transformations"
    # xmlns: data - type = "http://www.isac-net.org/std/Gating-ML/v2.0/datatypes"
    ns_g = "http://www.isac-net.org/std/Gating-ML/v2.0/gating"
    ns_dt = "http://www.isac-net.org/std/Gating-ML/v2.0/datatypes"
    ns_xform = "http://www.isac-net.org/std/Gating-ML/v2.0/transformations"
    ns_map = {
        'gating': ns_g,
        'data-type': ns_dt,
        'transforms': ns_xform
    }

    now = datetime.datetime.now().astimezone()
    mod_date = now.strftime('%a %b %d %H:%M:%S %Z %Y')

    root = etree.Element('Workspace', nsmap=ns_map)
    root.set('version', "20.0")
    root.set('modDate', mod_date)
    root.set('flowJoVersion', "10.6.2")
    root.set('curGroup', "All Samples")

    matrices_el = etree.SubElement(root, "Matrices")
    groups_el = etree.SubElement(root, "Groups")
    sample_list_el = etree.SubElement(root, "SampleList")

    template_strategy = group_gating_strategies['template']
    sample_strategies = group_gating_strategies['samples']

    # For now, we'll assume all the comps are in the template
    comp_prefix_counter = 0
    comp_prefix_lut = {}
    for matrix_id, matrix in template_strategy.comp_matrices.items():
        if comp_prefix_counter == 0:
            comp_prefix = 'Comp-'
        else:
            comp_prefix = 'Comp%d-' % comp_prefix_counter

        comp_prefix_lut[matrix_id] = comp_prefix

        _add_matrix_to_wsp(matrices_el, comp_prefix, matrix, ns_map)

        comp_prefix_counter += 1

    # create FJ sample ID LUT (used in various places)
    curr_sample_id = 1  # each sample needs an ID, we'll start at 1 & increment
    sample_id_lut = {}
    for sample in samples:
        # Store sample ID and increment
        sample_id_lut[sample.original_filename] = str(curr_sample_id)
        curr_sample_id += 1

    _add_group_node_to_wsp(groups_el, group_name, sample_id_lut.values())

    gate_ids = template_strategy.get_gate_ids()
    gates = []
    dim_xform_lut = {}  # keys are dim label, value is a set of xform refs

    # Also assume the xforms for all samples are the sam
    for g_id, g_path in gate_ids:
        gate = template_strategy.get_gate(g_id, g_path)
        gates.append(gate)

        for dim in gate.dimensions:
            if dim.id not in dim_xform_lut.keys():
                dim_xform_lut[dim.id] = set()

            dim_xform_lut[dim.id].add(dim.transformation_ref)

    for dim in dim_xform_lut:
        xform_refs = list(dim_xform_lut[dim])
        if len(xform_refs) > 1:
            raise ValueError(
                "The given GatingStrategy is incompatible with FlowJo. "
                "Multiple transformations are not allowed for the same parameter."
                "Parameter %s has multiple transformation references." % dim
            )

        # After verifying there's only 1 transformation reference, set
        # the value in the LUT to the single reference string
        dim_xform_lut[dim] = xform_refs[0]

    # Build SampleList branch
    # The SampleList element contains a series of Sample elements.
    # Each Sample element contains one of the following elements:
    #     - DataSet
    #     - Transformations
    #     - Keywords
    #     - SampleNode
    for sample in samples:
        sample_id = sample_id_lut[sample.original_filename]
        sample_strategy = sample_strategies[sample.original_filename]

        sample_el = etree.SubElement(sample_list_el, "Sample")

        # Start with DataSet sub-element. We don't have the exact
        # file path to the FCS file, so we'll make a relative
        # path using the Sample's original filename, hoping that
        # FlowJo can re-connect the files using that name.
        data_set_el = etree.SubElement(sample_el, "DataSet")
        data_set_el.set('uri', sample.original_filename)
        data_set_el.set('sampleID', sample_id)

        # Transforms in FlowJo are organized differently than in GatingML
        # or a FlowKit GatingStrategy. Instead of transforms being created
        # and referenced by one or more parameters, FlowJo transforms contain
        # the parameter as a child element. This means they cannot be created
        # and re-used for multiple parameters and that the parameter label
        # must be known prior to creating the transform. So, we must dig into
        # the Dimensions of each gate in the entire hierarchy to find all the
        # parameter labels and their reference FlowKit transform.
        xforms_el = etree.SubElement(sample_el, "Transformations")
        for dim, xform in dim_xform_lut.items():
            if xform is None:
                continue
            _add_transform_to_wsp(xforms_el, dim, sample_strategy.get_transform(xform), ns_map)

            # We also need to add comp-prefixed param transforms (excluding scatter and Time channels)
            if dim[:4] in ['FSC-', 'SSC-', 'Time']:
                continue

            for comp_prefix in comp_prefix_lut.values():
                _add_transform_to_wsp(xforms_el, comp_prefix + dim, sample_strategy.get_transform(xform), ns_map)

        # Add Keywords sub-element using the Sample metadata
        keywords_el = etree.SubElement(sample_el, "Keywords")
        _add_sample_keywords_to_wsp(keywords_el, sample)

        # Finally, add the SampleNode where all the gates are defined
        _add_sample_node_to_wsp(
            sample_el,
            sample.original_filename,
            sample_id,
            sample_strategy,
            comp_prefix_lut,
            ns_map
        )

    et = etree.ElementTree(root)

    et.write(file_handle, encoding="utf-8", xml_declaration=True, pretty_print=True)
