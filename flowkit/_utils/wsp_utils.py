"""
Utility functions related to FlowJo Workspace files (.wsp)
"""
import copy
import datetime
import re
import numpy as np
from lxml import etree
from .sample_utils import FCS_STANDARD_KEYWORDS
from .xml_utils import find_attribute_value, _get_xml_type, _build_hierarchy_tree
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
            try:
                xforms_lut[param_name] = _wsp_transforms.WSPBiexTransform(
                    param_name,
                    negative=float(param_neg),
                    width=float(param_width)
                )
            except ValueError as e:
                raise(ValueError("Channel %s" % param_name, e.args))
        else:
            error_msg = "FlowJo transform type '%s' is undocumented and not supported in FlowKit. " % xform_type
            error_msg += "Please edit the workspace in FlowJo and save all channel transformations as either " \
                "linear, log, biex, or logicle"

            raise ValueError(error_msg)

    return xforms_lut


def _convert_wsp_gate(wsp_gate, comp_matrix, xform_lut, ignore_transforms=False):
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

        if dim_label in xform_lut and not ignore_transforms:
            xform = xform_lut[dim_label]
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
            dim_label,
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

        gate = PolygonGate(wsp_gate.id, wsp_gate.parent, new_dims, vertices)
        gate.uid = wsp_gate.uid
        gate.parent_uid = wsp_gate.parent_uid
    elif isinstance(wsp_gate, GMLRectangleGate):
        gate = wsp_gate
        gate.dimensions = new_dims
    elif isinstance(wsp_gate, WSPEllipsoidGate):
        gate = wsp_gate
    else:
        raise NotImplemented(
            "%s gates for FlowJo workspaces are not currently supported." % type(wsp_gate).__name__
        )

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

        # FlowJo WSP gates are nested so we'll have to do a recursive search from the root
        # Sub-populations node
        sample_root_sub_pop_el = sample_node_el.find('Subpopulations', ns_map)

        if sample_root_sub_pop_el is None:
            continue

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

            gate = _convert_wsp_gate(gate, sample_comp, sample_xform_lut, ignore_transforms=ignore_transforms)
            # TODO: this will still overwrite re-used gate IDs on different branches
            wsp_dict[group][sample_name]['gates'][gate.uid] = gate

    # finally, convert 'gates' value to tree hierarchy
    for group_id, group_dict in wsp_dict.items():
        for sample_name, sample_dict in group_dict.items():
            try:
                tree = _build_hierarchy_tree(sample_dict['gates'], use_uid=True)
            except KeyError as ex:
                err_msg = "The above exception occurred processing gates for %s. "\
                          "This typically happens when the sample has a custom gate" \
                          " that differs from the global sample group gate." % sample_name
                raise KeyError(err_msg) from ex

            for d in tree.descendants:
                d.name = d.gate.id
            sample_dict['gates'] = tree

    return wsp_dict


def _add_transform_to_wsp(parent_el, parameter_label, transform, ns_map):
    if isinstance(transform, _transforms.LinearTransform):
        xform_el = etree.SubElement(parent_el, "{%s}linear" % ns_map['transforms'])
        xform_el.set('{%s}minRange' % ns_map['transforms'], str(transform.param_a))
        xform_el.set('{%s}maxRange' % ns_map['transforms'], str(transform.param_t))
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


def _add_polygon_gate(parent_el, gate, fj_gate_id, gating_strategy, ns_map):
    gate_instance_el = etree.SubElement(parent_el, "{%s}PolygonGate" % ns_map['gating'])
    gate_instance_el.set('quadID', '-1')
    gate_instance_el.set('gateResolution', '256')
    gate_instance_el.set('{%s}id' % ns_map['gating'], "ID%d" % fj_gate_id)

    xform_refs = []
    for dim in gate.dimensions:
        dim_el = etree.SubElement(gate_instance_el, "{%s}dimension" % ns_map['gating'])
        fcs_dim_el = etree.SubElement(dim_el, "{%s}fcs-dimension" % ns_map['data-type'])
        fcs_dim_el.set("{%s}name" % ns_map['data-type'], dim.label)

        xform_refs.append(dim.transformation_ref)

    for vertex in gate.vertices:
        vertex_el = etree.SubElement(gate_instance_el, "{%s}vertex" % ns_map['gating'])
        for dim_idx, coord in enumerate(vertex.coordinates):
            if xform_refs[dim_idx] is not None:
                xform = gating_strategy.get_transform(xform_refs[dim_idx])
                inv_coord = xform.inverse(coord)
            else:
                inv_coord = coord

            coord_el = etree.SubElement(vertex_el, "{%s}coordinate" % ns_map['gating'])
            coord_el.set("{%s}value" % ns_map['data-type'], str(inv_coord))


def _add_rectangle_gate(parent_el, gate, fj_gate_id, gating_strategy, ns_map):
    gate_instance_el = etree.SubElement(parent_el, "{%s}RectangleGate" % ns_map['gating'])
    gate_instance_el.set('percentX', '0')
    gate_instance_el.set('percentY', '0')
    gate_instance_el.set('{%s}id' % ns_map['gating'], "ID%d" % fj_gate_id)

    for dim in gate.dimensions:
        dim_el = etree.SubElement(gate_instance_el, "{%s}dimension" % ns_map['gating'])
        fcs_dim_el = etree.SubElement(dim_el, "{%s}fcs-dimension" % ns_map['data-type'])
        fcs_dim_el.set("{%s}name" % ns_map['data-type'], dim.label)

        xform_ref = dim.transformation_ref

        if xform_ref is not None:
            xform = gating_strategy.get_transform(xform_ref)
            dim_min = xform.inverse(dim.min)
            dim_max = xform.inverse(dim.max)
        else:
            dim_min = dim.min
            dim_max = dim.max

        dim_el.set('{%s}min' % ns_map['gating'], str(dim_min))
        dim_el.set('{%s}max' % ns_map['gating'], str(dim_max))


def _add_sample_node_to_wsp(parent_el, sample_name, sample_id, group_name, gating_strategy, ns_map):
    sample_node_el = etree.SubElement(parent_el, "SampleNode")
    sample_node_el.set('name', sample_name)
    sample_node_el.set('annotation', "")
    sample_node_el.set('owningGroup', "")
    sample_node_el.set('sampleID', str(sample_id))

    sub_pops_el = etree.SubElement(sample_node_el, "Subpopulations")

    gate_fj_id = 1

    for gate_id, gate_path in gating_strategy.get_gate_ids():
        pop_el = etree.SubElement(sub_pops_el, "Population")
        pop_el.set('name', gate_id)
        pop_el.set('annotation', "")
        pop_el.set('owningGroup', group_name)

        gate = gating_strategy.get_gate(gate_id, gate_path)

        # TODO: need to figure out how to generate and store the FlowJo gating IDs
        gate_el = etree.SubElement(pop_el, 'Gate')
        gate_el.set('{%s}id' % ns_map['gating'], "ID%d" % gate_fj_id)

        if isinstance(gate, PolygonGate):
            _add_polygon_gate(gate_el, gate, gate_fj_id, gating_strategy, ns_map)
        elif isinstance(gate, RectangleGate):
            _add_rectangle_gate(gate_el, gate, gate_fj_id, gating_strategy, ns_map)
        else:
            continue

        gate_fj_id += 1


def export_flowjo_wsp(gating_strategy, group_name, samples, file_handle):
    """
    Exports a FlowJo 10 workspace file (.wsp) from the given GatingStrategy instance
    :param gating_strategy: A GatingStrategy instance
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

    gate_ids = gating_strategy.get_gate_ids()
    gates = []
    dim_xform_lut = {}  # keys are dim label, value is a set of xform refs
    for g_id, g_path in gate_ids:
        gate = gating_strategy.get_gate(g_id, g_path)
        gates.append(gate)

        for dim in gate.dimensions:
            if dim.label not in dim_xform_lut.keys():
                dim_xform_lut[dim.label] = set()

            dim_xform_lut[dim.label].add(dim.transformation_ref)

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
    # element.
    curr_sample_id = 1  # each sample needs an ID, we'll start at 1 & increment
    sample_id_lut = {}
    for sample in samples:
        sample_el = etree.SubElement(sample_list_el, "Sample")

        # Start with DataSet sub-element. We don't have the exact
        # file path to the FCS file, so we'll make a relative
        # path using the Sample's original filename, hoping that
        # FlowJo can re-connect the files using that name.
        data_set_el = etree.SubElement(sample_el, "DataSet")
        data_set_el.set('uri', sample.original_filename)
        data_set_el.set('sampleID', str(curr_sample_id))

        # Store sample ID and increment
        sample_id_lut[sample.original_filename] = curr_sample_id
        curr_sample_id += 1

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
            _add_transform_to_wsp(xforms_el, dim, gating_strategy.get_transform(xform), ns_map)

        # Add Keywords sub-element using the Sample metadata
        keywords_el = etree.SubElement(sample_el, "Keywords")
        _add_sample_keywords_to_wsp(keywords_el, sample)

        # Finally, add the SampleNode where all the gates are defined
        _add_sample_node_to_wsp(
            sample_el,
            sample.original_filename,
            sample_id_lut[sample.original_filename],
            group_name,
            gating_strategy,
            ns_map
        )

    et = etree.ElementTree(root)

    et.write(file_handle, encoding="utf-8", xml_declaration=True, pretty_print=True)
