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


def construct_gates(gating_strategy, root_gml):
    # first, find Groups & SampleList elements
    groups_el = root_gml.find('Groups', root_gml.nsmap)
    sample_list_el = root_gml.find('SampleList', root_gml.nsmap)

    # next, find which samples belong to which groups
    group_node_els = groups_el.findall('GroupNode', root_gml.nsmap)
    sample_els = sample_list_el.findall('Sample', root_gml.nsmap)

    # Note, sample IDs are strings
    group_sample_lut = {}

    for group_node_el in group_node_els:
        group_el = group_node_el.find('Group', root_gml.nsmap)
        group_sample_refs_el = group_el.find('SampleRefs', root_gml.nsmap)

        if group_sample_refs_el is None:
            # TODO: handle 'Compensation' group if necessary
            continue
        group_sample_els = group_sample_refs_el.findall('SampleRef', root_gml.nsmap)

        sample_ids = []

        for sample_el in group_sample_els:
            sample_ids.append(sample_el.attrib['sampleID'])

        group_sample_lut[group_node_el.attrib['name']] = sample_ids

    samples = {}

    for sample_el in sample_els:
        xform_el = sample_el.find('Transformations', root_gml.nsmap)
        sample_node_el = sample_el.find('SampleNode', root_gml.nsmap)

        sample_id = sample_node_el.attrib['sampleID']
        sample_name = sample_node_el.attrib['name']

        # FlowJo WSP gates are nested so we'll have to do a recursive search from the root Sub-populations node
        sample_root_sub_pop_el = sample_node_el.find('Subpopulations', root_gml.nsmap)
        sample_gates = recurse_sub_populations(sample_root_sub_pop_el, gating_strategy, root_gml.nsmap)

        samples[sample_id] = {
            'name': sample_name,
            'gates': sample_gates
        }

    gates_dict = {}

    return gates_dict


def recurse_sub_populations(sub_pop_el, gating_strategy, nsmap):
    gates = []

    pop_els = sub_pop_el.findall('Population', nsmap)
    for pop_el in pop_els:
        pop_name = pop_el.attrib['name']
        owning_group = pop_el.attrib['owningGroup']
        gate_el = pop_el.find('Gate', nsmap)

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

        sub_pop_els = pop_el.findall('Subpopulations', nsmap)
        for el in sub_pop_els:
            gates.extend(recurse_sub_populations(el, gating_strategy, nsmap))

    return gates
