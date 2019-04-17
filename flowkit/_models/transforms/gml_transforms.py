from flowkit import _gml_utils
from .transforms import \
    RatioTransform, \
    LinearTransform, \
    LogTransform, \
    HyperlogTransform, \
    LogicleTransform, \
    AsinhTransform


class RatioGMLTransform(RatioTransform):
    def __init__(
            self,
            xform_element,
            xform_namespace,
            data_type_namespace
    ):
        t_id = _gml_utils.find_attribute_value(xform_element, xform_namespace, 'id')

        f_ratio_els = xform_element.findall(
            '%s:fratio' % xform_namespace,
            namespaces=xform_element.nsmap
        )

        if len(f_ratio_els) == 0:
            raise ValueError(
                "Ratio transform must specify an 'fratio' element (line %d)" % xform_element.sourceline
            )

        # f ratio transform has 3 parameters: A, B, and C
        # these are attributes of the 'fratio' element
        param_a = _gml_utils.find_attribute_value(f_ratio_els[0], xform_namespace, 'A')
        param_b = _gml_utils.find_attribute_value(f_ratio_els[0], xform_namespace, 'B')
        param_c = _gml_utils.find_attribute_value(f_ratio_els[0], xform_namespace, 'C')

        if None in [param_a, param_b, param_c]:
            raise ValueError(
                "Ratio transform must provide an 'A', a 'B', and a 'C' "
                "attribute (line %d)" % f_ratio_els[0].sourceline
            )

        fcs_dim_els = f_ratio_els[0].findall(
            '%s:fcs-dimension' % data_type_namespace,
            namespaces=xform_element.nsmap
        )

        dim_labels = []

        for dim_el in fcs_dim_els:
            label = _gml_utils.find_attribute_value(dim_el, data_type_namespace, 'name')

            if label is None:
                raise ValueError(
                    'Dimension name not found (line %d)' % dim_el.sourceline
                )
            dim_labels.append(label)

        RatioTransform.__init__(
            self,
            t_id,
            dim_labels,
            float(param_a),
            float(param_b),
            float(param_c)
        )

    def apply(self, sample):
        events = RatioTransform.apply(self, sample)
        return events


class LinearGMLTransform(LinearTransform):
    def __init__(
            self,
            xform_element,
            xform_namespace
    ):
        t_id = _gml_utils.find_attribute_value(xform_element, xform_namespace, 'id')

        f_lin_els = xform_element.findall(
            '%s:flin' % xform_namespace,
            namespaces=xform_element.nsmap
        )

        if len(f_lin_els) == 0:
            raise ValueError(
                "Linear transform must specify an 'flin' element (line %d)" % xform_element.sourceline
            )

        # f linear transform has 2 parameters: T and A
        # these are attributes of the 'flin' element
        param_t = _gml_utils.find_attribute_value(f_lin_els[0], xform_namespace, 'T')
        param_a = _gml_utils.find_attribute_value(f_lin_els[0], xform_namespace, 'A')

        if None in [param_t, param_a]:
            raise ValueError(
                "Linear transform must provide 'T' and 'A' attributes (line %d)" % f_lin_els[0].sourceline
            )

        LinearTransform.__init__(
            self,
            t_id,
            float(param_t),
            float(param_a)
        )

    def apply(self, events):
        events = LinearTransform.apply(self, events)
        return events


class LogGMLTransform(LogTransform):
    def __init__(
            self,
            xform_element,
            xform_namespace
    ):
        t_id = _gml_utils.find_attribute_value(xform_element, xform_namespace, 'id')

        f_log_els = xform_element.findall(
            '%s:flog' % xform_namespace,
            namespaces=xform_element.nsmap
        )

        if len(f_log_els) == 0:
            raise ValueError(
                "Log transform must specify an 'flog' element (line %d)" % xform_element.sourceline
            )

        # f log transform has 2 parameters: T and M
        # these are attributes of the 'flog' element
        param_t = _gml_utils.find_attribute_value(f_log_els[0], xform_namespace, 'T')
        param_m = _gml_utils.find_attribute_value(f_log_els[0], xform_namespace, 'M')

        if None in [param_t, param_m]:
            raise ValueError(
                "Log transform must provide an 'T' attribute (line %d)" % f_log_els[0].sourceline
            )

        LogTransform.__init__(
            self,
            t_id,
            float(param_t),
            float(param_m)
        )

    def apply(self, events):
        events = LogTransform.apply(self, events)
        return events


class HyperlogGMLTransform(HyperlogTransform):
    def __init__(
            self,
            xform_element,
            xform_namespace
    ):
        t_id = _gml_utils.find_attribute_value(xform_element, xform_namespace, 'id')

        hyperlog_els = xform_element.findall(
            '%s:hyperlog' % xform_namespace,
            namespaces=xform_element.nsmap
        )

        if len(hyperlog_els) == 0:
            raise ValueError(
                "Hyperlog transform must specify an 'hyperlog' element (line %d)" % xform_element.sourceline
            )

        # hyperlog transform has 4 parameters: T, W, M, and A
        # these are attributes of the 'hyperlog' element
        param_t = _gml_utils.find_attribute_value(hyperlog_els[0], xform_namespace, 'T')
        param_w = _gml_utils.find_attribute_value(hyperlog_els[0], xform_namespace, 'W')
        param_m = _gml_utils.find_attribute_value(hyperlog_els[0], xform_namespace, 'M')
        param_a = _gml_utils.find_attribute_value(hyperlog_els[0], xform_namespace, 'A')

        if None in [param_t, param_w, param_m, param_a]:
            raise ValueError(
                "Hyperlog transform must provide 'T', 'W', 'M', and 'A' "
                "attributes (line %d)" % hyperlog_els[0].sourceline
            )

        HyperlogTransform.__init__(
            self,
            t_id,
            float(param_t),
            float(param_w),
            float(param_m),
            float(param_a)
        )

    def apply(self, events):
        events = HyperlogTransform.apply(self, events)
        return events


class LogicleGMLTransform(LogicleTransform):
    def __init__(
            self,
            xform_element,
            xform_namespace
    ):
        t_id = _gml_utils.find_attribute_value(xform_element, xform_namespace, 'id')

        logicle_els = xform_element.findall(
            '%s:logicle' % xform_namespace,
            namespaces=xform_element.nsmap
        )

        if len(logicle_els) == 0:
            raise ValueError(
                "Logicle transform must specify an 'logicle' element (line %d)" % xform_element.sourceline
            )

        # logicle transform has 4 parameters: T, W, M, and A
        # these are attributes of the 'logicle' element
        param_t = _gml_utils.find_attribute_value(logicle_els[0], xform_namespace, 'T')
        param_w = _gml_utils.find_attribute_value(logicle_els[0], xform_namespace, 'W')
        param_m = _gml_utils.find_attribute_value(logicle_els[0], xform_namespace, 'M')
        param_a = _gml_utils.find_attribute_value(logicle_els[0], xform_namespace, 'A')

        if None in [param_t, param_w, param_m, param_a]:
            raise ValueError(
                "Logicle transform must provide 'T', 'W', 'M', and 'A' "
                "attributes (line %d)" % logicle_els[0].sourceline
            )

        LogicleTransform.__init__(
            self,
            t_id,
            float(param_t),
            float(param_w),
            float(param_m),
            float(param_a)
        )

    def apply(self, events):
        events = LogicleTransform.apply(self, events)
        return events


class AsinhGMLTransform(AsinhTransform):
    def __init__(
            self,
            xform_element,
            xform_namespace
    ):
        t_id = _gml_utils.find_attribute_value(xform_element, xform_namespace, 'id')

        f_asinh_els = xform_element.findall(
            '%s:fasinh' % xform_namespace,
            namespaces=xform_element.nsmap
        )

        if len(f_asinh_els) == 0:
            raise ValueError(
                "Asinh transform must specify an 'fasinh' element (line %d)" % xform_element.sourceline
            )

        # f asinh transform has 3 parameters: T, M, and A
        # these are attributes of the 'fasinh' element
        param_t = _gml_utils.find_attribute_value(f_asinh_els[0], xform_namespace, 'T')
        param_m = _gml_utils.find_attribute_value(f_asinh_els[0], xform_namespace, 'M')
        param_a = _gml_utils.find_attribute_value(f_asinh_els[0], xform_namespace, 'A')

        if None in [param_t, param_m, param_a]:
            raise ValueError(
                "Asinh transform must provide 'T', 'M', and 'A' attributes (line %d)" % f_asinh_els[0].sourceline
            )

        AsinhTransform.__init__(
            self,
            t_id,
            float(param_t),
            float(param_m),
            float(param_a)
        )

    def apply(self, events):
        events = AsinhTransform.apply(self, events)
        return events
