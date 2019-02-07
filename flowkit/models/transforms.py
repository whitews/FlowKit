from abc import ABC, abstractmethod
import numpy as np
import flowutils
from flowkit import utils


class Transform(ABC):
    def __init__(
            self,
            gating_strategy,
            transform_id
    ):
        self.__parent__ = gating_strategy
        self.id = transform_id
        self.dimensions = []

    @abstractmethod
    def apply(self, sample):
        pass


class GMLTransform(Transform):
    def __init__(self, xform_element, xform_namespace, gating_strategy):
        t_id = xform_element.xpath(
            '@%s:id' % xform_namespace,
            namespaces=xform_element.nsmap
        )[0]
        Transform.__init__(self, gating_strategy, t_id)

    @abstractmethod
    def apply(self, sample):
        pass


class RatioTransform(Transform):
    def __init__(
            self,
            gating_strategy,
            transform_id,
            dim_labels,
            param_a,
            param_b,
            param_c
    ):
        Transform.__init__(self, gating_strategy, transform_id)

        self.dimensions = dim_labels

        self.param_a = param_a
        self.param_b = param_b
        self.param_c = param_c

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f't: {self.param_a}, w: {self.param_b}, c: {self.param_c})'
        )

    def apply(self, sample):
        events = sample.get_raw_events()
        events = events.copy()

        dim_x_idx = sample.pnn_labels.index(self.dimensions[0])
        dim_y_idx = sample.pnn_labels.index(self.dimensions[1])
        dim_x = events[:, dim_x_idx]
        dim_y = events[:, dim_y_idx]

        new_events = self.param_a * ((dim_x - self.param_b) / (dim_y - self.param_c))

        return new_events


class RatioGMLTransform(GMLTransform, RatioTransform):
    def __init__(
            self,
            xform_element,
            xform_namespace,
            data_type_namespace,
            gating_strategy
    ):
        GMLTransform.__init__(
            self,
            xform_element,
            xform_namespace,
            gating_strategy
        )

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
        param_a_attribs = f_ratio_els[0].xpath(
            '@%s:A' % xform_namespace,
            namespaces=xform_element.nsmap
        )
        param_b_attribs = f_ratio_els[0].xpath(
            '@%s:B' % xform_namespace,
            namespaces=xform_element.nsmap
        )
        param_c_attribs = f_ratio_els[0].xpath(
            '@%s:C' % xform_namespace,
            namespaces=xform_element.nsmap
        )
        if len(param_a_attribs) == 0 or len(param_b_attribs) == 0 or len(param_c_attribs) == 0:
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
            label_attribs = dim_el.xpath(
                '@%s:name' % data_type_namespace,
                namespaces=xform_element.nsmap
            )

            if len(label_attribs) > 0:
                label = label_attribs[0]
            else:
                raise ValueError(
                    'Dimension name not found (line %d)' % dim_el.sourceline
                )
            dim_labels.append(label)

        RatioTransform.__init__(
            self,
            gating_strategy,
            self.id,
            dim_labels,
            float(param_a_attribs[0]),
            float(param_b_attribs[0]),
            float(param_c_attribs[0])
        )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.id}, t: {self.param_a}, w: {self.param_b}, c: {self.param_c})'
        )

    def apply(self, sample):
        events = RatioTransform.apply(self, sample)
        return events


class LinearTransform(Transform):
    def __init__(
            self,
            gating_strategy,
            transform_id,
            param_t,
            param_a
    ):
        Transform.__init__(self, gating_strategy, transform_id)

        self.param_a = param_a
        self.param_t = param_t

    def apply(self, events):
        new_events = (events.copy() + self.param_a) / (self.param_t + self.param_a)

        return new_events


class LinearGMLTransform(GMLTransform, LinearTransform):
    def __init__(
            self,
            xform_element,
            xform_namespace,
            gating_strategy
    ):
        GMLTransform.__init__(
            self,
            xform_element,
            xform_namespace,
            gating_strategy
        )

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
        param_t_attribs = f_lin_els[0].xpath(
            '@%s:T' % xform_namespace,
            namespaces=xform_element.nsmap
        )
        param_a_attribs = f_lin_els[0].xpath(
            '@%s:A' % xform_namespace,
            namespaces=xform_element.nsmap
        )

        if len(param_t_attribs) == 0 or len(param_a_attribs) == 0:
            raise ValueError(
                "Linear transform must provide 'T' and 'A' attributes (line %d)" % f_lin_els[0].sourceline
            )

        LinearTransform.__init__(
            self,
            gating_strategy,
            self.id,
            float(param_t_attribs[0]),
            float(param_a_attribs[0])
        )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.id}, t: {self.param_t}, a: {self.param_a})'
        )

    def apply(self, sample):
        events = LinearTransform.apply(self, sample)
        return events


class LogTransform(Transform):
    def __init__(
        self,
        gating_strategy,
        transform_id,
        param_t,
        param_m
    ):
        Transform.__init__(self, gating_strategy, transform_id)

        self.param_m = param_m
        self.param_t = param_t

    def apply(self, events):
        new_events = (1. / self.param_m) * np.log10(events.copy() / self.param_t) + 1.

        return new_events


class LogGMLTransform(GMLTransform, LogTransform):
    def __init__(
            self,
            xform_element,
            xform_namespace,
            gating_strategy
    ):
        GMLTransform.__init__(
            self,
            xform_element,
            xform_namespace,
            gating_strategy
        )

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
        param_t_attribs = f_log_els[0].xpath(
            '@%s:T' % xform_namespace,
            namespaces=xform_element.nsmap
        )
        param_m_attribs = f_log_els[0].xpath(
            '@%s:M' % xform_namespace,
            namespaces=xform_element.nsmap
        )

        if len(param_t_attribs) == 0 or len(param_m_attribs) == 0:
            raise ValueError(
                "Log transform must provide an 'T' attribute (line %d)" % f_log_els[0].sourceline
            )

        LogTransform.__init__(
            self,
            gating_strategy,
            self.id,
            float(param_t_attribs[0]),
            float(param_m_attribs[0])
        )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.id}, t: {self.param_t}, m: {self.param_m})'
        )

    def apply(self, sample):
        events = LogTransform.apply(self, sample)
        return events


class HyperlogTransform(Transform):
    def __init__(
        self,
        gating_strategy,
        transform_id,
        param_t,
        param_w,
        param_m,
        param_a
    ):
        Transform.__init__(self, gating_strategy, transform_id)

        self.param_a = param_a
        self.param_m = param_m
        self.param_t = param_t
        self.param_w = param_w

    def apply(self, events):
        hyperlog = utils.Hyperlog(
            self.param_t,
            self.param_w,
            self.param_m,
            self.param_a
        )

        new_events = []

        for e in events.copy():
            new_events.append(hyperlog.scale(e))

        return np.array(new_events)


class HyperlogGMLTransform(GMLTransform, HyperlogTransform):
    def __init__(
            self,
            xform_element,
            xform_namespace,
            gating_strategy
    ):
        GMLTransform.__init__(
            self,
            xform_element,
            xform_namespace,
            gating_strategy
        )

        hlog_els = xform_element.findall(
            '%s:hyperlog' % xform_namespace,
            namespaces=xform_element.nsmap
        )

        if len(hlog_els) == 0:
            raise ValueError(
                "Hyperlog transform must specify an 'hyperlog' element (line %d)" % xform_element.sourceline
            )

        # hyperlog transform has 4 parameters: T, W, M, and A
        # these are attributes of the 'hyperlog' element
        param_t_attribs = hlog_els[0].xpath(
            '@%s:T' % xform_namespace,
            namespaces=xform_element.nsmap
        )
        param_w_attribs = hlog_els[0].xpath(
            '@%s:W' % xform_namespace,
            namespaces=xform_element.nsmap
        )
        param_m_attribs = hlog_els[0].xpath(
            '@%s:M' % xform_namespace,
            namespaces=xform_element.nsmap
        )
        param_a_attribs = hlog_els[0].xpath(
            '@%s:A' % xform_namespace,
            namespaces=xform_element.nsmap
        )

        if len(param_t_attribs) == 0 or len(param_w_attribs) == 0 or \
                len(param_m_attribs) == 0 or len(param_a_attribs) == 0:
            raise ValueError(
                "Hyperlog transform must provide 'T', 'W', 'M', and 'A' "
                "attributes (line %d)" % hlog_els[0].sourceline
            )

        HyperlogTransform.__init__(
            self,
            gating_strategy,
            self.id,
            float(param_t_attribs[0]),
            float(param_w_attribs[0]),
            float(param_m_attribs[0]),
            float(param_a_attribs[0])
        )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.id}, t: {self.param_t}, w: {self.param_w},'
            f'm: {self.param_m}, a: {self.param_a})'
        )

    def apply(self, sample):
        events = HyperlogTransform.apply(self, sample)
        return events


class LogicleGMLTransform(GMLTransform):
    def __init__(
            self,
            xform_element,
            xform_namespace,
            gating_strategy
    ):
        GMLTransform.__init__(
            self,
            xform_element,
            xform_namespace,
            gating_strategy
        )

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
        param_t_attribs = logicle_els[0].xpath(
            '@%s:T' % xform_namespace,
            namespaces=xform_element.nsmap
        )
        param_w_attribs = logicle_els[0].xpath(
            '@%s:W' % xform_namespace,
            namespaces=xform_element.nsmap
        )
        param_m_attribs = logicle_els[0].xpath(
            '@%s:M' % xform_namespace,
            namespaces=xform_element.nsmap
        )
        param_a_attribs = logicle_els[0].xpath(
            '@%s:A' % xform_namespace,
            namespaces=xform_element.nsmap
        )

        if len(param_t_attribs) == 0 or len(param_w_attribs) == 0 or \
                len(param_m_attribs) == 0 or len(param_a_attribs) == 0:
            raise ValueError(
                "Logicle transform must provide 'T', 'W', 'M', and 'A' "
                "attributes (line %d)" % logicle_els[0].sourceline
            )

        self.param_t = float(param_t_attribs[0])
        self.param_w = float(param_w_attribs[0])
        self.param_m = float(param_m_attribs[0])
        self.param_a = float(param_a_attribs[0])

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.id}, t: {self.param_t}, w: {self.param_w},'
            f'm: {self.param_m}, a: {self.param_a})'
        )

    def apply(self, events):
        reshape = False

        if len(events.shape) == 1:
            events = events.copy().reshape(-1, 1)
            reshape = True

        new_events = flowutils.transforms.logicle(
            events,
            range(events.shape[1]),
            t=self.param_t,
            m=self.param_m,
            w=self.param_w,
            a=self.param_a
        )

        if reshape:
            new_events = new_events.reshape(-1)

        return new_events


class AsinhGMLTransform(GMLTransform):
    def __init__(
            self,
            xform_element,
            xform_namespace,
            gating_strategy
    ):
        GMLTransform.__init__(
            self,
            xform_element,
            xform_namespace,
            gating_strategy
        )

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
        param_t_attribs = f_asinh_els[0].xpath(
            '@%s:T' % xform_namespace,
            namespaces=xform_element.nsmap
        )
        param_m_attribs = f_asinh_els[0].xpath(
            '@%s:M' % xform_namespace,
            namespaces=xform_element.nsmap
        )
        param_a_attribs = f_asinh_els[0].xpath(
            '@%s:A' % xform_namespace,
            namespaces=xform_element.nsmap
        )

        if len(param_t_attribs) == 0 or len(param_m_attribs) == 0 or len(param_a_attribs) == 0:
            raise ValueError(
                "Asinh transform must provide 'T', 'M', and 'A' attributes (line %d)" % f_asinh_els[0].sourceline
            )

        self.param_t = float(param_t_attribs[0])
        self.param_m = float(param_m_attribs[0])
        self.param_a = float(param_a_attribs[0])

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.id}, t: {self.param_t}, m: {self.param_m}, a: {self.param_a})'
        )

    def apply(self, events):
        x_pre_scale = np.sinh(self.param_m * np.log(10)) / self.param_t
        x_transpose = self.param_a * np.log(10)
        x_divisor = (self.param_m + self.param_a) * np.log(10)

        new_events = (np.arcsinh(events.copy() * x_pre_scale) + x_transpose) / x_divisor

        return new_events
