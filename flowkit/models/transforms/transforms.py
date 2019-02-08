import numpy as np
import flowutils
from flowkit import utils
from.base_transform import Transform

__all__ = [
    'LinearTransform',
    'LogTransform',
    'RatioTransform',
    'HyperlogTransform',
    'LogicleTransform',
    'AsinhTransform'
]


class RatioTransform(Transform):
    def __init__(
            self,
            transform_id,
            dim_labels,
            param_a,
            param_b,
            param_c
    ):
        Transform.__init__(self, transform_id)

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


class LinearTransform(Transform):
    def __init__(
            self,
            transform_id,
            param_t,
            param_a
    ):
        Transform.__init__(self, transform_id)

        self.param_a = param_a
        self.param_t = param_t

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.id}, t: {self.param_t}, a: {self.param_a})'
        )

    def apply(self, events):
        new_events = (events.copy() + self.param_a) / (self.param_t + self.param_a)

        return new_events


class LogTransform(Transform):
    def __init__(
        self,
        transform_id,
        param_t,
        param_m
    ):
        Transform.__init__(self, transform_id)

        self.param_m = param_m
        self.param_t = param_t

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.id}, t: {self.param_t}, m: {self.param_m})'
        )

    def apply(self, events):
        new_events = (1. / self.param_m) * np.log10(events.copy() / self.param_t) + 1.

        return new_events


class HyperlogTransform(Transform):
    def __init__(
        self,
        transform_id,
        param_t,
        param_w,
        param_m,
        param_a
    ):
        Transform.__init__(self, transform_id)

        self.param_a = param_a
        self.param_m = param_m
        self.param_t = param_t
        self.param_w = param_w

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.id}, t: {self.param_t}, w: {self.param_w},'
            f'm: {self.param_m}, a: {self.param_a})'
        )

    def apply(self, events):
        hyperlog = utils.Hyperlog(
            self.param_t,
            self.param_w,
            self.param_m,
            self.param_a
        )

        new_events = []

        # TODO: This is slow, is there a way to vectorize the scale method?
        for e in events.copy():
            if len(events.shape) > 1:
                new_row_events = []
                for row_e in e:
                    new_row_events.append(hyperlog.scale(row_e))

                new_events.append(new_row_events)
            else:
                new_events.append(hyperlog.scale(e))

        return np.array(new_events)


class LogicleTransform(Transform):
    def __init__(
        self,
        transform_id,
        param_t,
        param_w,
        param_m,
        param_a
    ):
        Transform.__init__(self, transform_id)

        self.param_a = param_a
        self.param_m = param_m
        self.param_t = param_t
        self.param_w = param_w

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


class AsinhTransform(Transform):
    def __init__(
        self,
        transform_id,
        param_t,
        param_m,
        param_a
    ):
        Transform.__init__(self, transform_id)

        self.param_a = param_a
        self.param_m = param_m
        self.param_t = param_t

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
