"""
Basic Transform sub-classes
"""
import flowutils
from ._base_transform import Transform


class RatioTransform(Transform):
    """
    Parametrized ratio transformation, implemented as defined in the
    GatingML 2.0 specification:

    fratio(x, y, A, B, C) = A * ((x - B) / (y - C))

    Note: The RatioTransform does not have an inverse method.

    :param dim_ids: A list of length 2 specifying which dimension IDs to
        use for the ratio transformation. The 1st ID indicates the dimension
        to use for the numerator, the 2nd ID will be the dimension used for
        the denominator.
    :param param_a: parameter for scaling the ratio transform
    :param param_b: parameter for the numerator dimension offset
    :param param_c: parameter for the denominator dimension offset
    """
    def __init__(
            self,
            dim_ids,
            param_a,
            param_b,
            param_c
    ):
        Transform.__init__(self)

        if len(dim_ids) != 2:
            raise ValueError("RatioTransform takes exactly 2 dimension IDs but received %d" % len(dim_ids))

        self.dimensions = dim_ids

        self.param_a = param_a
        self.param_b = param_b
        self.param_c = param_c

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.dimensions[0]} / {self.dimensions[1]}, '
            f'a: {self.param_a}, b: {self.param_b}, c: {self.param_c})'
        )

    def apply(self, sample):
        """
        Apply RatioTransform to given 'raw' events in a Sample.

        The RatioTransform is different from other transform types in that
        it references 2 distinct channels of event data. Since the indices of those
        channels could vary across different FCS files, we need the Sample
        instance to introspect for the correct channels. All other Transform
        subclasses take an events array as the argument to the apply method.

        :param sample: Sample instance from which event data should be extracted
        :return: NumPy array of transformed events
        """
        events = sample.get_events(source='raw')

        dim_x_idx = sample.pnn_labels.index(self.dimensions[0])
        dim_y_idx = sample.pnn_labels.index(self.dimensions[1])
        dim_x = events[:, dim_x_idx]
        dim_y = events[:, dim_y_idx]

        new_events = self.param_a * ((dim_x - self.param_b) / (dim_y - self.param_c))

        return new_events


class LinearTransform(Transform):
    """
    Parametrized linear transformation, implemented as defined in the
    GatingML 2.0 specification:

    flin(x, T, A) = (x + A) / (T + A)

    This transformation linearly maps values from the interval [−A, T]
    to the interval [0, 1]. However, it is defined for all x ∈ R
    including outside the [−A, T] interval.

    :param param_t: parameter for the top of the linear scale (e.g. 262144)
    :param param_a: parameter for the offset, controls the bottom of the scale
    """
    def __init__(self, param_t, param_a):
        Transform.__init__(self)

        self.param_a = param_a
        self.param_t = param_t

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f't: {self.param_t}, a: {self.param_a})'
        )

    def apply(self, events):
        """
        Apply transform to given events.

        :param events: NumPy array of FCS event data
        :return: NumPy array of transformed events
        """
        new_events = (events + self.param_a) / (self.param_t + self.param_a)

        return new_events

    def inverse(self, events):
        """
        Apply the inverse transform to given events.

        :param events: NumPy array of FCS event data
        :return: NumPy array of inversely transformed events
        """
        # y = (x + a) / (t + a)
        # x = (y * (t + a)) - a
        new_events = (events * (self.param_t + self.param_a)) - self.param_a

        return new_events


class LogTransform(Transform):
    """
    Parametrized logarithmic transformation, implemented as defined in the
    GatingML 2.0 specification:

    flog(x, T, M) = (1 / M) * log_10(x / T) + 1

    This transformation provides a logarithmic display that maps scale values
    from the (0, T] interval to the (−∞, 1] interval such that the data value
    T is mapped to 1 and M decades of data are mapped onto the unit interval.

    :param param_t: parameter for the top of the linear scale (e.g. 262144)
    :param param_m: parameter for desired number of decades
    """
    def __init__(
        self,
        param_t,
        param_m
    ):
        Transform.__init__(self)

        self.param_m = param_m
        self.param_t = param_t

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f't: {self.param_t}, m: {self.param_m})'
        )

    def apply(self, events):
        """
        Apply transform to given events.

        :param events: NumPy array of FCS event data
        :return: NumPy array of transformed events
        """
        return flowutils.transforms.log(
            events,
            None,
            t=self.param_t,
            m=self.param_m
        )

    def inverse(self, events):
        """
        Apply the inverse transform to given events.

        :param events: NumPy array of FCS event data
        :return: NumPy array of inversely transformed events
        """
        return flowutils.transforms.log_inverse(
            events,
            None,
            t=self.param_t,
            m=self.param_m
        )


class HyperlogTransform(Transform):
    """
    Hyperlog transformation, implemented as defined in the
    GatingML 2.0 specification:

    hyperlog(x, T, W, M, A) = root(EH(y, T, W, M, A) − x)

    where EH is defined as:

    EH(y, T, W, M, A) = ae^(by) + cy − f

    The Hyperlog transformation was originally defined in the publication:

    Bagwell CB. Hyperlog-a flexible log-like transform for negative, zero, and
    positive valued data. Cytometry A., 2005:64(1):34–42.

    :param param_t: parameter for the top of the linear scale (e.g. 262144)
    :param param_m: parameter for desired number of decades
    :param param_w: parameter for the approximate number of decades in the linear region
    :param param_a: parameter for the additional number of negative decades
    """
    def __init__(
        self,
        param_t,
        param_w,
        param_m,
        param_a
    ):
        Transform.__init__(self)

        self.param_a = param_a
        self.param_m = param_m
        self.param_t = param_t
        self.param_w = param_w

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f't: {self.param_t}, w: {self.param_w}, '
            f'm: {self.param_m}, a: {self.param_a})'
        )

    def apply(self, events):
        """
        Apply transform to given events.

        :param events: NumPy array of FCS event data
        :return: NumPy array of transformed events
        """
        return flowutils.transforms.hyperlog(
            events,
            None,
            t=self.param_t,
            m=self.param_m,
            w=self.param_w,
            a=self.param_a
        )

    def inverse(self, events):
        """
        Apply the inverse transform to given events.

        :param events: NumPy array of FCS event data
        :return: NumPy array of inversely transformed events
        """
        return flowutils.transforms.hyperlog_inverse(
            events,
            None,
            t=self.param_t,
            m=self.param_m,
            w=self.param_w,
            a=self.param_a
        )


class LogicleTransform(Transform):
    """
    Logicle transformation, implemented as defined in the
    GatingML 2.0 specification:

    logicle(x, T, W, M, A) = root(B(y, T, W, M, A) − x)

    where B is a modified bi-exponential function defined as:

    B(y, T, W, M, A) = ae^(by) − ce^(−dy) − f

    The Logicle transformation was originally defined in the publication:

    Moore WA and Parks DR. Update for the logicle data scale including operational
    code implementations. Cytometry A., 2012:81A(4):273–277.

    The Logicle scale is the inverse of a modified biexponential function. It
    provides a Logicle display that maps scale values onto the [0, 1] interval
    such that the data value param_t is mapped to 1, large data values are mapped
    to locations similar to a logarithmic scale, and param_a decades of negative
    data are brought on scale. See the GatingML 2.0 specification for more details.

    :param param_t: parameter for the top of the linear scale (e.g. 262144)
    :param param_w: parameter for the approximate number of decades in the linear region
    :param param_m: parameter for the number of decades the true logarithmic scale
        approaches at the high end of the scale
    :param param_a: parameter for the additional number of negative decades
    """
    def __init__(
        self,
        param_t,
        param_w,
        param_m,
        param_a
    ):
        Transform.__init__(self)

        self.param_a = param_a
        self.param_m = param_m
        self.param_t = param_t
        self.param_w = param_w

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f't: {self.param_t}, w: {self.param_w}, '
            f'm: {self.param_m}, a: {self.param_a})'
        )

    def apply(self, events):
        """
        Apply transform to given events.

        :param events: NumPy array of FCS event data
        :return: NumPy array of transformed events
        """
        return flowutils.transforms.logicle(
            events,
            None,
            t=self.param_t,
            m=self.param_m,
            w=self.param_w,
            a=self.param_a
        )

    def inverse(self, events):
        """
        Apply the inverse transform to given events.

        :param events: NumPy array of FCS event data
        :return: NumPy array of inversely transformed events
        """
        return flowutils.transforms.logicle_inverse(
            events,
            None,
            t=self.param_t,
            m=self.param_m,
            w=self.param_w,
            a=self.param_a
        )


class AsinhTransform(Transform):
    """
    An implementation of the parametrized inverse hyperbolic sine function
    as defined in the GatingML 2.0 specification.

    This transformation provides an inverse hyperbolic sine transformation
    that maps a data value onto the interval [0,1] such that:
        * The top of scale value (i.e, param_t) is mapped to 1.
        * Large data values are mapped to locations similar to the logarithmic
          scale.
        * param_a decades of negative data are brought on scale.

    :param param_t: parameter specifying the top of the scale, (e.g. 262144)
    :param param_m: parameter for the number of decades
    :param param_a: parameter for the number of additional negative decades
    """
    def __init__(
        self,
        param_t,
        param_m,
        param_a
    ):
        Transform.__init__(self)

        self.param_a = param_a
        self.param_m = param_m
        self.param_t = param_t

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f't: {self.param_t}, m: {self.param_m}, a: {self.param_a})'
        )

    def apply(self, events):
        """
        Apply transform to given events.

        :param events: NumPy array of FCS event data
        :return: NumPy array of transformed events
        """
        return flowutils.transforms.asinh(
            events,
            None,
            t=self.param_t,
            m=self.param_m,
            a=self.param_a
        )

    def inverse(self, events):
        """
        Apply the inverse transform to given events.

        :param events: NumPy array of FCS event data
        :return: NumPy array of inversely transformed events
        """
        return flowutils.transforms.asinh_inverse(
            events,
            None,
            t=self.param_t,
            m=self.param_m,
            a=self.param_a
        )
