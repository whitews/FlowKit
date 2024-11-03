"""
Transform classes compatible with FlowJo 10
"""
import numpy as np
from scipy import interpolate
from ._base_transform import Transform


def _log_root(b, w):
    x_lo = 0
    x_hi = b
    d = (x_lo + x_hi) / 2
    dx = abs(int(x_lo - x_hi))
    dx_last = dx
    fb = -2 * np.log(b) + w * b
    f = 2. * np.log(d) + w * b + fb
    df = 2 / d + w

    if w == 0:
        return b

    for i in range(100):
        if (((d - x_hi) * df - f) - ((d - x_lo) * df - f)) > 0 or abs(2 * f) > abs(dx_last * df):
            dx = (x_hi - x_lo) / 2
            d = x_lo + dx
            if d == x_lo:
                return d
        else:
            dx = f / df
            t = d
            d -= dx
            if d == t:
                return d

        # if abs(int(dx)) < 1.0E-12:
        if abs(dx) < 1.0E-12:
            return d

        dx_last = dx
        f = 2 * np.log(d) + w * d + fb
        df = 2 / d + w
        if f < 0:
            x_lo = d
        else:
            x_hi = d

    return d


def generate_biex_lut(channel_range=4096, pos=4.418540, neg=0.0, width_basis=-10, max_value=262144.000029):
    """
    Creates a FlowJo compatible biex lookup table.

    Implementation ported from the R library cytolib, which claims to be directly ported from the
    legacy Java code from TreeStar.

    :param channel_range: Maximum positive value of the output range
    :param pos: Number of decades
    :param neg: Number of extra negative decades
    :param width_basis: Controls the amount of input range compressed in the zero / linear region. A higher
        width basis value will include more input values in the zero / linear region.
    :param max_value: maximum input value to scale
    :return: 2-column NumPy array of the LUT (column order: input, output)
    """
    ln10 = np.log(10.0)
    decades = pos
    low_scale = width_basis
    width = np.log10(-low_scale)

    decades = decades - (width / 2)

    extra = neg

    if extra < 0:
        extra = 0

    extra = extra + (width / 2)

    zero_point = int((extra * channel_range) / (extra + decades))
    zero_point = int(np.min([zero_point, channel_range / 2]))

    if zero_point > 0:
        decades = extra * channel_range / zero_point

    width = width / (2 * decades)

    maximum = max_value
    positive_range = ln10 * decades
    minimum = maximum / np.exp(positive_range)

    negative_range = _log_root(positive_range, width)

    max_channel_value = channel_range + 1
    n_points = max_channel_value

    step = (max_channel_value - 1) / (n_points - 1)

    values = np.arange(n_points)
    positive = np.exp(values / float(n_points) * positive_range)
    negative = np.exp(values / float(n_points) * -negative_range)

    # apply step to values
    values = values * step

    s = np.exp((positive_range + negative_range) * (width + extra / decades))

    negative *= s
    s = positive[zero_point] - negative[zero_point]

    positive[zero_point:n_points] = positive[zero_point:n_points] - negative[zero_point:n_points]
    positive[zero_point:n_points] = minimum * (positive[zero_point:n_points] - s)

    neg_range = np.arange(zero_point)
    m = 2 * zero_point - neg_range

    positive[neg_range] = -positive[m]

    return positive, values


class WSPLogTransform(Transform):
    """
    Logarithmic transform as implemented in FlowJo 10.

    :param offset: A positive number used to offset event data
    :param decades: A positive number of for the desired number of decades
    """
    def __init__(self, offset, decades):
        Transform.__init__(self)

        self.offset = offset
        self.decades = decades

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'offset: {self.offset}, decades: {self.decades})'
        )

    def apply(self, events):
        """
        Apply transform to given events.

        :param events: NumPy array of FCS event data
        :return: NumPy array of transformed events
        """
        new_events = np.copy(events)
        new_events[new_events < self.offset] = self.offset
        new_events = 1. / self.decades * (np.log10(new_events) - np.log10(self.offset))

        return new_events


class WSPBiexTransform(Transform):
    """
    Biex transform as implemented in FlowJo 10. This transform is applied exactly as
    the FlowJo 10 is implemented, using lookup tables with only a limited set
    of parameter values. See the supported values in the BIEX_NEG_VALUES and
    BIEX_WIDTH_VALUES arrays within the `transforms` module.

    Information on the input parameters from the FlowJo docs:

    Adjusting width:
    The value for w will determine the amount of data to be compressed into
    linear space around zero. The space of linear does not change, but rather the
    number of channels or bins being compressed into the linear space.

    Width should be set high enough that all the data in the histogram is visible
    on screen, but not so high that extra white space is seen to the left-hand side
    of your dimmest distribution. For most practical uses, once all events have been
    shifted off the axis and there is no more axis ‘pile-up’, then the optimal width
    basis value has been reached.

    Negative:
    Another component in the biexponential transform calculation is the negative
    decades or negative space. This is the only other value you will probably ever
    need to adjust. In cases where a high width basis may start compressing dim events
    into the negative cluster, you may want to lower the width basis (less compression
    around zero) and instead, increase the negative space by 0.5 – 1.0. Doing this
    will expand the space around zero so the dim events are still visible, but also
    expand the negative space to remove the cells from the axis and allow you to see
    the full distribution.

    Positive:
    The presence of the positive decade adjustment is due to the algorithm used for
    logicle transformation, but is not useful in 99.9% of the cases that require
    adjusting the biexponential transform. It may be appropriate to adjust this value
    only if you use data that displays data with a data range greater than 5 decades.

    :param negative: Value for the FlowJo biex option 'negative' (float)
    :param width: Value for the FlowJo biex option 'width' (float)
    :param positive: Value for the FlowJo biex option 'positive' (float)
    :param max_value: parameter for the top of the linear scale (default=262144.000029)
    """
    def __init__(
        self,
        negative=0,
        width=-10,
        positive=4.418540,
        max_value=262144.000029
    ):
        Transform.__init__(self)

        self.negative = negative
        self.width = width
        self.positive = positive
        self.max_value = max_value

        x, y = generate_biex_lut(neg=self.negative, width_basis=self.width, pos=positive, max_value=max_value)

        # create interpolation function with any values outside the range set to the min / max of LUT
        self._lut_func = interpolate.interp1d(
            x, y, kind='linear', bounds_error=False, fill_value=(np.min(y), np.max(y))
        )
        self._inverse_lut_func = interpolate.interp1d(
            y, x, kind='linear', bounds_error=False, fill_value=(np.min(x), np.max(x))
        )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'width: {self.width}, neg: {self.negative}, pos: {self.positive}, top: {self.max_value})'
        )

    def apply(self, events):
        """
        Apply transform to given events.

        :param events: NumPy array of FCS event data
        :return: NumPy array of transformed events
        """
        new_events = self._lut_func(events)

        return new_events

    def inverse(self, events):
        """
        Apply the inverse transform to given events.

        :param events: NumPy array of FCS event data
        :return: NumPy array of inversely transformed events
        """
        new_events = self._inverse_lut_func(events)

        return new_events
