"""
Utility functions related to transforming data.
"""
import copy

from .._models import transforms


def _create_transform(transform_class, top_range):
    min_range = 0.0  # a.k.a. param_a
    decades_default = 4.0
    decades_asinh = 3.5
    decades_biex = 4.418540
    lin_decades = 0.5  # a.k.a. param_w
    biex_width = -10

    if transform_class == transforms.LinearTransform:
        xform = transforms.LinearTransform(param_t=top_range, param_a=min_range)
    elif transform_class == transforms.LogTransform:
        xform = transforms.LogTransform(param_t=top_range, param_m=decades_default)
    elif transform_class == transforms.HyperlogTransform:
        xform = transforms.HyperlogTransform(
            param_t=top_range, param_m=decades_default, param_w=lin_decades, param_a=min_range
        )
    elif transform_class == transforms.LogicleTransform:
        xform = transforms.LogicleTransform(
            param_t=top_range, param_m=decades_default, param_w=lin_decades, param_a=min_range
        )
    elif transform_class == transforms.AsinhTransform:
        xform = transforms.AsinhTransform(
            param_t=top_range, param_m=decades_asinh, param_a=min_range
        )
    elif transform_class == transforms.WSPBiexTransform:
        xform = transforms.WSPBiexTransform(
            max_value=top_range, positive=decades_biex, width=biex_width, negative=min_range
        )
    else:
        raise NotImplementedError("Auto-generating %s instances is not yet supported." % transform_class.__name__)

    return xform


def generate_transforms(
        sample,
        scatter_xform_class=transforms.LinearTransform,
        fluoro_xform_class=transforms.LogicleTransform,
        time_xform_class=transforms.LinearTransform
):
    """
    Generate a dictionary of transforms for channels in a given Sample
    instance.

    A set of Transforms is generated for each type of channel: scatter
    channels, fluorescent channels, and the time channel. If given a class
    reference to a Transform, then a default set of parameters will be chosen.

    For any transform, the top of scale parameter (e.g. `T` in Logicle) will
    be determined by the channel's PnR value within the given Sample. The
    exception to is the Time channel, where the top of scale will be
    calculated as the max Time event value.

    The available Transforms generated from a class reference are listed below
    along with their default parameter values for the Transform subclasses are:

    - LinearTransform(t: PnR_value, a: 0.0)
    - LogTransform(t: PnR_value, m: 4.0)
    - LogicleTransform(t: PnR_value, w: 0.5, m: 4.0, a: 0.0)
    - AsinhTransform(t: PnR_value, m: 3.5, a: 0.0)
    - HyperlogTransform(t: PnR_value, w: 0.5, m: 4.0, a: 0.0)
    - WSPBiexTransform(width: -10, neg: 0.0, pos: 4.41854, top: PnR_value)

    NOTE: Each channel will have independent transform instances, not references
    to the same transform instance.

    :param sample: Sample instance
    :param scatter_xform_class: Transform subclass reference to use for scatter channels
        or a specific instance of a Transform subclass.
    :param fluoro_xform_class: Transform subclass reference to use for fluorescent channels
    :param time_xform_class: Transform subclass reference to use for time channel
    :return: dictionary with PnN labels as keys, values are Transform instances
    """
    # Check if given transform class is any kind of Transform instance.
    # If so, we can use the instance directly. If not, assume it is a
    # Transform class reference, and we'll create the instance later
    # per channel.
    scatter_xform_instance = None
    fluoro_xform_instance = None
    time_xform_instance = None

    # Start with scatter
    # noinspection PyProtectedMember
    if isinstance(scatter_xform_class, transforms._transforms.Transform):
        scatter_xform_instance = scatter_xform_class

    # Same for the fluorescent transform
    # noinspection PyProtectedMember
    if isinstance(fluoro_xform_class, transforms._transforms.Transform):
        fluoro_xform_instance = fluoro_xform_class

    # Same for the time transform
    # noinspection PyProtectedMember
    if isinstance(time_xform_class, transforms._transforms.Transform):
        time_xform_instance = time_xform_class

    xform_lut = {}  # keys are PnN, values are Transform instance

    for _, channel in sample.channels.iterrows():
        chan_idx = sample.get_channel_index(channel['channel_number'])
        chan_xform = None

        if chan_idx in sample.scatter_indices:
            if scatter_xform_instance is None:
                chan_xform = _create_transform(
                    scatter_xform_class,
                    channel['pnr']
                )
            else:
                chan_xform = copy.deepcopy(scatter_xform_instance)
        elif chan_idx in sample.fluoro_indices:
            if fluoro_xform_instance is None:
                chan_xform = _create_transform(
                    fluoro_xform_class,
                    channel['pnr']
                )
            else:
                chan_xform = copy.deepcopy(fluoro_xform_instance)
        elif chan_idx == sample.time_index:
            if time_xform_instance is None:
                # for time, the top of range is max time
                # Use the channel index to get the label to retrieve channel events
                max_time = sample.get_channel_events(channel['pnn'], source='raw').max()
                chan_xform = _create_transform(
                    time_xform_class,
                    max_time
                )
            else:
                chan_xform = copy.deepcopy(time_xform_instance)

        if chan_xform is None:
            # skip this channel
            continue

        xform_lut[channel['pnn']] = chan_xform

    return xform_lut
