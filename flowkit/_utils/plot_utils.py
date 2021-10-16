"""
Utility functions related to plotting
"""
import numpy as np
from scipy.interpolate import interpn
import colorsys
from matplotlib import cm, colors
from bokeh.plotting import figure
from bokeh.models import Ellipse, Patch, Span, BoxAnnotation, Rect, ColumnDataSource


line_color = "#1F77B4"
line_color_contrast = "#73D587"
line_width = 3
fill_color = 'lime'
fill_alpha = 0.08


def _generate_custom_colormap(colormap_sample_indices, base_colormap):
    x = np.linspace(0, np.pi, base_colormap.N)
    new_lum = (np.sin(x) * 0.75) + .25

    new_color_list = []

    for i in colormap_sample_indices:
        (r, g, b, a) = base_colormap(i)
        (h, s, v) = colorsys.rgb_to_hsv(r, g, b)

        mod_v = (v * ((196 - abs(i - 196)) / 196) + new_lum[i]) / 2.

        new_r, new_g, new_b = colorsys.hsv_to_rgb(h, 1., mod_v)
        (_, new_l, _) = colorsys.rgb_to_hls(new_r, new_g, new_b)

        new_color_list.append((new_r, new_g, new_b))

    return colors.LinearSegmentedColormap.from_list(
        'custom_' + base_colormap.name,
        new_color_list,
        256
    )


cm_sample = [
    0, 8, 16, 24, 32, 40, 48, 52, 60, 64, 72, 80, 92,
    100, 108, 116, 124, 132,
    139, 147, 155, 159,
    163, 167, 171, 175, 179, 183, 187, 191, 195, 199, 215, 231, 239
]

new_jet = _generate_custom_colormap(cm_sample, cm.get_cmap('jet'))


def _get_false_bounds(bool_array):
    diff = np.diff(np.hstack((0, bool_array, 0)))

    start = np.where(diff == 1)
    end = np.where(diff == -1)

    return start[0], end[0]


def plot_channel(channel_events, label, subplot_ax, xform=None, flagged_events=None):
    """
    Plots a single-channel of FCS event data with the x-axis as the event number (similar to having
    time on the x-axis, but events are equally spaced). This function takes a Matplotlib Axes object
    to enable embedding multiple channel plots within the same figure (created outside this function).

    :param channel_events: 1-D NumPy array of event data
    :param label: string to use as the plot title
    :param subplot_ax: Matplotlib Axes instance used to render the plot
    :param xform: an optional Transform instance used to transform the given event data. channel_events can
        be given already pre-processed (compensated and/or transformed), in this case set xform to None.
    :param flagged_events: optional Boolean array of "flagged" events, regions of flagged events will
        be highlighted in red if flagged_events is given.
    :return: None
    """
    if xform:
        channel_events = xform.apply(channel_events)

    bins = int(np.sqrt(channel_events.shape[0]))
    event_range = range(0, channel_events.shape[0])

    subplot_ax.set_title(label, fontsize=16)
    subplot_ax.set_xlabel("Events", fontsize=14)

    subplot_ax.hist2d(
        event_range,
        channel_events,
        bins=[bins, 128],
        cmap='rainbow',
        cmin=1
    )

    if flagged_events is not None:
        starts, ends = _get_false_bounds(flagged_events)

        for i, s in enumerate(starts):
            subplot_ax.axvspan(
                event_range[s],
                event_range[ends[i] - 1],
                facecolor='pink',
                alpha=0.3,
                edgecolor='deeppink'
            )


def _calculate_extent(data_1d, d_min=None, d_max=None, pad=0.0):
    data_min = data_1d.min()
    data_max = data_1d.max()

    # determine padding to keep min/max events off the edge
    pad_d = max(abs(data_1d.min()), abs(data_1d.max())) * pad

    if d_min is None:
        d_min = data_min - pad_d
    if d_max is None:
        d_max = data_max + pad_d

    return d_min, d_max


def render_polygon(vertices):
    """
    Renders a Bokeh polygon for plotting
    :param vertices: list of 2-D coordinates representing vertices of the polygon
    :return: tuple containing the Bokeh ColumnDataSource and polygon glyphs (as Patch object)
    """
    x_coords, y_coords = list(zip(*[v.coordinates for v in vertices]))

    source = ColumnDataSource(dict(x=x_coords, y=y_coords))

    poly = Patch(
        x='x',
        y='y',
        fill_color=fill_color,
        fill_alpha=fill_alpha,
        line_width=line_width,
        line_color=line_color_contrast
    )

    return source, poly


def render_ranges(dim_minimums, dim_maximums):
    """
    Renders Bokeh Span & BoxAnnotation objects for plotting simple range gates, essentially divider lines.
    There should be no more than 3 items total between dim_minimums & dim_maximums, else the object should
    be rendered as a rectangle.

    :param dim_minimums: list of minimum divider values (max of 2)
    :param dim_maximums: list of maximum divider values (max of 2)
    :return: tuple of Span objects for every item in dim_minimums & dim_maximums
    """
    renderers = []
    left = None
    right = None
    bottom = None
    top = None

    if dim_minimums[0] is not None:
        left = dim_minimums[0]
        renderers.append(
            Span(location=left, dimension='height', line_width=line_width, line_color=line_color)
        )
    if dim_maximums[0] is not None:
        right = dim_maximums[0]
        renderers.append(
            Span(location=right, dimension='height', line_width=line_width, line_color=line_color)
        )
    if len(dim_minimums) > 1:
        if dim_minimums[1] is not None:
            bottom = dim_minimums[1]
            renderers.append(
                Span(location=bottom, dimension='width', line_width=line_width, line_color=line_color)
            )
        if dim_maximums[1] is not None:
            top = dim_maximums[1]
            renderers.append(
                Span(location=top, dimension='width', line_width=line_width, line_color=line_color)
            )

    mid_box = BoxAnnotation(
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        fill_alpha=fill_alpha,
        fill_color=fill_color
    )
    renderers.append(mid_box)

    return renderers


def render_rectangle(dim_minimums, dim_maximums):
    """
    Renders Bokeh Rect object for plotting a rectangle gate.

    :param dim_minimums: list of 2 values representing the lower left corner of a rectangle
    :param dim_maximums: list of 2 values representing the upper right corner of a rectangle
    :return: Bokeh Rect object
    """
    x_center = (dim_minimums[0] + dim_maximums[0]) / 2.0
    y_center = (dim_minimums[1] + dim_maximums[1]) / 2.0
    x_width = dim_maximums[0] - dim_minimums[0]
    y_height = dim_maximums[1] - dim_minimums[1]
    rect = Rect(
        x=x_center,
        y=y_center,
        width=x_width,
        height=y_height,
        fill_color=fill_color,
        fill_alpha=fill_alpha,
        line_width=line_width
    )

    return rect


def render_dividers(x_locs, y_locs):
    """
    Renders lines for divider boundaries (2-D only)
    :param x_locs: list of divider locations in x-axis
    :param y_locs: list of divider locations in y-axis
    :return: list of Bokeh renderer objects
    """
    renderers = []

    for x_loc in x_locs:
        renderers.append(
            Span(location=x_loc, dimension='height', line_width=line_width, line_color=line_color)
        )
    for y_loc in y_locs:
        renderers.append(
            Span(location=y_loc, dimension='width', line_width=line_width, line_color=line_color)
        )

    return renderers


def render_ellipse(center_x, center_y, covariance_matrix, distance_square):
    """
    Renders a Bokeh Ellipse object given the ellipse center point, covariance, and distance square

    :param center_x: x-coordinate of ellipse center
    :param center_y: y-coordinate of ellipse center
    :param covariance_matrix: NumPy array containing the covariance matrix of the ellipse
    :param distance_square: value for distance square of ellipse
    :return: Bokeh Ellipse object
    """
    values, vectors = np.linalg.eigh(covariance_matrix)
    order = values.argsort()[::-1]
    values = values[order]
    vectors = vectors[:, order]

    angle_rads = np.arctan2(*vectors[:, 0][::-1])

    # Width and height are full width (the axes lengths are thus multiplied by 2.0 here)
    width, height = 2.0 * np.sqrt(values * distance_square)

    ellipse = Ellipse(
        x=center_x,
        y=center_y,
        width=width,
        height=height,
        angle=angle_rads,
        line_width=line_width,
        line_color=line_color,
        fill_color=fill_color,
        fill_alpha=fill_alpha
    )

    return ellipse


def plot_histogram(x, x_label='x', bins=None):
    """
    Creates a Bokeh histogram plot of the given 1-D data array.

    :param x: 1-D array of data values
    :param x_label: Label to use for the x-axis
    :param bins: Number of bins to use for the histogram or a string compatible
            with the NumPy histogram function. If None, the number of bins is
            determined by the square root rule.
    :return: Bokeh Figure object containing the histogram
    """
    if bins is None:
        bins = 'sqrt'

    hist, edges = np.histogram(x, density=False, bins=bins)

    tools = "crosshair,hover,pan,zoom_in,zoom_out,box_zoom,undo,redo,reset,save,"

    p = figure(tools=tools)
    p.title.align = 'center'
    p.quad(
        top=hist,
        bottom=0,
        left=edges[:-1],
        right=edges[1:],
        alpha=0.5
    )

    p.y_range.start = 0
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = 'Event Count'

    return p


def plot_scatter(
        x,
        y,
        dim_ids=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        color_density=True
):
    """
    Creates a Bokeh scatter plot from the two 1-D data arrays.

    :param x: 1-D array of data values for the x-axis
    :param y: 1-D array of data values for the y-axis
    :param dim_ids: Labels to use for the x-axis & y-axis, respectively
    :param x_min: Lower bound of x-axis. If None, channel's min value will
        be used with some padding to keep events off the edge of the plot.
    :param x_max: Upper bound of x-axis. If None, channel's max value will
        be used with some padding to keep events off the edge of the plot.
    :param y_min: Lower bound of y-axis. If None, channel's min value will
        be used with some padding to keep events off the edge of the plot.
    :param y_max: Upper bound of y-axis. If None, channel's max value will
        be used with some padding to keep events off the edge of the plot.
    :param color_density: Whether to color the events by density, similar
        to a heat map. Default is True.
    :return: A Bokeh Figure object containing the interactive scatter plot.
    """
    if len(x) > 0:
        x_min, x_max = _calculate_extent(x, d_min=x_min, d_max=x_max, pad=0.02)
    if len(y) > 0:
        y_min, y_max = _calculate_extent(y, d_min=y_min, d_max=y_max, pad=0.02)

    if y_max > x_max:
        radius_dimension = 'y'
        radius = 0.003 * y_max
    else:
        radius_dimension = 'x'
        radius = 0.003 * x_max

    if color_density:
        data, x_e, y_e = np.histogram2d(x, y, bins=[38, 38])
        z = interpn(
            (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
            data,
            np.vstack([x, y]).T,
            method="splinef2d",
            bounds_error=False
        )
        z[np.isnan(z)] = 0

        # sort by density (z) so the more dense points are on top for better
        # color display
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    else:
        z = np.zeros(len(x))

    colors_array = new_jet(colors.Normalize()(z))
    z_colors = [
        "#%02x%02x%02x" % (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in colors_array
    ]

    tools = "crosshair,hover,pan,zoom_in,zoom_out,box_zoom,undo,redo,reset,save,"
    p = figure(
        tools=tools,
        x_range=(x_min, x_max),
        y_range=(y_min, y_max)
    )

    p.xaxis.axis_label = dim_ids[0]
    p.yaxis.axis_label = dim_ids[1]

    p.scatter(
        x,
        y,
        radius=radius,
        radius_dimension=radius_dimension,
        fill_color=z_colors,
        fill_alpha=0.4,
        line_color=None
    )

    return p
