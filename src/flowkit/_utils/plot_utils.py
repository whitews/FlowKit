"""
Utility functions related to plotting
"""
import numpy as np
from scipy.interpolate import interpn
import contourpy
from bokeh.plotting import figure
from bokeh.models import Ellipse, Patch, Span, BoxAnnotation, Rect, ColumnDataSource, Title
from scipy.stats import gaussian_kde
from .._models import gates, dimension
from .._models.gating_strategy import GatingStrategy


LINE_COLOR_DEFAULT = "#1F77B4"
LINE_COLOR_CONTRAST = "#73D587"
LINE_WIDTH_DEFAULT = 3
FILL_COLOR_DEFAULT = 'lime'
FILL_ALPHA_DEFAULT = 0.08


custom_heat_palette = [
    '#000020', '#000021', '#000022', '#000023', '#000024', '#000025', '#000026', '#000027',
    '#000028', '#000029', '#00002a', '#00002b', '#00002c', '#00002d', '#00002e', '#00002f',
    '#000030', '#000031', '#000032', '#000033', '#000034', '#000035', '#000036', '#000037',
    '#000038', '#00003a', '#00003c', '#00003f', '#000042', '#000045', '#000048', '#00004b',
    '#00004e', '#000151', '#000254', '#000357', '#00045a', '#00045d', '#000560', '#000662',
    '#000964', '#000c66', '#000f68', '#00126a', '#00166d', '#001a6f', '#001e72', '#002274',
    '#002677', '#002a79', '#002e7b', '#00327e', '#003680', '#003a83', '#003e85', '#004287',
    '#00468a', '#004a8c', '#004e8e', '#005290', '#005692', '#005a94', '#005e96', '#006298',
    '#006699', '#006a9a', '#006f9b', '#00749c', '#007a9d', '#00809e', '#00869f', '#008ca0',
    '#0091a1', '#0096a2', '#009aa3', '#009ea4', '#00a2a5', '#00a6a6', '#00aaa5', '#00ada4',
    '#00b0a2', '#00b2a0', '#00b49d', '#00b69a', '#00b797', '#00b894', '#00b991', '#00ba8e',
    '#00bb8b', '#00bc88', '#00bd85', '#00be82', '#00bf7f', '#00c07c', '#00c179', '#00c276',
    '#00c373', '#00c470', '#00c56d', '#00c56a', '#00c667', '#00c764', '#00c860', '#00c95b',
    '#00ca55', '#00cb4e', '#00cb47', '#00cc40', '#00cd39', '#00ce32', '#00ce2b', '#00cf24',
    '#02d01f', '#07d11a', '#0cd116', '#12d212', '#17d20e', '#1cd30a', '#22d406', '#27d402',
    '#2dd500', '#34d500', '#3ad600', '#41d600', '#48d700', '#4ed700', '#55d800', '#5cd800',
    '#62d900', '#68d900', '#6eda00', '#74da00', '#7ada00', '#7fdb00', '#85db00', '#8bdc00',
    '#90dc00', '#95dc00', '#9add00', '#9edd00', '#a3dd00', '#a8de00', '#acde00', '#b1de00',
    '#b4de00', '#b6df00', '#b8df00', '#badf00', '#bcdf00', '#bedf00', '#c0df00', '#c2df00',
    '#c4df00', '#c5df00', '#c7e000', '#c9e000', '#cbe000', '#cce000', '#cee000', '#d0e000',
    '#d2df00', '#d3de00', '#d5dd00', '#d7db00', '#d9da00', '#dbd900', '#dcd800', '#ded600',
    '#dfd500', '#dfd300', '#dfd100', '#e0cf00', '#e0cd00', '#e0cc00', '#e0ca00', '#e1c800',
    '#e1c600', '#e1c500', '#e1c300', '#e1c100', '#e1c000', '#e1be00', '#e1bc00', '#e1bb00',
    '#e1b900', '#e1b700', '#e1b600', '#e1b400', '#e1b200', '#e1b100', '#e1af00', '#e1ae00',
    '#e1ac00', '#e1aa00', '#e1a900', '#e1a700', '#e1a500', '#e1a400', '#e1a200', '#e1a000',
    '#e19f00', '#e09d00', '#e09b00', '#e09900', '#e09800', '#e09600', '#e09400', '#e09300',
    '#e09100', '#e08f00', '#e08e00', '#e08c00', '#e08a00', '#e08900', '#e08700', '#df8500',
    '#df8400', '#df8200', '#df8000', '#df7f00', '#df7d00', '#df7b00', '#df7900', '#df7800',
    '#de7600', '#de7400', '#dd7200', '#dc7000', '#dc6e00', '#db6c00', '#da6a00', '#da6800',
    '#d76200', '#d45b00', '#d05300', '#cd4c00', '#ca4500', '#c63e00', '#c33700', '#c03000',
    '#bc2a00', '#b72400', '#b31f00', '#ae1900', '#aa1300', '#a50e00', '#a10800', '#9d0200',
    '#990200', '#950100', '#920100', '#8e0100', '#8b0000', '#870000', '#840000', '#800000'
]


def _calculate_extent(data_1d, d_min=None, d_max=None, pad=0.0):
    data_min = np.min(data_1d)
    data_max = np.max(data_1d)

    # determine padding to keep min/max events off the edge
    pad_d = max(abs(data_min), abs(data_max)) * pad

    if d_min is None:
        d_min = data_min - pad_d
    if d_max is None:
        d_max = data_max + pad_d

    return d_min, d_max


def _quantiles_to_levels(data, quantiles):
    """Return data levels corresponding to quantile cuts of mass."""
    # Make sure quantiles is a NumPy array
    quantiles = np.array(quantiles)
    values = np.ravel(data)
    sorted_values = np.sort(values)[::-1]

    normalized_values = np.cumsum(sorted_values) / values.sum()

    idx = np.searchsorted(normalized_values, 1 - quantiles)
    levels = np.take(sorted_values, idx, mode="clip")

    return levels


def _calculate_2d_gaussian_kde(x, y, bw_method='scott', grid_size=200, pad_factor=3):
    """Calculate a 2D PDF from a Gaussian KDE"""
    # First get the KDE, so we can get the calculated bandwidths
    # for each dimension to use for padding the grid.
    kernel = gaussian_kde([x, y], bw_method=bw_method)
    bw_x, bw_y = np.sqrt(np.diag(kernel.covariance).squeeze())

    min_x, max_x = x.min(), x.max()
    min_y, max_y = y.min(), y.max()

    # need to pad data edges for the grid calculation
    grid_min_x = min_x - bw_x * pad_factor
    grid_max_x = max_x + bw_x * pad_factor

    grid_min_y = min_y - bw_y * pad_factor
    grid_max_y = max_y + bw_y * pad_factor

    # create meshgrid
    meshgrid_x, meshgrid_y = np.meshgrid(
        np.linspace(grid_min_x, grid_max_x, grid_size),
        np.linspace(grid_min_y, grid_max_y, grid_size)
    )
    positions = np.vstack([meshgrid_x.flatten(), meshgrid_y.flatten()])

    # this is the bottleneck step
    estimated_pdf = np.reshape(kernel(positions).T, meshgrid_x.shape)

    return meshgrid_x, meshgrid_y, estimated_pdf


def _build_contour_generator(mesh_x, mesh_y, estimated_pdf):
    c_gen = contourpy.contour_generator(x=mesh_x, y=mesh_y, z=estimated_pdf)

    return c_gen


def render_polygon(
        vertices,
        line_color=LINE_COLOR_CONTRAST,
        line_width=LINE_WIDTH_DEFAULT,
        fill_color=FILL_COLOR_DEFAULT,
        fill_alpha=FILL_ALPHA_DEFAULT
):
    """
    Renders a Bokeh polygon for plotting
    :param vertices: list of 2-D coordinates representing vertices of the polygon
    :param line_color: Color for the polygon boundary (as RGB hex string or CSS color name)
    :param line_width: Line width in pixels for the polygon boundary
    :param fill_color: Color for the polygon interior (as RGB hex string or CSS color name)
    :param fill_alpha: Opacity of the polygon as a float from 0 (transparent) to 1 (opaque)
    :return: tuple containing the Bokeh ColumnDataSource and polygon glyphs (as Patch object)
    """
    x_coords, y_coords = list(zip(*[v for v in vertices]))

    source = ColumnDataSource(dict(x=x_coords, y=y_coords))

    poly = Patch(
        x='x',
        y='y',
        fill_color=fill_color,
        fill_alpha=fill_alpha,
        line_width=line_width,
        line_color=line_color
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
            Span(location=left, dimension='height', line_width=LINE_WIDTH_DEFAULT, line_color=LINE_COLOR_DEFAULT)
        )
    if dim_maximums[0] is not None:
        right = dim_maximums[0]
        renderers.append(
            Span(location=right, dimension='height', line_width=LINE_WIDTH_DEFAULT, line_color=LINE_COLOR_DEFAULT)
        )
    if len(dim_minimums) > 1:
        if dim_minimums[1] is not None:
            bottom = dim_minimums[1]
            renderers.append(
                Span(location=bottom, dimension='width', line_width=LINE_WIDTH_DEFAULT, line_color=LINE_COLOR_DEFAULT)
            )
        if dim_maximums[1] is not None:
            top = dim_maximums[1]
            renderers.append(
                Span(location=top, dimension='width', line_width=LINE_WIDTH_DEFAULT, line_color=LINE_COLOR_DEFAULT)
            )

    mid_box = BoxAnnotation(
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        fill_alpha=FILL_ALPHA_DEFAULT,
        fill_color=FILL_COLOR_DEFAULT
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
        fill_color=FILL_COLOR_DEFAULT,
        fill_alpha=FILL_ALPHA_DEFAULT,
        line_width=LINE_WIDTH_DEFAULT
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
            Span(location=x_loc, dimension='height', line_width=LINE_WIDTH_DEFAULT, line_color=LINE_COLOR_DEFAULT)
        )
    for y_loc in y_locs:
        renderers.append(
            Span(location=y_loc, dimension='width', line_width=LINE_WIDTH_DEFAULT, line_color=LINE_COLOR_DEFAULT)
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
        line_width=LINE_WIDTH_DEFAULT,
        line_color=LINE_COLOR_DEFAULT,
        fill_color=FILL_COLOR_DEFAULT,
        fill_alpha=FILL_ALPHA_DEFAULT
    )

    return ellipse


def plot_histogram(x, x_label='x', bins=None, width=600, height=600):
    """
    Creates a Bokeh histogram plot of the given 1-D data array.

    :param x: 1-D array of data values
    :param x_label: Label to use for the x-axis
    :param bins: Number of bins to use for the histogram or a string compatible
            with the NumPy histogram function. If None, the number of bins is
            determined by the square root rule.
    :param height: Height of plot in pixels. Default is 600.
    :param width: Width of plot in pixels. Default is 600.
    :return: Bokeh Figure object containing the histogram
    """
    if bins is None:
        bins = 'sqrt'

    hist, edges = np.histogram(x, density=False, bins=bins)

    tools = "crosshair,hover,pan,zoom_in,zoom_out,box_zoom,undo,redo,reset,save,"

    p = figure(tools=tools, width=width, height=height)
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

    # set padding to match scatter plot
    # scatter has 0.02, but we need to account for the bar width
    # so doubling that looks about right
    p.x_range.range_padding = 0.04

    return p


def plot_scatter(
        x,
        y,
        x_label=None,
        y_label=None,
        event_mask=None,
        highlight_mask=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        color_density=True,
        bin_width=4,
        height=600,
        width=600
):
    """
    Creates a Bokeh scatter plot from the two 1-D data arrays.

    :param x: 1-D array of data values for the x-axis
    :param y: 1-D array of data values for the y-axis
    :param x_label: Label for the x-axis
    :param y_label: Label for the y-axis
    :param event_mask: Boolean array of events to plot. Takes precedence
            over highlight_mask (i.e. events marked False in event_mask will
            never be plotted).
    :param highlight_mask: Boolean array of event indices to highlight
        in color. Non-highlighted events will be light grey.
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
    :param bin_width: Bin size to use for the color density, in units of
        event point size. Larger values produce smoother gradients.
        Default is 4 for a 4x4 grid size.
    :param height: Height of plot in pixels. Default is 600.
    :param width: Width of plot in pixels. Default is 600.
    :return: A Bokeh Figure object containing the interactive scatter plot.
    """
    # before anything, check for event_mask
    if event_mask is not None:
        # filter x & y
        x = x[event_mask]
        y = y[event_mask]

        # sync highlight_mask if given
        if highlight_mask is not None:
            highlight_mask = highlight_mask[event_mask]

    if len(x) > 0:
        x_min, x_max = _calculate_extent(x, d_min=x_min, d_max=x_max, pad=0.02)
    else:
        # empty array, set extents to 0 to avoid errors
        x_min = x_max = 0

        # turn off color density
        color_density = False

    if len(y) > 0:
        y_min, y_max = _calculate_extent(y, d_min=y_min, d_max=y_max, pad=0.02)
    else:
        # empty array, set extents to 0 to avoid errors
        y_min = y_max = 0

    if y_max > x_max:
        radius_dimension = 'y'
        radius = 0.003 * y_max
    else:
        radius_dimension = 'x'
        radius = 0.003 * x_max

    if color_density:
        # bin size set to cover NxN radius (radius size is percent of view)
        # can be set by user via bin_width kwarg
        bin_count = int(1 / (bin_width * 0.003))

        # But that's just the bins needed for the requested plot ranges.
        # We need to extend those bins to the full data range
        x_view_range = x_max - x_min
        y_view_range = y_max - y_min

        x_data_min = np.min(x)
        x_data_max = np.max(x)
        y_data_min = np.min(y)
        y_data_max = np.max(y)
        x_data_range = x_data_max - x_data_min
        y_data_range = y_data_max - y_data_min

        x_bin_multiplier = x_data_range / x_view_range
        x_bin_count = int(x_bin_multiplier * bin_count)
        y_bin_multiplier = y_data_range / y_view_range
        y_bin_count = int(y_bin_multiplier * bin_count)

        # avoid bin count of zero
        if x_bin_count <= 0:
            x_bin_count = 1
        if y_bin_count <= 0:
            y_bin_count = 1

        cd_x_min = x_data_min - (x_data_range / x_bin_count)
        cd_x_max = x_data_max + (x_data_range / x_bin_count)
        cd_y_min = y_data_min - (y_data_range / y_bin_count)
        cd_y_max = y_data_max + (y_data_range / y_bin_count)

        # noinspection PyTypeChecker
        hist_data, x_edges, y_edges = np.histogram2d(
            x,
            y,
            bins=[x_bin_count, y_bin_count],
            range=[[cd_x_min, cd_x_max], [cd_y_min, cd_y_max]]
        )
        z = interpn(
            (0.5 * (x_edges[1:] + x_edges[:-1]), 0.5 * (y_edges[1:] + y_edges[:-1])),
            hist_data,
            np.vstack([x, y]).T,
            method="linear",  # use linear not spline, spline tends to overshoot into negative values
            bounds_error=False
        )
        z[np.isnan(z)] = 0

        # sort by density (z) so the more dense points are on top for better
        # color display
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        if highlight_mask is not None:
            # re-order the highlight indices to match
            highlight_mask = highlight_mask[idx]

        # check if z max - z min is 0 (e.g. a single data point)
        if z.max() - z.min() == 0:
            z_norm = np.zeros(len(x))
        else:
            z_norm = (z - z.min()) / (z.max() - z.min())
    else:
        z_norm = np.zeros(len(x))

    z_colors = np.array([custom_heat_palette[int(z * 255)] for z in z_norm])

    if highlight_mask is not None:
        z_colors[~highlight_mask] = "#d3d3d3"
        fill_alpha = np.zeros(len(z_colors))
        fill_alpha[~highlight_mask] = 0.3
        fill_alpha[highlight_mask] = 0.4

        highlight_idx = np.flatnonzero(highlight_mask)
        non_light_idx = np.flatnonzero(~highlight_mask)
        final_idx = np.concatenate([non_light_idx, highlight_idx])

        x = x[final_idx]
        y = y[final_idx]
        z_colors = z_colors[final_idx]
        fill_alpha = fill_alpha[final_idx]
    else:
        fill_alpha = 0.4

    tools = "crosshair,hover,pan,zoom_in,zoom_out,box_zoom,undo,redo,reset,save,"
    p = figure(
        tools=tools,
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
        width=width,
        height=height
    )

    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = y_label

    if len(x) > 0:
        p.circle(
            x,
            y,
            radius=radius,
            radius_dimension=radius_dimension,
            fill_color=z_colors,
            fill_alpha=fill_alpha,
            line_color=None
        )

    return p


def plot_contours(
        x,
        y,
        x_label=None,
        y_label=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        plot_events=False,
        fill=False,
        width=600,
        height=600
):
    """
    Create a Bokeh plot of contours from the two 1-D data arrays.

    :param x: 1-D array of data values for the x-axis
    :param y: 1-D array of data values for the y-axis
    :param x_label: Label for the x-axis
    :param y_label: Label for the y-axis
    :param x_min: Lower bound of x-axis. If None, channel's min value will
        be used with some padding to keep events off the edge of the plot.
    :param x_max: Upper bound of x-axis. If None, channel's max value will
        be used with some padding to keep events off the edge of the plot.
    :param y_min: Lower bound of y-axis. If None, channel's min value will
        be used with some padding to keep events off the edge of the plot.
    :param y_max: Upper bound of y-axis. If None, channel's max value will
        be used with some padding to keep events off the edge of the plot.
    :param plot_events: Whether to plot the events as a scatter plot in
        addition to the contours.
    :param fill: Whether to color fill contours by density, similar
        to a heat map. Default is False.
    :param height: Height of plot in pixels. Default is 600.
    :param width: Width of plot in pixels. Default is 600.
    :return: A Bokeh Figure object containing the interactive scatter plot.
    """
    # Calculate Gaussian KDE, using default bandwidth & grid size (maybe expose these later?)
    mesh_x, mesh_y, est_pdf = _calculate_2d_gaussian_kde(x, y)

    # Get contour generator from our PDF on our grid
    c_gen = _build_contour_generator(mesh_x, mesh_y, est_pdf)

    # Generate a modest set of quantiles to plot (not too many as it gets crowded easily).
    # And, don't start at 0 (there's not really a contour there).
    quantiles = list(np.linspace(0.04, 1, 9).round(2))

    # Convert quantiles to corresponding levels in our PDF
    levels = _quantiles_to_levels(est_pdf, quantiles)

    # Unless user set them, set our axis bounds based on contour bounds
    # instead of the data ranges b/c the contours can be wider.
    if x_min is None:
        x_min = mesh_x.min()
    if x_max is None:
        x_max = mesh_x.max()
    if y_min is None:
        y_min = mesh_y.min()
    if y_max is None:
        y_max = mesh_y.max()

    # if we are plotting events, get the Bokeh figure from plot_scatter,
    # else we'll make a new one
    if plot_events:
        fig = plot_scatter(
            x, y,
            x_label=x_label, y_label=y_label,
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            height=height, width=width
        )
    else:
        tools = "crosshair,hover,pan,zoom_in,zoom_out,box_zoom,undo,redo,reset,save,"
        fig = figure(
            tools=tools,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            height=height,
            width=width
        )

        fig.xaxis.axis_label = x_label
        fig.yaxis.axis_label = y_label

    # Add the sets of contours as polygons to the figure
    for i, level in enumerate(levels):
        poly_lines = c_gen.lines(level)

        if fill:
            fill_color = custom_heat_palette[int(quantiles[i] * 255)]
            fill_alpha = min(quantiles[i] * 2, 1)
            line_color = None
        else:
            fill_color = None
            fill_alpha = 0
            line_color = LINE_COLOR_DEFAULT

        for poly in poly_lines:
            source, glyph = render_polygon(
                poly, line_color=line_color, fill_color=fill_color, fill_alpha=fill_alpha
            )
            fig.add_glyph(source, glyph)

    return fig


def plot_gate(
        gate_id,
        gating_strategy: GatingStrategy,
        sample,
        subsample_count=10000,
        random_seed=1,
        event_mask=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        color_density=True,
        bin_width=4,
        hist_bins=None,
        width=600,
        height=600
):
    """
    Returns an interactive plot for the specified gate. The type of plot is
    determined by the number of dimensions used to define the gate: single
    dimension gates will be histograms, 2-D gates will be returned as a
    scatter plot.

    :param gate_id: tuple of gate name and gate path (also a tuple)
    :param gating_strategy: GatingStrategy containing gate_id
    :param sample: Sample instance to plot
    :param subsample_count: Number of events to use as a subsample. If the number of
        events in the Sample is less than the requested subsample count, then the
        maximum number of available events is used for the subsample.
    :param random_seed: Random seed used for subsampling events
    :param event_mask: Boolean array of events to plot (i.e. parent gate event membership)
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
    :param bin_width: Bin size to use for the color density, in units of
        event point size. Larger values produce smoother gradients.
        Default is 4 for a 4x4 grid size.
    :param hist_bins: If the gate is only in 1 dimension, this option
        controls the number of bins to use for the histogram. If None,
        the number of bins is determined by the square root rule. This
        option is ignored for any gates in more than 1 dimension.
    :param height: Height of plot in pixels. Default is 600.
    :param width: Width of plot in pixels. Default is 600.
    :return: A Bokeh Figure object containing the interactive scatter plot.
    """
    (gate_name, gate_path) = gate_id
    sample_id = sample.id
    gate = gating_strategy.get_gate(gate_name, gate_path=gate_path, sample_id=sample_id)

    # check for a boolean gate, there's no reasonable way to plot these
    if isinstance(gate, gates.BooleanGate):
        raise TypeError("Plotting Boolean gates is not allowed (gate %s)" % gate.gate_name)

    dim_ids_ordered = []
    dim_is_ratio = []
    dim_comp_refs = []
    dim_min = []
    dim_max = []
    for i, dim in enumerate(gate.dimensions):
        if isinstance(dim, dimension.RatioDimension):
            dim_ids_ordered.append(dim.ratio_ref)
            tmp_dim_min = dim.min
            tmp_dim_max = dim.max
            is_ratio = True
        elif isinstance(dim, dimension.QuadrantDivider):
            dim_ids_ordered.append(dim.dimension_ref)
            tmp_dim_min = None
            tmp_dim_max = None
            is_ratio = False
        else:
            dim_ids_ordered.append(dim.id)
            tmp_dim_min = dim.min
            tmp_dim_max = dim.max
            is_ratio = False

        dim_min.append(tmp_dim_min)
        dim_max.append(tmp_dim_max)
        dim_is_ratio.append(is_ratio)
        dim_comp_refs.append(dim.compensation_ref)

    # dim count determines if we need a histogram, scatter, or multi-scatter
    dim_count = len(dim_ids_ordered)
    if dim_count == 1:
        gate_type = 'hist'
    elif dim_count == 2:
        gate_type = 'scatter'
    elif dim_count > 2:
        raise NotImplementedError("Plotting of gates with >2 dimensions is not supported")
    else:
        # there are no dimensions
        raise ValueError("Gate %s appears to not reference any dimensions" % gate_name)

    # Apply requested subsampling
    sample.subsample_events(subsample_count=subsample_count, random_seed=random_seed)

    # TODO: investigate whether caching processed events speeds up plotting
    # noinspection PyProtectedMember
    events = gating_strategy._preprocess_sample_events(
        sample,
        gate
    )

    # Use event mask, if given
    if event_mask is not None:
        is_subsample = np.zeros(sample.event_count, dtype=bool)
        is_subsample[sample.subsample_indices] = True
        idx_to_plot = np.logical_and(event_mask, is_subsample)
    else:
        idx_to_plot = sample.subsample_indices

    x = events.loc[idx_to_plot, dim_ids_ordered[0]].values

    dim_ids = []

    if dim_is_ratio[0]:
        dim_ids.append(dim_ids_ordered[0])
        x_pnn_label = None
    else:
        try:
            x_index = sample.get_channel_index(dim_ids_ordered[0])
        except ValueError:
            # might be a label reference in the comp matrix
            matrix = gating_strategy.get_comp_matrix(dim_comp_refs[0])
            try:
                matrix_dim_idx = matrix.fluorochromes.index(dim_ids_ordered[0])
            except ValueError:
                raise ValueError("%s not found in list of matrix fluorochromes" % dim_ids_ordered[0])
            detector = matrix.detectors[matrix_dim_idx]
            x_index = sample.get_channel_index(detector)

        x_pnn_label = sample.pnn_labels[x_index]

        if sample.pns_labels[x_index] != '':
            dim_ids.append('%s (%s)' % (sample.pns_labels[x_index], x_pnn_label))
        else:
            dim_ids.append(sample.pnn_labels[x_index])

    y_pnn_label = None

    if dim_count > 1:
        if dim_is_ratio[1]:
            dim_ids.append(dim_ids_ordered[1])

        else:
            try:
                y_index = sample.get_channel_index(dim_ids_ordered[1])
            except ValueError:
                # might be a label reference in the comp matrix
                matrix = gating_strategy.get_comp_matrix(dim_comp_refs[1])
                try:
                    matrix_dim_idx = matrix.fluorochromes.index(dim_ids_ordered[1])
                except ValueError:
                    raise ValueError("%s not found in list of matrix fluorochromes" % dim_ids_ordered[1])
                detector = matrix.detectors[matrix_dim_idx]
                y_index = sample.get_channel_index(detector)

            y_pnn_label = sample.pnn_labels[y_index]

            if sample.pns_labels[y_index] != '':
                dim_ids.append('%s (%s)' % (sample.pns_labels[y_index], y_pnn_label))
            else:
                dim_ids.append(sample.pnn_labels[y_index])

    if gate_type == 'scatter':
        y = events.loc[idx_to_plot, dim_ids_ordered[1]].values

        p = plot_scatter(
            x,
            y,
            x_label=dim_ids[0],
            y_label=dim_ids[1],
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            color_density=color_density,
            bin_width=bin_width,
            height=height,
            width=width
        )
    elif gate_type == 'hist':
        p = plot_histogram(x, dim_ids[0], height=height, width=width, bins=hist_bins)
    else:
        raise NotImplementedError("Only histograms and scatter plots are supported in this version of FlowKit")

    if isinstance(gate, gates.PolygonGate):
        source, glyph = render_polygon(gate.vertices)
        p.add_glyph(source, glyph)
    elif isinstance(gate, gates.EllipsoidGate):
        ellipse = render_ellipse(
            gate.coordinates[0],
            gate.coordinates[1],
            gate.covariance_matrix,
            gate.distance_square
        )
        p.add_glyph(ellipse)
    elif isinstance(gate, gates.RectangleGate):
        # rectangle gates in GatingML may not actually be rectangles, as the min/max for the dimensions
        # are options. So, if any of the dim min/max values are missing it is essentially a set of ranges.

        if None in dim_min or None in dim_max or dim_count == 1:
            renderers = render_ranges(dim_min, dim_max)

            p.renderers.extend(renderers)
        else:
            # a true rectangle
            rect = render_rectangle(dim_min, dim_max)
            p.add_glyph(rect)
    elif isinstance(gate, gates.QuadrantGate):
        x_locations = []
        y_locations = []

        for div in gate.dimensions:
            if div.dimension_ref == x_pnn_label:
                x_locations.extend(div.values)
            elif div.dimension_ref == y_pnn_label and y_pnn_label is not None:
                y_locations.extend(div.values)

        renderers = render_dividers(x_locations, y_locations)
        p.renderers.extend(renderers)
    else:
        raise NotImplementedError(
            "Plotting of %s gates is not supported in this version of FlowKit" % gate.__class__.__name__
        )

    if gate_path is not None:
        full_gate_path = gate_path[1:]  # omit 'root'
        full_gate_path = full_gate_path + (gate_name,)
        sub_title = ' > '.join(full_gate_path)

        # truncate beginning of long gate paths
        if len(sub_title) > 72:
            sub_title = '...' + sub_title[-69:]
        p.add_layout(
            Title(text=sub_title, text_font_style="italic", text_font_size="1em", align='center'),
            'above'
        )
    else:
        p.add_layout(
            Title(text=gate_name, text_font_style="italic", text_font_size="1em", align='center'),
            'above'
        )

    plot_title = "%s" % sample_id
    p.add_layout(
        Title(text=plot_title, text_font_size="1.1em", align='center'),
        'above'
    )

    return p
