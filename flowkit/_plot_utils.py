import numpy as np
from scipy.interpolate import interpn
import colorsys
from matplotlib import cm, colors
from matplotlib import pyplot
from matplotlib.patches import Ellipse
from bokeh.plotting import figure


def generate_custom_colormap(cmap_sample_indices, base_cmap):
    x = np.linspace(0, np.pi, base_cmap.N)
    new_lum = (np.sin(x) * 0.75) + .1

    new_color_list = []

    for i in cmap_sample_indices:
        (r, g, b, a) = base_cmap(i)
        (h, s, v) = colorsys.rgb_to_hsv(r, g, b)

        mod_v = (v * ((196 - abs(i - 196)) / 196) + new_lum[i]) / 2

        new_r, new_g, new_b = colorsys.hsv_to_rgb(h, s, mod_v)
        (_, new_l, _) = colorsys.rgb_to_hls(new_r, new_g, new_b)

        new_color_list.append((new_r, new_g, new_b))

    return colors.LinearSegmentedColormap.from_list(
        'custom_' + base_cmap.name,
        new_color_list,
        256
    )


cm_sample = [
    0, 8, 16, 24, 32, 40, 48, 52, 60, 64, 72, 80, 92,
    100, 108, 116, 124, 132,
    139, 147, 155, 159,
    163, 167, 171, 175, 179, 183, 187, 191, 195, 199, 215, 231, 239
]

new_jet = generate_custom_colormap(cm_sample, cm.get_cmap('jet'))


def get_false_bounds(bool_array):
    diff = np.diff(np.hstack((0, bool_array, 0)))

    start = np.where(diff == 1)
    end = np.where(diff == -1)

    return start[0], end[0]


# TODO: integrate functionality into Sample class, change xform to accept/use Sample instance xform
def plot_channel(chan_events, label, subplot_ax, xform=False, bad_events=None):
    if xform:
        # TODO: change to accept a Transform sub-class instance
        chan_events = np.arcsinh(chan_events * 0.003)

    my_cmap = pyplot.cm.get_cmap('jet')
    my_cmap.set_under('w', alpha=0)

    bins = int(np.sqrt(chan_events.shape[0]))
    event_range = range(0, chan_events.shape[0])

    subplot_ax.set_title(label, fontsize=16)
    subplot_ax.set_xlabel("Events", fontsize=14)

    subplot_ax.hist2d(
        event_range,
        chan_events,
        bins=[bins, bins],
        cmap=my_cmap,
        vmin=0.9
    )

    if bad_events is not None:
        starts, ends = get_false_bounds(bad_events)

        for i, s in enumerate(starts):
            subplot_ax.axvspan(
                event_range[s],
                event_range[ends[i] - 1],
                facecolor='pink',
                alpha=0.3,
                edgecolor='deeppink'
            )


def calculate_extent(data_1d, d_min=None, d_max=None, pad=0.0):
    data_min = data_1d.min()
    data_max = data_1d.max()

    # determine padding to keep min/max events off the edge
    pad_d = max(abs(data_1d.min()), abs(data_1d.max())) * pad

    if d_min is None:
        d_min = data_min - pad_d
    if d_max is None:
        d_max = data_max + pad_d

    return d_min, d_max


# TODO: change to Bokeh Ellipse
def calculate_ellipse(center_x, center_y, covariance_matrix, n_std_dev=3):
    values, vectors = np.linalg.eigh(covariance_matrix)
    order = values.argsort()[::-1]
    values = values[order]
    vectors = vectors[:, order]

    theta = np.degrees(np.arctan2(*vectors[:, 0][::-1]))

    # make all angles positive
    if theta < 0:
        theta += 360

    # Width and height are "full" widths, not radius
    width, height = 2.0 * n_std_dev * np.sqrt(values)

    ellipse = Ellipse(
        xy=(center_x, center_y),
        width=width,
        height=height,
        angle=float(theta)
    )

    return ellipse


def plot_histogram(x, x_label='x', bins=None, title=None):
    if bins is None:
        bins = 'sqrt'

    hist, edges = np.histogram(x, density=False, bins=bins)

    tools = "crosshair,pan,zoom_in,zoom_out,box_zoom,undo,redo,reset,save,"

    p = figure(title=title, tools=tools)
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
        dim_labels=None,
        title=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        color_density=True
):
    x_min, x_max = calculate_extent(x, d_min=x_min, d_max=x_max, pad=0.02)
    y_min, y_max = calculate_extent(y, d_min=y_min, d_max=y_max, pad=0.02)

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

    tools = "crosshair,pan,zoom_in,zoom_out,box_zoom,undo,redo,reset,save,"
    p = figure(
        tools=tools,
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
        title=title
    )
    p.title.align = 'center'

    p.xaxis.axis_label = dim_labels[0]
    p.yaxis.axis_label = dim_labels[1]

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
