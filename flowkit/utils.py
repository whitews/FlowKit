import re
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import cm, colors
from matplotlib import pyplot
import colorsys


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


sample = [
    0, 8, 16, 24, 32, 40, 48, 52, 60, 64, 72, 80, 92,
    100, 108, 116, 124, 132,
    139, 147, 155, 159,
    163, 167, 171, 175, 179, 183, 187, 191, 195, 199, 215, 231, 239
]

new_jet = generate_custom_colormap(sample, cm.jet)


def convert_matrix_text_to_array(matrix_text, fluoro_labels, fluoro_indices):
    """
    Converts a CSV text string to a NumPy array
    :param matrix_text: CSV text string where header contains the fluoro labels
    :param fluoro_labels: text labels of the FCS fluorescent channels
    :param fluoro_indices: channel indices of the fluorescent channels
    :return:
    """
    # parse the matrix text and validate the number of params match
    # the number of fluoro params and that the matrix
    # values are numbers (can be exp notation)
    matrix_text = matrix_text.splitlines()

    # TODO: verify function can parse FCS spill value properly

    # but first we must find a valid header line and we will require that the matrix
    # follows on the next lines, ignoring any additional lines before or after
    # the header contains labels matching the PnN value(FCS text field)
    # and may be tab or comma delimited
    # (spaces can't be delimiters b/c they are allowed in the PnN value)
    header = None
    header_line_index = None
    non_matching_labels = fluoro_labels.copy()
    for i, line in enumerate(matrix_text):
        line_values = re.split('[\t,]', line)

        label_diff = set(fluoro_labels).symmetric_difference(line_values)

        if len(label_diff) != 0:
            # if any labels are missing or extra ones found, then not a valid header row
            # And, for more informative error reporting, we keep track of the mis-matches
            # to include in the error message
            if len(label_diff) < len(non_matching_labels):
                non_matching_labels = label_diff
            continue
        else:
            header = line_values
            header_line_index = i
            break

    if header is None:
        error_message = "\n".join(
            [
                "Matrix labels do not match fluoro channel labels in FCS file",
                "Mis-matched labels:",
                ", ".join(non_matching_labels)
            ]
        )
        raise ValueError(error_message)

    matrix_start = header_line_index + 1
    matrix_end = matrix_start + len(fluoro_labels)

    if len(matrix_text) < matrix_end:
        raise ValueError("Too few rows in compensation")

    matrix_text = matrix_text[matrix_start:matrix_end]

    header_channel_numbers = []

    for h in header:
        # check if label is in given fluoro labels, will raise a ValueError if missing
        # Note: this will give us the index of the given fluoro labels, which is not the
        # index of the fluoro channel in the original FCS file because of scatter channels,
        # etc. We look up the 'true' index from given fluoro_indices. Also, we increment
        # to match the original PnN numbers that index from 1 (not 0).
        # We store the channel number in the first row of the numpy array, as it is more
        # reliable to identify parameters than some concatenation of parameter attributes.
        fluoro_index = fluoro_labels.index(h)
        true_fluoro_index = fluoro_indices[fluoro_index]
        header_channel_numbers.append(true_fluoro_index + 1)

    matrix_array = np.array(header_channel_numbers)

    # convert the matrix text to numpy array
    for line in matrix_text:
        line_values = re.split('[\t,]', line)
        for i, value in enumerate(line_values):
            line_values[i] = float(line_values[i])
        if len(line_values) > len(fluoro_labels):
            raise ValueError("Too many values in line: %s" % line)
        elif len(line_values) < len(fluoro_labels):
            raise ValueError("Too few values in line: %s" % line)
        else:
            matrix_array = np.vstack([matrix_array, line_values])

    return matrix_array


def parse_compensation_matrix(compensation, channel_labels):
    """
    Returns a NumPy array with the compensation matrix where the first row are the
    indices of the fluorescent channels
    :param compensation: Compensation matrix: may be a NumPy array, a CSV file path,
        a pathlib Path object to a CSV or TSV file or a string of CSV text.
        If a string, both multi-line, traditional CSV and the single
        line FCS spill formats are supported. If a NumPy array, we assume the columns are in
        the same order as the channel labels
    :param channel_labels: Channel labels from the FCS file's PnN fields, must be in
        the same order as they appear in the FCS file
    :return:
    """
    non_fluoro_channels = [
        'FSC-A',
        'FSC-H',
        'FSC-W',
        'SSC-A',
        'SSC-H',
        'SSC-W',
        'Time'
    ]

    fluoro_indices = []
    fluoro_labels = []

    for i, label in enumerate(channel_labels):
        if label not in non_fluoro_channels:
            fluoro_indices.append(i)
            fluoro_labels.append(label)

    # Determine compensation object type
    if isinstance(compensation, str):
        # if a string, may be a file path or CSV text, we'll first test if file path exists
        # and if so try and open as CSV. If this fails, we'll test splitting the string by
        # new lines and commas. If this fails, raise a ValueError
        if os.path.isfile(compensation):
            fh = open(compensation)
            matrix_text = fh.read()
            fh.close()
        else:
            # may be a CSV string
            matrix_text = compensation

        matrix = convert_matrix_text_to_array(matrix_text, fluoro_labels, fluoro_indices)
    elif isinstance(compensation, Path):
        fh = compensation.open('r')
        matrix_text = fh.read()
        fh.close()

        matrix = convert_matrix_text_to_array(matrix_text, fluoro_labels, fluoro_indices)
    elif isinstance(compensation, np.ndarray):
        matrix = compensation
    else:
        raise ValueError("Compensation given is not a string or a NumPy array")

    # Make sure we have the correct number of rows (the header of matrix adds a row)
    if matrix.shape[0] > len(fluoro_labels) + 1:
        raise ValueError("Too many rows in compensation matrix")
    elif matrix.shape[0] < len(fluoro_labels) + 1:
        raise ValueError("Too few rows in compensation matrix")

    return matrix


def get_false_bounds(bool_array):
    diff = np.diff(np.hstack((0, bool_array, 0)))

    start = np.where(diff == 1)
    end = np.where(diff == -1)

    return start[0], end[0]


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.values.strides + (a.values.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def plot_channel(chan_events, label, subplot_ax, xform=False, bad_events=None):
    if xform:
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


def filter_anomalous_events(
        transformed_events,
        channel_labels,
        p_value_threshold=0.03,
        roll=10000,
        rng=None,
        ref_set_count=3,
        ref_size=10000,
        plot=False
):
    if rng is None:
        rng = np.random.RandomState()

    event_count = transformed_events.shape[0]

    # start with boolean array where True is a bad event, initially all set to False,
    # we will OR them for each channel
    bad_events = np.zeros(event_count, dtype=bool)

    for i, label in enumerate(channel_labels):
        if label == 'Time':
            continue

        chan_events = pd.Series(transformed_events[:, i])

        rolling_mean = chan_events.rolling(
            roll,
            min_periods=1,
            center=True
        ).mean()

        median = np.median(rolling_mean)

        # find absolute difference from the median of the moving average
        median_diff = np.abs(rolling_mean - median)

        # sort the differences and take a random sample of size=roll from the top 20%
        # TODO: add check for whether there are ~ 2x the events of roll size
        reference_indices = np.argsort(median_diff)

        # create reference sets, we'll average the p-values from these
        ref_sets = []
        for j in range(0, ref_set_count):
            ref_subsample_idx = rng.choice(int(event_count * 0.5), ref_size, replace=False)
            ref_sets.append(chan_events[reference_indices.values[ref_subsample_idx]])

        # calculate piece-wise KS test, we'll test every roll / 5 interval, cause
        # doing a true rolling window takes way too long
        strides = rolling_window(chan_events, roll)

        ks_x = []
        ks_y = []
        ks_y_stats = []

        test_idx = list(range(0, len(strides), int(roll / 5)))
        test_idx.append(len(strides) - 1)  # cover last stride, to get any remainder

        for test_i in test_idx:
            kss = []
            kss_stats = []

            for ref in ref_sets:
                # TODO: maybe this should be the Anderson-Darling test?
                test_result = stats.ks_2samp(ref, strides[test_i])
                kss.append(test_result.pvalue)
                kss_stats.append(test_result.statistic)

            ks_x.append(test_i)

            ks_y.append(np.mean(kss))  # TODO: should this be max or min?
            ks_y_stats.append(np.mean(kss_stats))  # TODO: this isn't used yet, should it?

        ks_y_roll = pd.Series(ks_y).rolling(
            3,
            min_periods=1,
            center=True
        ).mean()

        # interpolate our piecewise tests back to number of actual events
        interp_y = np.interp(range(0, event_count), ks_x, ks_y_roll)

        bad_events = np.logical_or(bad_events, interp_y < p_value_threshold)

        if plot:
            fig = pyplot.figure(figsize=(16, 12))
            ax = fig.add_subplot(4, 1, 1)

            plot_channel(chan_events, " - ".join([str(i + 1), label]), ax, xform=False)

            ax = fig.add_subplot(4, 1, 2)
            ax.set_title(
                "Median Difference",
                fontsize=16
            )
            ax.set_xlim([0, event_count])
            pyplot.plot(
                np.arange(0, event_count),
                median_diff,
                c='cornflowerblue',
                alpha=1.0,
                linewidth=1
            )

            ax = fig.add_subplot(4, 1, 3)
            ax.set_title(
                "KS Test p-value",
                fontsize=16
            )
            ax.set_xlim([0, event_count])
            ax.set_ylim([0, 1])
            pyplot.plot(
                ks_x,
                ks_y,
                c='cornflowerblue',
                alpha=0.6,
                linewidth=1
            )
            pyplot.plot(
                np.arange(0, event_count),
                interp_y,
                c='darkorange',
                alpha=1.0,
                linewidth=2
            )

            ax.axhline(p_value_threshold, linestyle='-', linewidth=1, c='coral')

            combined_refs = np.hstack(ref_sets)
            ref_y_min = combined_refs.min()
            ref_y_max = combined_refs.max()

            for ref_i, reference_events in enumerate(ref_sets):
                ax = fig.add_subplot(4, ref_set_count, (3 * ref_set_count) + ref_i + 1)
                ax.set_xlim([0, reference_events.shape[0]])
                ax.set_ylim([ref_y_min, ref_y_max])
                pyplot.scatter(
                    np.arange(0, reference_events.shape[0]),
                    reference_events,
                    s=2,
                    edgecolors='none'
                )

            fig.tight_layout()
            pyplot.show()

    return np.where(bad_events)[0]
