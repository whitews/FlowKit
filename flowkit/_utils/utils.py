"""
Utility functions for FlowKit
"""
import re
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot
import flowutils
# noinspection PyUnresolvedReferences, PyProtectedMember
from .. import _utils_c as utils_c
from .._utils import plot_utils


def _parse_multiline_matrix(matrix_text, fluoro_labels):
    # first, we must find a valid header line and we will require that the matrix
    # follows on the next lines, ignoring any additional lines before or after
    # the header contains labels matching the PnN value(FCS text field)
    # and may be tab or comma delimited
    # (spaces can't be delimiters b/c they are allowed in the PnN value)
    header = None
    header_line_index = None
    for i, line in enumerate(matrix_text):
        # header may begin with a '#' and some white space
        match = re.search('^#\\s*', line)
        if match is not None:
            line = line[match.end():]
        line_values = re.split('[\t,]', line)

        label_diff = set(fluoro_labels).symmetric_difference(line_values)

        if len(label_diff) != 0:
            # if any labels are missing or extra ones found, then not a valid header row
            continue
        else:
            header = line_values
            header_line_index = i
            break

    matrix_start = header_line_index + 1
    matrix_end = matrix_start + len(fluoro_labels)

    if len(matrix_text) < matrix_end:
        raise ValueError("Too few rows in compensation")

    matrix_text = matrix_text[matrix_start:matrix_end]
    matrix_array = []

    # convert the matrix text to numpy array
    for line in matrix_text:
        line_values = re.split('[\t,]', line)
        line_values = [float(value) for value in line_values]

        if len(line_values) > len(fluoro_labels):
            raise ValueError("Too many values in line: %s" % line)
        elif len(line_values) < len(fluoro_labels):
            raise ValueError("Too few values in line: %s" % line)
        else:
            matrix_array.append(line_values)

    matrix_array = np.array(matrix_array)

    return matrix_array, header


def _convert_matrix_text_to_array(matrix_text, fluoro_labels, fluoro_indices):
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

    if len(matrix_text) == 0:
        raise ValueError("matrix text appears to be empty")
    elif len(matrix_text) == 1:
        # probably a single-line CSV from FCS metadata
        matrix, header = flowutils.compensate.get_spill(matrix_text[0])
    else:
        matrix, header = _parse_multiline_matrix(matrix_text, fluoro_labels)

    label_diff = set(fluoro_labels).symmetric_difference(header)

    # re-order matrix according to provided fluoro label order
    idx_order = [header.index(fluoro_label) for fluoro_label in fluoro_labels]
    matrix = matrix[idx_order, :][:, idx_order]

    if len(label_diff) > 0:
        in_fcs_not_comp = []
        in_comp_not_fcs = []

        for label in label_diff:
            if label in fluoro_labels:
                in_fcs_not_comp.append(label)
            else:
                in_comp_not_fcs.append(label)

        error_message = "Matrix labels do not match given fluorescent labels"

        if len(in_fcs_not_comp) > 0:
            error_message = "\n".join(
                [
                    error_message,
                    "",
                    "Labels in FCS file not found in comp matrix (null channels?):",
                    ", ".join(in_fcs_not_comp),
                    "",
                    "Null channels can be specified when creating a Sample instance"
                ]
            )

        if len(in_comp_not_fcs) > 0:
            error_message = "\n".join(
                [
                    error_message,
                    "",
                    "Labels in comp matrix not found in FCS file (wrong matrix chosen?):",
                    ", ".join(in_comp_not_fcs)
                ]
            )

        raise ValueError(error_message)

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

    matrix_array = np.vstack([header_channel_numbers, matrix])

    return matrix_array


def parse_compensation_matrix(compensation, channel_labels, null_channels=None):
    """
    Returns a NumPy array with the compensation matrix where the first row are
    the indices of the fluorescent channels
    :param compensation: Compensation matrix: may be a NumPy array, a CSV file
        path, a pathlib Path object to a CSV or TSV file or a string of CSV
        text. If a string, both multi-line, traditional CSV, and the single
        line FCS spill formats are supported. If a NumPy array, we assume the
        columns are in the same order as the channel labels
    :param channel_labels: Channel labels from the FCS file's PnN fields, must be in
        the same order as they appear in the FCS file
    :param null_channels: Specify any empty channels that were collected and
        present in the channel_labels argument. These will be ignored when
        validating and creating the compensation matrix
    :return: Compensation matrix as NumPy array where header contains the
        channel numbers (not indices!)
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

    if null_channels is not None:
        non_fluoro_channels.extend(null_channels)

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

        matrix = _convert_matrix_text_to_array(matrix_text, fluoro_labels, fluoro_indices)

    elif isinstance(compensation, Path):
        fh = compensation.open('r')
        matrix_text = fh.read()
        fh.close()

        matrix = _convert_matrix_text_to_array(matrix_text, fluoro_labels, fluoro_indices)
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


def _rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.values.strides + (a.values.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


# TODO: somehow save anomalous event data in Sample class, maybe in some new class AnomData?
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
            # noinspection PyUnresolvedReferences
            ref_sets.append(chan_events[reference_indices.values[ref_subsample_idx]])

        # calculate piece-wise KS test, we'll test every roll / 5 interval, cause
        # doing a true rolling window takes way too long
        strides = _rolling_window(chan_events, roll)

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
                ks_stat, p_value = stats.ks_2samp(ref, strides[test_i])
                kss.append(p_value)
                kss_stats.append(ks_stat)

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

            plot_utils.plot_channel(chan_events, " - ".join([str(i + 1), label]), ax, xform=False)

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
            ref_y_min = np.min(combined_refs)
            ref_y_max = np.max(combined_refs)

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


def points_in_ellipsoid(
        ellipsoid_covariance_matrix,
        ellipsoid_means,
        ellipsoid_distance_square,
        points
):
    # we only take points that have already been filtered by the correct
    # columns (i.e. those columns that are included in the ellipsoid

    # First, subtract ellipse centers (consider the ellipse at the origin)
    points_translated = points - ellipsoid_means

    # Get the inverse covariance matrix
    ell_cov_mat_inv = np.linalg.inv(ellipsoid_covariance_matrix)

    # Matrix multiplication of translated points by inverse covariance matrix,
    # rotates the points instead of rotating the ellipse
    points_rot = np.dot(points_translated, ell_cov_mat_inv)
    points_rot = points_rot * points_translated

    # Points are inclusive if they are <= than the distance square
    # since boundary points are considered inclusive
    results = points_rot.sum(axis=1) <= ellipsoid_distance_square

    return results


def points_in_polygon(poly_vertices, points):
    """
    Determines whether points in an array are inside a polygon. Points on the
    edge of the polygon are considered inclusive. This function uses the
    winding number method and is robust to complex polygons with crossing
    boundaries, including the presence of 'holes' created by boundary crosses.

    :param poly_vertices: Polygon vertices (NumPy array of 2-D points)
    :param points: Points to test for polygon inclusion
    :return: List of boolean values for each point. True is inside polygon.
    """
    wind_counts = utils_c.points_in_polygon(poly_vertices, len(poly_vertices), points, len(points))
    return wind_counts % 2 != 0


def rotate_point_around_point(point, cov_mat, center_point=(0, 0)):
    # rotates point around center_point
    point_trans = np.array([point[0] - center_point[0], point[1] - center_point[1]])
    point_rot = np.dot(point_trans, cov_mat)
    point_untrans = point_rot + center_point

    return point_untrans
