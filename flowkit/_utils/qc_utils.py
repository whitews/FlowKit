import numpy as np
import pandas as pd
from matplotlib import pyplot
from scipy import stats
from .._utils import plot_utils


def _rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.values.strides + (a.values.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


# TODO: somehow save anomalous event data in Sample class, maybe in some new class AnomalyData?
def filter_anomalous_events(
        transformed_events,
        channel_labels,
        p_value_threshold=0.03,
        rolling_window=10000,
        rng=None,
        ref_set_count=3,
        ref_size=10000,
        plot=False
):
    """
    Identifies anomalous events from given event data using the Kolmogorov-Smirnov statistic.
    Ideally, the events given are transformed.

    :param transformed_events: NumPy array of transformed event data
    :param channel_labels: list of channel labels (count must match the number of columns in transformed_events).
    :param p_value_threshold: The p-value above which events will be marked as anomalous. Default is 0.03.
    :param rolling_window: Size of the moving window. Default is 10,000 events.
    :param rng: an instance of NumPy RandomState. If None, a new RandomState is created. Default is None.
    :param ref_set_count: The number of reference sets from which the base p-values are calculated. Default is 3.
    :param ref_size: The size (in events) of each reference set. Default is 10,000 events.
    :param plot: Whether to plot graphs of the passing/anomalous events. Default is False
    :return:
    """
    if rng is None:
        rng = np.random.RandomState()

    event_count = transformed_events.shape[0]

    # start with boolean array where True is a bad event, initially all set to False,
    # we will OR them for each channel
    bad_events = np.zeros(event_count, dtype=bool)

    for i, label in enumerate(channel_labels):
        if label == 'Time':
            continue

        channel_events = pd.Series(transformed_events[:, i])

        rolling_mean = channel_events.rolling(
            rolling_window,
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
            ref_sets.append(channel_events[reference_indices.values[ref_subsample_idx]])

        # calculate piece-wise KS test, we'll test every roll / 5 interval, cause
        # doing a true rolling window takes way too long
        strides = _rolling_window(channel_events, rolling_window)

        ks_x = []
        ks_y = []
        ks_y_stats = []

        test_idx = list(range(0, len(strides), int(rolling_window / 5)))
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

            plot_utils.plot_channel(channel_events, " - ".join([str(i + 1), label]), ax, xform=False)

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
