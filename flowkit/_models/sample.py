import flowio
import os
from pathlib import Path
import io
from tempfile import TemporaryFile
import numpy as np
from flowkit._models.transforms import transforms
from flowkit._models.transforms.matrix import Matrix
from .. import _utils
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn
from bokeh.plotting import figure
from bokeh.layouts import gridplot
import warnings


class Sample(object):
    """
    Represents a single FCS sample from an FCS file, NumPy array or Pandas
    DataFrame.
    """
    def __init__(
            self,
            fcs_path_or_data,
            channel_labels=None,
            compensation=None,
            null_channel_list=None
    ):
        """
        Create a Sample instance

        :param fcs_path_or_data: FCS data, can be either:
                - a file path or file handle to an FCS file
                - a pathlib Path object
                - a FlowIO FlowData object
                - a NumPy array of FCS event data (must provide channel_labels)
                - a Pandas DataFrame containing FCS event data (channel labels as headers)
        :param channel_labels: A list of strings or a list of tuples to use for the channel
            labels. Required if fcs_path_or_data is a NumPy array
        :param compensation: Compensation matrix. Can be either:
                - a text string in CSV or TSV format
                - a string path to a CSV or TSV file
                - a pathlib Path object to a CSV or TSV file
        :param null_channel_list: List of PnN labels for channels that were collected
            but do not contain useful data. Note, this should only be used if there were
            truly no fluorochromes used targeting those detectors and the channels
            do not contribute to compensation.
        """
        # inspect our fcs_path_or_data argument
        if isinstance(fcs_path_or_data, str):
            # if a string, we only handle file paths, so try creating a FlowData object
            flow_data = flowio.FlowData(fcs_path_or_data)
        elif isinstance(fcs_path_or_data, io.IOBase):
            flow_data = flowio.FlowData(fcs_path_or_data)
        elif isinstance(fcs_path_or_data, Path):
            flow_data = flowio.FlowData(fcs_path_or_data.open('rb'))
        elif isinstance(fcs_path_or_data, flowio.FlowData):
            flow_data = fcs_path_or_data
        elif isinstance(fcs_path_or_data, np.ndarray):
            tmp_file = TemporaryFile()
            flowio.create_fcs(
                fcs_path_or_data.flatten().tolist(),
                channel_names=channel_labels,
                file_handle=tmp_file
            )

            flow_data = flowio.FlowData(tmp_file)
        else:
            raise ValueError("'fcs_path_or_data' is not a supported type")

        try:
            self.version = flow_data.header['version']
        except KeyError:
            self.version = None

        self.null_channels = null_channel_list
        self.event_count = flow_data.event_count
        self.channels = flow_data.channels
        self.pnn_labels = list()
        self.pns_labels = list()
        self.fluoro_indices = list()

        channel_gain = []
        channel_lin_log = []
        channel_range = []
        self.metadata = flow_data.text

        for n in sorted([int(k) for k in self.channels.keys()]):
            chan_label = self.channels[str(n)]['PnN']
            self.pnn_labels.append(chan_label)

            if 'p%dg' % n in self.metadata:
                channel_gain.append(float(self.metadata['p%dg' % n]))
            else:
                channel_gain.append(1.0)

            if 'p%dr' % n in self.metadata:
                channel_range.append(float(self.metadata['p%dr' % n]))
            else:
                channel_range.append(None)

            if 'p%de' % n in self.metadata:
                (decades, log0) = [
                    float(x) for x in self.metadata['p%de' % n].split(',')
                ]
                if log0 == 0 and decades != 0:
                    log0 = 1.0  # FCS std states to use 1.0 for invalid 0 value
                channel_lin_log.append((decades, log0))
            else:
                channel_lin_log.append((0.0, 0.0))

            if chan_label.lower()[:4] not in ['fsc-', 'ssc-', 'time']:
                self.fluoro_indices.append(n - 1)

            if 'PnS' in self.channels[str(n)]:
                self.pns_labels.append(self.channels[str(n)]['PnS'])
            else:
                self.pns_labels.append('')

        # Raw events need to be scaled according to channel gain, as well
        # as corrected for proper lin/log display
        # These are the only pre-processing we will do on raw events
        raw_events = np.reshape(
            np.array(flow_data.events, dtype=np.float),
            (-1, flow_data.channel_count)
        )

        # But first, we'll save the unprocessed events
        self._orig_events = raw_events.copy()

        for i, (decades, log0) in enumerate(channel_lin_log):
            if decades > 0:
                raw_events[:, i] = (10 ** (decades * raw_events[:, i] / channel_range[i])) * log0

        self._raw_events = raw_events / channel_gain
        self._comp_events = None
        self._transformed_events = None
        self.compensation = None
        self.transform = None
        self._subsample_count = None
        self._subsample_seed = None

        self.apply_compensation(compensation)

        # if filtering any events, save those in case they want to be retrieved
        self.negative_scatter_indices = None
        self.anomalous_indices = None
        self.subsample_indices = None

        try:
            self.acquisition_date = self.metadata['date']
        except KeyError:
            self.acquisition_date = None

        try:
            self.original_filename = self.metadata['fil']
        except KeyError:
            if isinstance(fcs_path_or_data, str):
                self.original_filename = os.path.basename(fcs_path_or_data)
            else:
                self.original_filename = None

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'v{self.version}, {self.original_filename}, '
            f'{len(self.pnn_labels)} channels, {self.event_count} events)'
        )

    def filter_negative_scatter(self, reapply_subsample=True):
        """
        Determines indices of negative scatter events, optionally re-subsample the Sample events afterward
        """
        scatter_indices = []
        for i, p in enumerate(self.pnn_labels):
            if p.lower()[:4] in ['fsc-', 'ssc-']:
                scatter_indices.append(i)

        is_neg = np.where(self._raw_events[:, scatter_indices] < 0)[0]

        self.negative_scatter_indices = is_neg

        if reapply_subsample and self._subsample_count is not None:
            self.subsample_indices(self._subsample_count, self._subsample_seed)

    def filter_anomalous_events(
            self,
            random_seed=1,
            p_value_threshold=0.03,
            ref_size=10000,
            channel_labels_or_numbers=None,
            reapply_subsample=True
    ):
        """
        Anomalous events are determined via Kolmogorov-Smirnov (KS) statistical
        test performed on each channel. The reference distribution is chosen based on
        the difference from the median.

        :param random_seed: Random seed used for initializing the anomaly detection routine. Default is 1
        :param p_value_threshold: Controls the sensitivity for anomalous event detection. The value is the p-value
            threshold for the KS test. A higher value will filter more events. Default is 0.03
        :param ref_size: The number of reference groups to sample from the 'stable' regions. Default is 3
        :param channel_labels_or_numbers: List of fluorescent channel labels or numbers (not indices)
            to evaluate for anomalous events. If None, then all fluorescent channels will be evaluated.
            Default is None
        :param reapply_subsample: Whether to re-subsample the Sample events after filtering. Default is True
        :return: None
        """
        rng = np.random.RandomState(seed=random_seed)

        logicle_xform = transforms.LogicleTransform(
            'my_xform',
            param_t=262144,
            param_w=1.0,
            param_m=4.5,
            param_a=0
        )
        xform_events = self._transform(logicle_xform)

        eval_indices = []
        eval_labels = []
        if channel_labels_or_numbers is not None:
            for label_or_num in channel_labels_or_numbers:
                c_idx = self.get_channel_index(label_or_num)
                eval_indices.append(c_idx)
        else:
            eval_indices = self.fluoro_indices

        for idx in eval_indices:
            eval_labels.append(self.pnn_labels[idx])

        anomalous_idx = _utils.filter_anomalous_events(
            xform_events[:, eval_indices],
            eval_labels,
            rng=rng,
            ref_set_count=3,
            p_value_threshold=p_value_threshold,
            ref_size=ref_size,
            plot=False
        )
        self.anomalous_indices = anomalous_idx

        if reapply_subsample and self._subsample_count is not None:
            self.subsample_indices(self._subsample_count, self._subsample_seed)

    def subsample_events(
            self,
            subsample_count=10000,
            random_seed=1
    ):
        """
        Returns a sub-sample of FCS raw events

        Returns NumPy array if sub-sampling succeeds
        Also updates self.subsample_indices

        :param subsample_count: Number of events to use as a sub-sample. If the number of
            events in the Sample is less than the requested sub-sample count, then the
            maximum number of available events is used for the sub-sample.
        :param random_seed: Random seed used for sub-sampling events
        """
        # get raw event count as it might be less than original event count
        # due to filtered negative scatter events
        raw_event_count = self._raw_events.shape[0]
        shuffled_indices = np.arange(raw_event_count)

        self._subsample_seed = random_seed
        rng = np.random.RandomState(seed=self._subsample_seed)

        bad_idx = np.empty(0)

        if self.negative_scatter_indices is not None:
            bad_idx = self.negative_scatter_indices

        if self.anomalous_indices is not None:
            bad_idx = np.unique(np.concatenate([bad_idx, self.anomalous_indices]))

        bad_count = bad_idx.shape[0]
        if bad_count > 0:
            shuffled_indices = np.delete(shuffled_indices, bad_idx)

        if (raw_event_count - bad_count) < subsample_count:
            # if total event count is less than requested subsample count,
            # sub-sample will be all events (minus negative scatter if filter is True)
            self._subsample_count = self.event_count - bad_count
        else:
            self._subsample_count = subsample_count

        # generate random indices for subsample
        # using a new RandomState with given seed
        rng.shuffle(shuffled_indices)

        self.subsample_indices = shuffled_indices[:self._subsample_count]

    def apply_compensation(self, compensation, comp_id='fcs'):
        """
        Applies given compensation matrix to Sample events. If any
        transformation has been applied, those events will be deleted.
        Compensated events can be retrieved afterward by calling
        `get_comp_events`.

        :param compensation: Compensation matrix, which can be a:
                - NumPy array
                - CSV file path
                - pathlib Path object to a CSV or TSV file
                - string of CSV text

            If a string, both multi-line traditional CSV, and the single
            line FCS spill formats are supported. If a NumPy array, we
            assume the columns are in the same order as the channel labels.
        :param comp_id: text ID for identifying compensation matrix
        :return: None
        """
        comp_labels = self.pnn_labels

        if compensation is not None:
            spill = _utils.parse_compensation_matrix(
                compensation,
                comp_labels,
                null_channels=self.null_channels
            )
            fluorochromes = [self.pns_labels[i] for i in self.fluoro_indices]
            detectors = [self.pnn_labels[i] for i in self.fluoro_indices]
            self.compensation = Matrix(comp_id, fluorochromes, detectors, spill[1:, :])
            self._transformed_events = None
            self._comp_events = self.compensation.apply(self)
        else:
            self.compensation = None

    def get_metadata(self):
        """
        Retrieve FCS metadata

        :return: Dictionary of FCS metadata
        """
        return self.metadata

    def get_orig_events(self, subsample=False):
        """
        Returns 'original' events, i.e. not pre-processed, compensated,
        or transformed.

        :param subsample: Whether to return all events or just the sub-sampled
            events. Default is False (all events)
        :return: NumPy array of original events
        """
        if subsample:
            return self._orig_events[self.subsample_indices]
        else:
            return self._orig_events

    def get_raw_events(self, subsample=False):
        """
        Returns 'raw' events that have been pre-processed to adjust for channel
        gain and lin/log display, but have not been compensated or transformed.

        :param subsample: Whether to return all events or just the sub-sampled
            events. Default is False (all events)
        :return: NumPy array of raw events
        """
        if subsample:
            return self._raw_events[self.subsample_indices]
        else:
            return self._raw_events

    def get_comp_events(self, subsample=False):
        """
        Returns compensated events, (not transformed)

        :param subsample: Whether to return all events or just the sub-sampled
            events. Default is False (all events)
        :return: NumPy array of compensated events or None if no compensation
            matrix has been applied.
        """
        if self._comp_events is None:
            warnings.warn(
                "No compensation has been applied, call 'compensate' method first.",
                UserWarning
            )
            return None

        if subsample:
            return self._comp_events[self.subsample_indices]
        else:
            return self._comp_events

    def get_transformed_events(self, subsample=False):
        """
        Returns transformed events. Note, if a compensation matrix has been
        applied then the events returned will be compensated and transformed.

        :param subsample: Whether to return all events or just the sub-sampled
            events. Default is False (all events)
        :return: NumPy array of transformed events or None if no transform
            has been applied.
        """
        if self._transformed_events is None:
            warnings.warn(
                "No transform has been applied, call a transform method first.",
                UserWarning
            )
            return None

        if subsample:
            return self._transformed_events[self.subsample_indices]
        else:
            return self._transformed_events

    def get_channel_number_by_label(self, label):
        """
        Returns the channel number for the given PnN label. Note, this is the
        channel number as defined in the FCS data (not the channel index), so
        the 1st channel's number is 1 (not 0).

        :param label: PnN label of a channel
        :return: Channel number (not index)
        """
        return self.pnn_labels.index(label) + 1

    def get_channel_index(self, channel_label_or_number):
        """
        Returns the channel index for the given PnN label. Note, this is
        different from the channel number. The 1st channel's index is 0 (not 1).

        :param channel_label_or_number: A channel's PnN label or number
        :return: Channel index
        """
        if isinstance(channel_label_or_number, str):
            index = self.get_channel_number_by_label(channel_label_or_number) - 1
        elif isinstance(channel_label_or_number, int):
            index = channel_label_or_number - 1
        else:
            raise ValueError("x_label_or_number must be a label string or channel number")

        return index

    def get_channel_data(self, channel_index, source='xform', subsample=False):
        """
        Returns a NumPy array of event data for the specified channel index.

        :param channel_index: Channel index for which data is returned
        :param source: 'raw', 'comp', 'xform' for whether the raw, compensated
            or transformed events will be returned
        :param subsample: Whether to return all events or just the sub-sampled
            events. Default is False (all events)
        :return: NumPy array of event data for the specified channel index
        """
        if subsample:
            idx = self.subsample_indices
        else:
            idx = np.arange(self.event_count)

        if source == 'xform':
            channel_data = self._transformed_events[idx, channel_index]
        elif source == 'comp':
            channel_data = self._comp_events[idx, channel_index]
        elif source == 'raw':
            channel_data = self._raw_events[idx, channel_index]
        else:
            raise ValueError("source must be one of 'raw', 'comp', or 'xform'")

        return channel_data

    def _transform(self, transform):
        if self._comp_events is not None:
            transformed_events = self._comp_events.copy()
        else:
            transformed_events = self._raw_events.copy()

        transformed_events[:, self.fluoro_indices] = transform.apply(
            transformed_events[:, self.fluoro_indices]
        )

        return transformed_events

    def apply_transform(self, transform):
        self._transformed_events = self._transform(transform)
        self.transform = transform

    def plot_contour(
            self,
            x_label_or_number,
            y_label_or_number,
            source='xform',
            subsample=False,
            plot_contour=True,
            plot_events=False,
            x_min=None,
            x_max=None,
            y_min=None,
            y_max=None,
            fig_size=(8, 8)
    ):
        """
        Returns a contour plot of the specified channel events, available
        as raw, compensated, or transformed data.

        :param x_label_or_number:  A channel's PnN label or number for x-axis
            data
        :param y_label_or_number: A channel's PnN label or number for y-axis
            data
        :param source: 'raw', 'comp', 'xform' for whether the raw, compensated
            or transformed events are used for plotting
        :param subsample: Whether to use all events for plotting or just the
            sub-sampled events. Default is False (all events). Plotting
            sub-sampled events can be much faster.
        :param plot_contour: Whether to display the contour lines. Default is True.
        :param plot_events: Whether to display the event data points in
            addition to the contours. Default is False.
        :param x_min: Lower bound of x-axis. If None, channel's min value will
            be used with some padding to keep events off the edge of the plot.
        :param x_max: Upper bound of x-axis. If None, channel's max value will
            be used with some padding to keep events off the edge of the plot.
        :param y_min: Lower bound of y-axis. If None, channel's min value will
            be used with some padding to keep events off the edge of the plot.
        :param y_max: Upper bound of y-axis. If None, channel's max value will
            be used with some padding to keep events off the edge of the plot.
        :param fig_size: Tuple of 2 values specifying the size of the returned
            figure. Values are in Matplotlib size units.
        :return: Matplotlib figure of the contour plot
        """
        x_index = self.get_channel_index(x_label_or_number)
        y_index = self.get_channel_index(y_label_or_number)

        x = self.get_channel_data(x_index, source=source, subsample=subsample)
        y = self.get_channel_data(y_index, source=source, subsample=subsample)

        x_min, x_max = _utils.calculate_extent(x, d_min=x_min, d_max=x_max, pad=0.02)
        y_min, y_max = _utils.calculate_extent(y, d_min=y_min, d_max=y_max, pad=0.02)

        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_title(self.original_filename)

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_xlabel(self.pnn_labels[x_index])
        ax.set_ylabel(self.pnn_labels[y_index])

        if plot_events:
            seaborn.scatterplot(
                x,
                y,
                palette=_utils.new_jet,
                legend=False,
                s=5,
                linewidth=0,
                alpha=0.4
            )

        if plot_contour:
            seaborn.kdeplot(
                x,
                y,
                bw='scott',
                cmap=_utils.new_jet,
                linewidths=2,
                alpha=1
            )

        return fig

    def plot_scatter(
            self,
            x_label_or_number,
            y_label_or_number,
            source='xform',
            subsample=False,
            color_density=True,
            x_min=None,
            x_max=None,
            y_min=None,
            y_max=None
    ):
        """
        Returns an interactive scatter plot for the specified channel data.

        :param x_label_or_number:  A channel's PnN label or number for x-axis
            data
        :param y_label_or_number: A channel's PnN label or number for y-axis
            data
        :param source: 'raw', 'comp', 'xform' for whether the raw, compensated
            or transformed events are used for plotting
        :param subsample: Whether to use all events for plotting or just the
            sub-sampled events. Default is False (all events). Plotting
            sub-sampled events can be much faster.
        :param color_density: Whether to color the events by density, similar
            to a heat map. Default is True.
        :param x_min: Lower bound of x-axis. If None, channel's min value will
            be used with some padding to keep events off the edge of the plot.
        :param x_max: Upper bound of x-axis. If None, channel's max value will
            be used with some padding to keep events off the edge of the plot.
        :param y_min: Lower bound of y-axis. If None, channel's min value will
            be used with some padding to keep events off the edge of the plot.
        :param y_max: Upper bound of y-axis. If None, channel's max value will
            be used with some padding to keep events off the edge of the plot.
        :return: A Bokeh Figure object containing the interactive scatter plot.
        """
        # First, sanity check on requested source type
        if source == 'xform' and self._transformed_events is None:
            raise AttributeError(
                "Transformed events were requested but do not exist.\n"
                "Have you called a transform method?"
            )

        x_index = self.get_channel_index(x_label_or_number)
        y_index = self.get_channel_index(y_label_or_number)

        x = self.get_channel_data(x_index, source=source, subsample=subsample)
        y = self.get_channel_data(y_index, source=source, subsample=subsample)

        x_min, x_max = _utils.calculate_extent(x, d_min=x_min, d_max=x_max, pad=0.02)
        y_min, y_max = _utils.calculate_extent(y, d_min=y_min, d_max=y_max, pad=0.02)

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

        colors_array = _utils.new_jet(colors.Normalize()(z))
        z_colors = [
            "#%02x%02x%02x" % (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in colors_array
        ]

        tools = "crosshair,pan,zoom_in,zoom_out,box_zoom,undo,redo,reset,save,"
        p = figure(
            tools=tools,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            title=self.original_filename
        )
        p.title.align = 'center'

        p.xaxis.axis_label = self.pnn_labels[x_index]
        p.yaxis.axis_label = self.pnn_labels[y_index]

        if y_max > x_max:
            radius_dimension = 'y'
            radius = 0.003 * y_max
        else:
            radius_dimension = 'x'
            radius = 0.003 * x_max

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

    def plot_scatter_matrix(
            self,
            source='xform',
            subsample=False,
            channel_labels_or_numbers=None,
            color_density=False,
            plot_height=256,
            plot_width=256
    ):
        """
        Returns an interactive scatter plot matrix for all channel combinations
        except for the Time channel.

        :param source: 'raw', 'comp', 'xform' for whether the raw, compensated
            or transformed events are used for plotting
        :param subsample: Whether to use all events for plotting or just the
            sub-sampled events. Default is False (all events). Plotting
            sub-sampled events can be much faster.
        :param channel_labels_or_numbers: List of channel PnN labels or channel
            numbers to use for the scatter plot matrix. If None, then all
            channels will be plotted (except Time).
        :param color_density: Whether to color the events by density, similar
            to a heat map. Default is False.
        :param plot_height: Height of plot in pixels (screen units)
        :param plot_width: Width of plot in pixels (screen units)
        :return: A Bokeh Figure object containing the interactive scatter plot
            matrix.
        """
        plots = []
        channels = []

        if channel_labels_or_numbers is None:
            channels = self.pnn_labels
        else:
            for c in channel_labels_or_numbers:
                c_index = self.get_channel_index(c)
                c_label = self.pnn_labels[c_index]

                if c_label not in channels:
                    channels.append(c_label)

        for channel_y in channels:
            if channel_y == 'Time':
                continue
            row = []
            for channel_x in channels:
                if channel_x == 'Time':
                    continue

                plot = self.plot_scatter(
                    channel_x,
                    channel_y,
                    source=source,
                    subsample=subsample,
                    color_density=color_density
                )
                plot.height = plot_height
                plot.width = plot_width
                row.append(plot)
            plots.append(row)

        grid = gridplot(plots)

        return grid

    def plot_histogram(
            self,
            channel_label_or_number,
            source='xform',
            subsample=False,
            bins=None,
            x_min=None,
            x_max=None,
            fig_size=(15, 7)
    ):
        """
        Returns a histogram plot of the specified channel events, available
        as raw, compensated, or transformed data. Plot also contains a curve
        of the gaussian kernel density estimate.

        :param channel_label_or_number:  A channel's PnN label or number to use
            for plotting the histogram
        :param source: 'raw', 'comp', 'xform' for whether the raw, compensated
            or transformed events are used for plotting
        :param subsample: Whether to use all events for plotting or just the
            sub-sampled events. Default is False (all events). Plotting
            sub-sampled events can be much faster.
        :param bins: Number of bins to use for the histogram. If None, the
            number of bins is determined by the Freedman-Diaconis rule.
        :param x_min: Lower bound of x-axis. If None, channel's min value will
            be used with some padding to keep events off the edge of the plot.
        :param x_max: Upper bound of x-axis. If None, channel's max value will
            be used with some padding to keep events off the edge of the plot.
        :param fig_size: Tuple of 2 values specifying the size of the returned
            figure. Values are in Matplotlib size units.
        :return: Matplotlib figure of the histogram plot with KDE curve.
        """

        channel_index = self.get_channel_index(channel_label_or_number)
        channel_data = self.get_channel_data(channel_index, source=source, subsample=subsample)

        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_title(self.original_filename)

        if x_min is None:
            x_min = channel_data.min()
        if x_max is None:
            x_max = channel_data.max()

        ax.set_xlim([x_min, x_max])
        ax.set_xlabel(self.pnn_labels[channel_index])

        seaborn.distplot(
            channel_data,
            hist_kws=dict(edgecolor="w", linewidth=1),
            label=self.pnn_labels[channel_index],
            bins=bins
        )

        return fig

    def export(
            self,
            filename,
            source='xform',
            exclude=None,
            subsample=False,
            directory=None
    ):
        """
        Export Sample event data to either a new FCS file or a CSV file. Format determined by filename extension.

        :param filename: Text string to use for the exported file name.
        :param source: 'raw', 'comp', 'xform' for whether the raw, compensated
            or transformed events are used for exporting
        :param exclude: Specifies whether to exclude events. Options are 'good', 'bad', or None.
            'bad' excludes neg. scatter or anomalous, 'good' will export the bad events.
            Default is None (exports all events)
        :param subsample: Whether to export all events or just the
            sub-sampled events. Default is False (all events).
        :param directory: Directory path where the CSV will be saved
        :return: None
        """
        if directory is not None:
            output_path = os.path.join(directory, filename)
        else:
            output_path = filename

        if subsample:
            idx = np.zeros(self.event_count, np.bool)
            idx[self.subsample_indices] = True
        else:
            # include all events to start with
            idx = np.ones(self.event_count, np.bool)

        if exclude == 'bad':
            idx[self.anomalous_indices] = False
        elif exclude == 'good':
            good_idx = np.zeros(self.event_count, np.bool)
            good_idx[self.anomalous_indices] = True
            idx = np.logical_and(idx, good_idx)

        if source == 'xform':
            events = self._transformed_events[idx, :]
        elif source == 'comp':
            events = self._comp_events[idx, :]
        elif source == 'raw':
            events = self._raw_events[idx, :]
        elif source == 'orig':
            events = self._orig_events[idx, :]
        else:
            raise ValueError("source must be one of 'raw', 'comp', or 'xform'")

        ext = os.path.splitext(filename)[-1]

        if ext == 'csv':
            np.savetxt(
                output_path,
                events,
                delimiter=',',
                header=",".join(self.pnn_labels),
                comments=''
            )
        elif ext == 'fcs':
            fh = open(output_path, 'wb')

            flowio.create_fcs(
                events.flatten().tolist(),
                channel_names=self.pnn_labels,
                opt_channel_names=self.pns_labels,
                file_handle=fh
            )
            fh.close()
