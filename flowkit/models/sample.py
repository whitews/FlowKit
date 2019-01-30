import flowio
import flowutils
import os
from pathlib import Path
import io
from tempfile import TemporaryFile
import numpy as np
from flowkit import utils
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
            subsample_count=10000,
            random_seed=1,
            filter_negative_scatter=False,
            filter_anomalous_events=False,
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
        :param subsample_count: Number of events to use as a sub-sample. If None, then no
            sub-sampling is performed. If the number of events in the Sample is less than the
            requested sub-sample count, then the maximum number of available events is used
            for the sub-sample.
        :param random_seed: Random seed used for sub-sampling events
        :param filter_negative_scatter: If True, negative scatter events are omitted from
            the sub-sample. Only used for sub-sampling.
        :param filter_anomalous_events: If True, anomalous events are omitted from the
            sub-sample. Anomalous events are determined via Kolmogorov-Smirnov statistical
            test performed on each channel. The reference distribution is chosen based on
            the difference from the median.
        :param null_channel_list: List of PnN labels for channels that were collected
            but do not contain useful data. Note, this should only be used if there were
            truly no fluorochromes used targeting those detectors and the channels
            do not contribute to compensation.
        """
        # inspect our fcs_path_or_data argument
        self._flow_data = None
        if isinstance(fcs_path_or_data, str):
            # if a string, we only handle file paths, so try creating a FlowData object
            self._flow_data = flowio.FlowData(fcs_path_or_data)
        elif isinstance(fcs_path_or_data, io.IOBase):
            self._flow_data = flowio.FlowData(fcs_path_or_data)
        elif isinstance(fcs_path_or_data, Path):
            self._flow_data = flowio.FlowData(fcs_path_or_data.open('rb'))
        elif isinstance(fcs_path_or_data, flowio.FlowData):
            self._flow_data = fcs_path_or_data
        elif isinstance(fcs_path_or_data, np.ndarray):
            tmp_file = TemporaryFile()
            flowio.create_fcs(
                fcs_path_or_data.flatten().tolist(),
                channel_names=channel_labels,
                file_handle=tmp_file
            )

            self._flow_data = flowio.FlowData(tmp_file)

        try:
            self.version = self._flow_data.header['version']
        except KeyError:
            self.version = None

        self.null_channels = null_channel_list
        self.event_count = self._flow_data.event_count
        self.channels = self._flow_data.channels
        self.pnn_labels = list()
        self.pns_labels = list()
        self.fluoro_indices = list()

        channel_gain = []
        channel_lin_log = []
        channel_range = []

        for n in sorted([int(k) for k in self.channels.keys()]):
            chan_label = self.channels[str(n)]['PnN']
            self.pnn_labels.append(chan_label)

            if 'p%dg' % n in self._flow_data.text:
                channel_gain.append(float(self._flow_data.text['p%dg' % n]))
            else:
                channel_gain.append(1.0)

            if 'p%dr' % n in self._flow_data.text:
                channel_range.append(float(self._flow_data.text['p%dr' % n]))
            else:
                channel_range.append(None)

            if 'p%de' % n in self._flow_data.text:
                (decades, log0) = [
                    float(x) for x in self._flow_data.text['p%de' % n].split(',')
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
            np.array(self._flow_data.events, dtype=np.float),
            (-1, self._flow_data.channel_count)
        )

        for i, (decades, log0) in enumerate(channel_lin_log):
            if decades > 0:
                raw_events[:, i] = (10 ** (decades * raw_events[:, i] / channel_range[i])) * log0

        self._raw_events = raw_events / channel_gain
        self._comp_events = None
        self._transformed_events = None  # TODO: should save transform settings
        self.compensation = None

        self.apply_compensation(compensation)

        # if filtering anomalous events, save those in case they want to be retrieved
        self.anomalous_indices = None

        # Save sub-sampled indices if requested
        if subsample_count is not None:
            self.subsample_indices = self._generate_subsample(
                subsample_count,
                random_seed,
                filter_negative_scatter=filter_negative_scatter,
                filter_anomalous_events=filter_anomalous_events  # will store anomalous events
            )
        else:
            self.subsample_indices = None

        try:
            self.acquisition_date = self._flow_data.text['date']
        except KeyError:
            self.acquisition_date = None

        try:
            self.original_filename = self._flow_data.text['fil']
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

    def _negative_scatter_indices(self):
        """Returns indices of negative scatter events"""
        scatter_indices = []
        for i, p in enumerate(self.pnn_labels):
            if p.lower()[:4] in ['fsc-', 'ssc-']:
                scatter_indices.append(i)

        is_neg = np.where(self._raw_events[:, scatter_indices] < 0)[0]

        return is_neg

    def _generate_subsample(
            self,
            subsample_count,
            random_seed,
            filter_negative_scatter=True,
            filter_anomalous_events=True
    ):
        """
        Returns a sub-sample of FCS raw events

        Returns NumPy array if sub-sampling succeeds
        Also updates self.subsample_indices
        """
        # get raw event count as it might be less than original event count
        # due to filtered negative scatter events
        raw_event_count = self._raw_events.shape[0]
        shuffled_indices = np.arange(raw_event_count)
        neg_scatter_idx = np.empty(0)
        anomalous_idx = np.empty(0)
        rng = np.random.RandomState(seed=random_seed)

        if filter_negative_scatter:
            neg_scatter_idx = self._negative_scatter_indices()

        if filter_anomalous_events:
            anomalous_idx = utils.filter_anomalous_events(
                flowutils.transforms.asinh(
                    self._raw_events,
                    self.fluoro_indices,
                    pre_scale=0.01
                ),
                self.pnn_labels,
                rng=rng,
                ref_set_count=3,
                plot=False
            )
            self.anomalous_indices = anomalous_idx

        bad_idx = np.unique(np.concatenate([neg_scatter_idx, anomalous_idx]))
        bad_count = bad_idx.shape[0]
        if bad_count > 0:
            shuffled_indices = np.delete(shuffled_indices, bad_idx)

        if (raw_event_count - bad_count) < subsample_count:
            # if total event count is less than requested subsample count,
            # sub-sample will be all events (minus negative scatter if filter is True)
            subsample_count = self.event_count - bad_count

        # generate random indices for subsample
        # using a new RandomState with given seed
        rng.shuffle(shuffled_indices)

        return shuffled_indices[:subsample_count]

    def _compensate(self):
        """
        Applies compensation to sample events. If self.compensation is None, the identity
        matrix is assumed.

        Saves NumPy array of compensated events to self._comp_events
        """
        # self.compensate has headers for the channel numbers, but
        # flowutils compensate() takes the plain matrix and indices as
        # separate arguments
        # (also note channel #'s vs indices)
        if self.compensation is not None:
            indices = self.compensation[0, :]  # headers are channel #'s
            indices = [int(i - 1) for i in indices]
            comp_matrix = self.compensation[1:, :]  # just the matrix
            self._comp_events = flowutils.compensate.compensate(
                self._raw_events,
                comp_matrix,
                indices
            )
        else:
            # assume identity matrix
            self._comp_events = self._raw_events.copy()

    def apply_compensation(self, compensation):
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
        :return: None
        """
        comp_labels = self.pnn_labels

        if compensation is not None:
            self.compensation = utils.parse_compensation_matrix(
                compensation,
                comp_labels,
                null_channels=self.null_channels
            )
        else:
            self.compensation = None

        self._transformed_events = None
        self._compensate()

    def get_metadata(self):
        """
        Retrieve FCS metadata

        :return: Dictionary of FCS metadata
        """
        return self._flow_data.text

    def get_raw_events(self, subsample=False):
        """
        Returns 'raw' events, i.e. not compensated or transformed.

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

    def apply_logicle_transform(self, logicle_t=262144, logicle_w=0.5):
        """
        Applies logicle transform to compensated data

        Retrieve transformed data via `get_transformed_events`
        """
        # only transform fluorescent channels
        self._transformed_events = flowutils.transforms.logicle(
            self._comp_events,
            self.fluoro_indices,
            t=logicle_t,
            w=logicle_w
        )

    def apply_asinh_transform(self, pre_scale=0.01):
        """
        Applies inverse hyperbolic sine transform on compensated events

        By default, the compensated data will be transformed and the default
        pre-scale factor is 0.01

        Retrieve transformed data via `get_transformed_events`
        """
        # only transform fluorescent channels
        self._transformed_events = flowutils.transforms.asinh(
            self._comp_events,
            self.fluoro_indices,
            pre_scale=pre_scale
        )

    def plot_contour(
            self,
            x_label_or_number,
            y_label_or_number,
            source='xform',
            subsample=False,
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

        # determine padding to keep min/max events off the edge
        pad_x = max(abs(x.min()), abs(x.max())) * 0.02
        pad_y = max(abs(y.min()), abs(y.max())) * 0.02

        if x_min is None:
            x_min = x.min() - pad_x
        if x_max is None:
            x_max = x.max() + pad_x
        if y_min is None:
            y_min = y.min() - pad_y
        if y_max is None:
            y_max = y.max() + pad_y

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
                palette=utils.new_jet,
                legend=False,
                s=5,
                linewidth=0,
                alpha=0.4
            )

        seaborn.kdeplot(
            x,
            y,
            bw='scott',
            cmap=utils.new_jet,
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

        # determine padding to keep min/max events off the edge,
        # but only if user didn't specify the limits
        pad_x = max(abs(x.min()), abs(x.max())) * 0.02
        pad_y = max(abs(y.min()), abs(y.max())) * 0.02

        if x_min is None:
            x_min = x.min() - pad_x
        if x_max is None:
            x_max = x.max() + pad_x
        if y_min is None:
            y_min = y.min() - pad_y
        if y_max is None:
            y_max = y.max() + pad_y

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

        colors_array = utils.new_jet(colors.Normalize()(z))
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
            color_density=False
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
                plot.height = 196
                plot.width = 196
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

    def export_csv(
            self,
            source='xform',
            subsample=False,
            filename=None,
            directory=None
    ):
        """
        Export event data to a CSV file.

        :param source: 'raw', 'comp', 'xform' for whether the raw, compensated
            or transformed events are used for exporting
        :param subsample: Whether to export all events or just the
            sub-sampled events. Default is False (all events).
        :param filename: Text string to use for the exported file name. If
            None, the FCS file's original file name will be used (if present).
        :param directory: Directory path where the CSV will be saved
        :return: None
        """
        if self.original_filename is None and filename is None:
            raise(
                ValueError(
                    "Sample has no original filename, please provide a 'filename' argument"
                )
            )
        elif filename is None:
            filename = self.original_filename

        if directory is not None:
            output_path = os.path.join(directory, filename)
        else:
            output_path = filename

        header = ",".join(self.pnn_labels)

        if subsample:
            idx = self.subsample_indices
        else:
            idx = np.arange(self.event_count)

        if source == 'xform':
            np.savetxt(
                output_path,
                self._transformed_events[idx, :],
                delimiter=',',
                header=header,
                comments=''
            )
        elif source == 'comp':
            np.savetxt(
                output_path,
                self._comp_events[idx, :],
                delimiter=',',
                header=header,
                comments=''
            )
        elif source == 'raw':
            np.savetxt(
                output_path,
                self._raw_events[idx, :],
                delimiter=',',
                header=header,
                comments=''
            )
        else:
            raise ValueError("source must be one of 'raw', 'comp', or 'xform'")

    def export_fcs(
            self,
            source='xform',
            subsample=False,
            filename=None,
            directory=None
    ):
        """
        Export event data to a new FCS file.

        :param source: 'raw', 'comp', 'xform' for whether the raw, compensated
            or transformed events are used for exporting
        :param subsample: Whether to export all events or just the
            sub-sampled events. Default is False (all events).
        :param filename: Text string to use for the exported file name. If
            None, the FCS file's original file name will be used (if present).
        :param directory: Directory path where the FCS file will be saved
        :return: None
        """
        if self.original_filename is None and filename is None:
            raise(
                ValueError(
                    "Sample has no original filename, please provide a 'filename' argument"
                )
            )
        elif filename is None:
            filename = self.original_filename

        if directory is not None:
            output_path = os.path.join(directory, filename)
        else:
            output_path = filename

        if subsample:
            idx = self.subsample_indices
        else:
            idx = np.arange(self.event_count)

        if source == 'xform':
            events = self._transformed_events[idx, :]
        elif source == 'comp':
            events = self._comp_events[idx, :]
        elif source == 'raw':
            events = self._raw_events[idx, :]
        else:
            raise ValueError("source must be one of 'raw', 'comp', or 'xform'")

        fh = open(output_path, 'wb')

        flowio.create_fcs(
            events.flatten().tolist(),
            channel_names=self.pnn_labels,
            opt_channel_names=self.pns_labels,
            file_handle=fh
        )

        fh.close()

    def export_anomalous_fcs(self, source='xform', filename=None, directory=None):
        """
        Export anomalous event data to a new FCS file.

        :param source: 'raw', 'comp', 'xform' for whether the raw, compensated
            or transformed events are used for exporting
        :param filename: Text string to use for the exported file name. If
            None, the FCS file's original file name will be used (if present).
        :param directory: Directory path where the FCS file will be saved
        :return: None
        """
        if self.original_filename is None and filename is None:
            raise(
                ValueError(
                    "Sample has no original filename, please provide a 'filename' argument"
                )
            )
        elif filename is None:
            filename = self.original_filename

        if directory is not None:
            output_path = os.path.join(directory, filename)
        else:
            output_path = filename

        if source == 'xform':
            events = self._transformed_events[self.anomalous_indices, :]
        elif source == 'comp':
            events = self._comp_events[self.anomalous_indices, :]
        elif source == 'raw':
            events = self._raw_events[self.anomalous_indices, :]
        else:
            raise ValueError("source must be one of 'raw', 'comp', or 'xform'")

        fh = open(output_path, 'wb')

        flowio.create_fcs(
            events.flatten().tolist(),
            channel_names=self.pnn_labels,
            opt_channel_names=self.pns_labels,
            file_handle=fh
        )

        fh.close()
