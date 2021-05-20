"""
Sample class
"""

import flowio
import os
from pathlib import Path
import io
from tempfile import TemporaryFile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from bokeh.layouts import gridplot
from bokeh.models import Title
import warnings
# noinspection PyProtectedMember
from .._models.transforms import _transforms
# noinspection PyProtectedMember
from .._models.transforms._matrix import Matrix
from .._utils import plot_utils, qc_utils


class Sample(object):
    """
    Represents a single FCS sample from an FCS file, NumPy array or Pandas
    DataFrame.

    :param fcs_path_or_data: FCS data, can be either:

        - a file path or file handle to an FCS file
        - a pathlib Path object
        - a FlowIO FlowData object
        - a NumPy array of FCS event data (must provide channel_labels)
        - a Pandas DataFrame containing FCS event data (channel labels as column labels)

    :param channel_labels: A list of strings or a list of tuples to use for the channel
        labels. Required if fcs_path_or_data is a NumPy array

    :param compensation: Compensation matrix, which can be a:

        - Matrix instance
        - NumPy array
        - CSV file path
        - pathlib Path object to a CSV or TSV file
        - string of CSV text

    :param null_channel_list: List of PnN labels for channels that were collected
        but do not contain useful data. Note, this should only be used if there were
        truly no fluorochromes used targeting those detectors and the channels
        do not contribute to compensation.
    """
    def __init__(
            self,
            fcs_path_or_data,
            channel_labels=None,
            compensation=None,
            null_channel_list=None,
            ignore_offset_error=False
    ):
        """
        Create a Sample instance
        """
        # inspect our fcs_path_or_data argument
        if isinstance(fcs_path_or_data, str):
            # if a string, we only handle file paths, so try creating a FlowData object
            flow_data = flowio.FlowData(
                fcs_path_or_data,
                ignore_offset_error=ignore_offset_error
            )
        elif isinstance(fcs_path_or_data, io.IOBase):
            flow_data = flowio.FlowData(
                fcs_path_or_data,
                ignore_offset_error=ignore_offset_error
            )
        elif isinstance(fcs_path_or_data, Path):
            flow_data = flowio.FlowData(
                fcs_path_or_data.open('rb'),
                ignore_offset_error=ignore_offset_error
            )
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
        elif isinstance(fcs_path_or_data, pd.DataFrame):
            tmp_file = TemporaryFile()

            # Handle MultiIndex columns since that is what the as_dataframe method creates.
            if fcs_path_or_data.columns.nlevels > 1:
                pnn_labels = fcs_path_or_data.columns.get_level_values(0)
                pns_labels = fcs_path_or_data.columns.get_level_values(1)
            else:
                pnn_labels = fcs_path_or_data.columns
                pns_labels = None

            flowio.create_fcs(
                fcs_path_or_data.values.flatten().tolist(),
                channel_names=pnn_labels,
                file_handle=tmp_file,
                opt_channel_names=pns_labels
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
        self.scatter_indices = list()
        self.time_index = None

        channel_gain = []
        channel_lin_log = []
        channel_range = []
        self.metadata = flow_data.text

        for n in sorted([int(k) for k in self.channels.keys()]):
            channel_label = self.channels[str(n)]['PnN']
            self.pnn_labels.append(channel_label)

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

            if channel_label.lower()[:4] not in ['fsc-', 'ssc-', 'time']:
                self.fluoro_indices.append(n - 1)
            elif channel_label.lower()[:4] in ['fsc-', 'ssc-']:
                self.scatter_indices.append(n - 1)
            elif channel_label.lower() == 'time':
                self.time_index = n - 1

            if 'PnS' in self.channels[str(n)]:
                self.pns_labels.append(self.channels[str(n)]['PnS'])
            else:
                self.pns_labels.append('')

        self._flowjo_pnn_labels = [label.replace('/', '_') for label in self.pnn_labels]

        # Start processing the event data. First, we'll save the unprocessed events
        self._orig_events = np.reshape(
            np.array(flow_data.events, dtype=float),
            (-1, flow_data.channel_count)
        )

        # Event data must be scaled according to channel gain, as well
        # as corrected for proper lin/log display, and the time channel
        # scaled by the 'timestep' keyword value (if present).
        # This is the only pre-processing we will do on raw events
        raw_events = self._orig_events.copy()

        # Note: The time channel is scaled by the timestep (if present),
        # but should not be scaled by any gain value present in PnG.
        # It seems common for cytometers to include a gain value for the
        # time channel that matches the fluoro channels. Not sure why
        # they do this but it makes no sense to have an amplifier gain
        # on the time data. Here, we set any time gain to 1.0.
        if self.time_index is not None:
            channel_gain[self.time_index] = 1.0

        if 'timestep' in self.metadata and self.time_index is not None:
            time_step = float(self.metadata['timestep'])
            raw_events[:, self.time_index] = raw_events[:, self.time_index] * time_step

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

        if compensation is not None:
            self.apply_compensation(compensation)

        # if filtering any events, save those in case they want to be retrieved
        self.negative_scatter_indices = None
        self.anomalous_indices = None
        self.subsample_indices = None

        try:
            self.acquisition_date = self.metadata['date']
        except KeyError:
            self.acquisition_date = None

        # TODO: Allow user to set some sort of Sample ID or the orig filename,
        #       would be useful for Samples created from data arrays or if
        #       2 FCS files had the same file name.
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

        :param reapply_subsample: Whether to re-subsample the Sample events after filtering. Default is True
        """
        scatter_indices = []
        for i, p in enumerate(self.pnn_labels):
            if p.lower()[:4] in ['fsc-', 'ssc-']:
                scatter_indices.append(i)

        is_neg = np.where(self._raw_events[:, scatter_indices] < 0)[0]

        self.negative_scatter_indices = is_neg

        if reapply_subsample and self._subsample_count is not None:
            self.subsample_events(self._subsample_count, self._subsample_seed)

    def filter_anomalous_events(
            self,
            random_seed=1,
            p_value_threshold=0.03,
            ref_size=10000,
            channel_labels_or_numbers=None,
            reapply_subsample=True,
            plot=False
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
        :param plot: Whether to plot the intermediate data for the provided channel labels
        :return: None
        """
        rng = np.random.RandomState(seed=random_seed)

        logicle_xform = _transforms.LogicleTransform(
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

        anomalous_idx = qc_utils.filter_anomalous_events(
            xform_events[:, eval_indices],
            eval_labels,
            rng=rng,
            ref_set_count=3,
            p_value_threshold=p_value_threshold,
            ref_size=ref_size,
            plot=plot
        )
        self.anomalous_indices = anomalous_idx

        if reapply_subsample and self._subsample_count is not None:
            self.subsample_events(self._subsample_count, self._subsample_seed)

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

        bad_idx = np.empty(0, dtype=int)

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

    def apply_compensation(self, compensation, comp_id='custom_spill'):
        """
        Applies given compensation matrix to Sample events. If any
        transformation has been applied, those events will be deleted.
        Compensated events can be retrieved afterward by calling
        `get_comp_events`.

        :param compensation: Compensation matrix, which can be a:

                - Matrix instance
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
        if isinstance(compensation, Matrix):
            self.compensation = compensation
            self._comp_events = self.compensation.apply(self)
        elif compensation is not None:
            detectors = [self.pnn_labels[i] for i in self.fluoro_indices]
            fluorochromes = [self.pns_labels[i] for i in self.fluoro_indices]
            self.compensation = Matrix(comp_id, compensation, detectors, fluorochromes)
            self._comp_events = self.compensation.apply(self)
        else:
            # compensation must be None so clear any matrix and comp events
            self.compensation = None
            self._comp_events = None

        # Clear any previously transformed events
        # TODO: Consider caching the transform and re-applying
        self._transformed_events = None

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

    # TODO: make event type names/references consistent across the API...is it xform/transform comp/compensated, etc.
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

    def as_dataframe(self, source='xform', subsample=False, col_order=None, col_names=None):
        """
        Returns a Pandas DataFrame of event data

        :param source: 'orig', 'raw', 'comp', 'xform' for whether the original (no gain applied),
            raw (orig + gain), compensated (raw + comp), or transformed (comp + xform) events will
            be returned
        :param subsample: Whether to return all events or just the sub-sampled
            events. Default is False (all events)
        :param col_order: list of PnN labels. Determines the order of columns
            in the output DataFrame. If None, the column order will match the FCS file.
        :param col_names: list of new column labels. If None (default), the DataFrame
            columns will be a MultiIndex of the PnN / PnS labels.
        :return: Pandas DataFrame of event data
        """
        if source == 'xform':
            events = self.get_transformed_events(subsample=subsample)
        elif source == 'comp':
            events = self.get_comp_events(subsample=subsample)
        elif source == 'raw':
            events = self.get_raw_events(subsample=subsample)
        elif source == 'orig':
            events = self.get_orig_events(subsample=subsample)
        else:
            raise ValueError("source must be one of 'orig', 'raw', 'comp', or 'xform'")

        multi_cols = pd.MultiIndex.from_arrays([self.pnn_labels, self.pns_labels], names=['pnn', 'pns'])
        events_df = pd.DataFrame(data=events, columns=multi_cols)

        if col_order is not None:
            events_df = events_df[col_order]

        if col_names is not None:
            events_df.columns = col_names

        return events_df

    def get_channel_number_by_label(self, label):
        """
        Returns the channel number for the given PnN label. Note, this is the
        channel number as defined in the FCS data (not the channel index), so
        the 1st channel's number is 1 (not 0).

        :param label: PnN label of a channel
        :return: Channel number (not index)
        """
        if label in self.pnn_labels:
            return self.pnn_labels.index(label) + 1
        else:
            # as a last resort we can try the FJ labels and fail if no match
            return self._flowjo_pnn_labels.index(label) + 1

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
            if channel_label_or_number < 1:
                raise ValueError("Channel numbers are indexed at 1, got %d" % channel_label_or_number)
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
            if not isinstance(self.subsample_indices, np.ndarray):
                raise ValueError("Subsampling requested, but sample hasn't been sub-sampled: call `subsample_events`")
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

    def _transform(self, transform, include_scatter=False):
        if isinstance(transform, _transforms.RatioTransform):
            raise NotImplementedError(
                "RatioTransform cannot be applied to a Sample instance directly.\n"
                "To apply a RatioTransform, either:\n"
                "    1) Provide the Sample instance to the transform `apply` method\n"
                "    2) Use the RatioTransform as part of a GatingStrategy\n"
            )

        if self._comp_events is not None:
            transformed_events = self._comp_events.copy()
        else:
            transformed_events = self._raw_events.copy()

        if isinstance(transform, dict):
            for pnn_label, param_xform in transform.items():
                param_idx = self.get_channel_index(pnn_label)

                transformed_events[:, param_idx] = param_xform.apply(
                    transformed_events[:, param_idx]
                )
        else:
            if include_scatter:
                transform_indices = self.scatter_indices + self.fluoro_indices
            else:
                transform_indices = self.fluoro_indices

            transformed_events[:, transform_indices] = transform.apply(
                transformed_events[:, transform_indices]
            )

        return transformed_events

    def apply_transform(self, transform, include_scatter=False):
        """
        Applies given transform to Sample events, and overwrites the `transform` attribute.
        By default, only the fluorescent channels are transformed. For fully customized transformations
        per channel, the `transform` can be specified as a dictionary mapping PnN labels to an instance
        of the Transform sub-class. If a dictionary of transforms is specified, the `include_scatter`
        option is ignored and only the channels explicitly included in the transform dictionary will
        be transformed.

        :param transform: an instance of a Transform sub-class or a dictionary where the keys correspond
            to the PnN labels and the value is an instance of a Transform sub-class.
        :param include_scatter: Whether to transform the scatter channel in addition to the
            fluorescent channels. Default is False.
        """

        self._transformed_events = self._transform(transform, include_scatter=include_scatter)
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

        x_min, x_max = plot_utils.calculate_extent(x, d_min=x_min, d_max=x_max, pad=0.02)
        y_min, y_max = plot_utils.calculate_extent(y, d_min=y_min, d_max=y_max, pad=0.02)

        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_title(self.original_filename)

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_xlabel(self.pnn_labels[x_index])
        ax.set_ylabel(self.pnn_labels[y_index])

        if plot_events:
            seaborn.scatterplot(
                x=x,
                y=y,
                palette=plot_utils.new_jet,
                legend=False,
                s=5,
                linewidth=0,
                alpha=0.4
            )

        if plot_contour:
            seaborn.kdeplot(
                x=x,
                y=y,
                bw_method='scott',
                cmap=plot_utils.new_jet,
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
                "Have you called a transform method? \n"
                "Or, maybe you meant to plot the non-transformed events? If so, use the source='raw' option."
            )

        x_index = self.get_channel_index(x_label_or_number)
        y_index = self.get_channel_index(y_label_or_number)

        x = self.get_channel_data(x_index, source=source, subsample=subsample)
        y = self.get_channel_data(y_index, source=source, subsample=subsample)

        dim_labels = []

        if self.pns_labels[x_index] != '':
            dim_labels.append('%s (%s)' % (self.pns_labels[x_index], self.pnn_labels[x_index]))
        else:
            dim_labels.append(self.pnn_labels[x_index])

        if self.pns_labels[y_index] != '':
            dim_labels.append('%s (%s)' % (self.pns_labels[y_index], self.pnn_labels[y_index]))
        else:
            dim_labels.append(self.pnn_labels[y_index])

        p = plot_utils.plot_scatter(
            x,
            y,
            dim_labels,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            color_density=color_density
        )

        p.title = Title(text=self.original_filename, align='center')

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
            bins=None
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
        :return: Matplotlib figure of the histogram plot with KDE curve.
        """

        channel_index = self.get_channel_index(channel_label_or_number)
        channel_data = self.get_channel_data(channel_index, source=source, subsample=subsample)

        p = plot_utils.plot_histogram(
            channel_data,
            x_label=self.pnn_labels[channel_index],
            bins=bins
        )

        p.title = Title(text=self.original_filename, align='center')

        return p

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
        :param source: 'orig', 'raw', 'comp', 'xform' for whether the original (no gain applied),
            raw (orig + gain), compensated (raw + comp), or transformed (comp + xform) events  are
            used for exporting
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
            idx = np.zeros(self.event_count, bool)
            idx[self.subsample_indices] = True
        else:
            # include all events to start with
            idx = np.ones(self.event_count, bool)

        if exclude == 'bad':
            idx[self.anomalous_indices] = False
        elif exclude == 'good':
            good_idx = np.zeros(self.event_count, bool)
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
            raise ValueError("source must be one of 'orig', 'raw', 'comp', or 'xform'")

        ext = os.path.splitext(filename)[-1]

        # TODO: support exporting to HDF5 format, but as optional dependency/import
        if ext == '.csv':
            np.savetxt(
                output_path,
                events,
                delimiter=',',
                header=",".join(self.pnn_labels),
                comments=''
            )
        elif ext == '.fcs':
            fh = open(output_path, 'wb')

            flowio.create_fcs(
                events.flatten().tolist(),
                channel_names=self.pnn_labels,
                opt_channel_names=self.pns_labels,
                file_handle=fh
            )
            fh.close()
