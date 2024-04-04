"""
Sample class
"""

from functools import total_ordering
import copy
import flowio
import os
from pathlib import Path
import io
from tempfile import TemporaryFile
import numpy as np
import pandas as pd
from bokeh.layouts import gridplot
from bokeh.models import Title, Range1d
import warnings
# noinspection PyProtectedMember
from .._models.transforms import _transforms
# noinspection PyProtectedMember
from .._models.transforms._matrix import Matrix
from .._utils import plot_utils
from ..exceptions import FlowKitException


@total_ordering
class Sample(object):
    """
    Represents a single FCS sample from an FCS file, NumPy array or pandas
    DataFrame.

    For Sample plot methods, pay attention to the defaults for the subsample
    arguments, as most will use the sub-sampled events by default for better
    performance. For compensation and transformation routines, all events are
    always processed.

    Note on ignore_offset_error:
        Some FCS files incorrectly report the location of the last data byte
        as the last byte exclusive of the data section rather than the last
        byte inclusive of the data section. Technically, these are invalid
        FCS files but these are not corrupted data files. To attempt to read
        in these files, set the `ignore_offset_error` option to True.

    Note on ignore_offset_discrepancy and use_header_offset:
        The byte offset location for the DATA segment is defined in 2 places
        in an FCS file: the HEADER and the TEXT segments. By default, FlowIO
        uses the offset values found in the TEXT segment. If the HEADER values
        differ from the TEXT values, a DataOffsetDiscrepancyError will be
        raised. This option allows overriding this error to force the loading
        of the FCS file. The related `use_header_offset` can be used to
        force loading the file using the data offset locations found in the
        HEADER section rather than the TEXT section. Setting `use_header_offset`
        to True is equivalent to setting both options to True, meaning no
        error will be raised for an offset discrepancy.

    :param fcs_path_or_data: FCS data, can be either:

        - a file path or file handle to an FCS file
        - a pathlib Path object
        - a FlowIO FlowData object
        - a NumPy array of FCS event data (must provide sample_id & channel_labels)
        - a pandas DataFrame containing FCS event data (channel labels as column labels, must provide sample_id)

    :param sample_id: A text string to use for the Sample's ID. If None, the ID will be
        taken from the 'fil' keyword of the metadata. If the 'fil' keyword is not present,
        the value will be the filename if given a file. For a NumPy array or Pandas
        DataFrame, a text value is required.

    :param channel_labels: A list of strings or a list of tuples to use for the channel
        labels. Required if fcs_path_or_data is a NumPy array

    :param compensation: Compensation matrix, which can be a:

        - Matrix instance
        - NumPy array
        - CSV file path
        - pathlib Path object to a CSV or TSV file
        - string of CSV text

    :param null_channel_list: List of PnN labels for acquired channels that do not contain
        useful data. Note, this should only be used if no fluorochromes were used to target
        those detectors. Null channels do not contribute to compensation and should not be
        included in a compensation matrix for this sample.

    :param ignore_offset_error: option to ignore data offset error (see above note), default is False

    :param ignore_offset_discrepancy: option to ignore discrepancy between the HEADER
        and TEXT values for the DATA byte offset location, default is False

    :param use_header_offsets: use the HEADER section for the data offset locations, default is False.
        Setting this option to True also suppresses an error in cases of an offset discrepancy.

    :param cache_original_events: Original events are the unprocessed events as stored in the FCS binary,
        meaning they have not been scaled according to channel gain, corrected for proper lin/log display,
        or had the time channel scaled by the 'timestep' keyword value (if present). By default, these
        events are not retained by the Sample class as they are typically not useful. To retrieve the
        original events, set this to True and call the get_events() method with source='orig'.

    :param subsample: The number of events to use for sub-sampling. The number of sub-sampled events
        can be changed after instantiation using the `subsample_events` method. The random seed can
        also be specified using that method. Sub-sampled events are used predominantly for speeding
        up plotting methods.
    """
    def __init__(
            self,
            fcs_path_or_data,
            sample_id=None,
            channel_labels=None,
            compensation=None,
            null_channel_list=None,
            ignore_offset_error=False,
            ignore_offset_discrepancy=False,
            use_header_offsets=False,
            cache_original_events=False,
            subsample=10000
    ):
        """
        Create a Sample instance
        """
        # inspect our fcs_path_or_data argument
        if isinstance(fcs_path_or_data, str):
            # if a string, we only handle file paths, so try creating a FlowData object
            flow_data = flowio.FlowData(
                fcs_path_or_data,
                ignore_offset_error=ignore_offset_error,
                ignore_offset_discrepancy=ignore_offset_discrepancy,
                use_header_offsets=use_header_offsets
            )
        elif isinstance(fcs_path_or_data, io.IOBase):
            flow_data = flowio.FlowData(
                fcs_path_or_data,
                ignore_offset_error=ignore_offset_error,
                ignore_offset_discrepancy=ignore_offset_discrepancy,
                use_header_offsets=use_header_offsets
            )
        elif isinstance(fcs_path_or_data, Path):
            flow_data = flowio.FlowData(
                fcs_path_or_data.open('rb'),
                ignore_offset_error=ignore_offset_error,
                ignore_offset_discrepancy=ignore_offset_discrepancy,
                use_header_offsets=use_header_offsets
            )
        elif isinstance(fcs_path_or_data, flowio.FlowData):
            flow_data = fcs_path_or_data
        elif isinstance(fcs_path_or_data, np.ndarray):
            if sample_id is None:
                raise ValueError("'sample_id' is required for a NumPy array")

            tmp_file = TemporaryFile()
            flowio.create_fcs(
                tmp_file,
                fcs_path_or_data.flatten().tolist(),
                channel_names=channel_labels
            )

            flow_data = flowio.FlowData(tmp_file)
        elif isinstance(fcs_path_or_data, pd.DataFrame):
            if sample_id is None:
                raise ValueError("'sample_id' is required for a Pandas DataFrame")

            tmp_file = TemporaryFile()

            # Handle MultiIndex columns since that is what the as_dataframe method creates.
            if fcs_path_or_data.columns.nlevels > 1:
                pnn_labels = fcs_path_or_data.columns.get_level_values(0)
                pns_labels = fcs_path_or_data.columns.get_level_values(1)
            else:
                pnn_labels = fcs_path_or_data.columns
                pns_labels = None

            flowio.create_fcs(
                tmp_file,
                fcs_path_or_data.values.flatten().tolist(),
                channel_names=pnn_labels,
                opt_channel_names=pns_labels
            )

            flow_data = flowio.FlowData(tmp_file)
        else:
            raise ValueError("'fcs_path_or_data' is not a supported type")

        try:
            self.version = flow_data.header['version']
        except KeyError:
            self.version = None

        # Ensure null channels is a list for checking later
        if null_channel_list is None:
            self.null_channels = []
        else:
            self.null_channels = null_channel_list

        self.event_count = flow_data.event_count

        # make a temp channels dict, self.channels will be a DataFrame built from it
        tmp_channels = flow_data.channels
        self.pnn_labels = list()
        self.pns_labels = list()
        self.fluoro_indices = list()
        self.scatter_indices = list()
        self.time_index = None

        channel_gain = []
        channel_lin_log = []
        channel_range = []
        self.metadata = flow_data.text

        for n in sorted([int(k) for k in tmp_channels.keys()]):
            channel_label = tmp_channels[str(n)]['PnN']
            self.pnn_labels.append(channel_label)

            if 'p%dg' % n in self.metadata:
                channel_gain.append(float(self.metadata['p%dg' % n]))
            else:
                channel_gain.append(1.0)

            # PnR range values are required for all channels
            channel_range.append(float(self.metadata['p%dr' % n]))

            # PnE specifies whether the parameter data is stored in on linear or log scale
            # and includes 2 values: (f1, f2)
            # where:
            #     f1 is the number of log decades (valid values are f1 >= 0)
            #     f2 is the value to use for log(0) (valid values are f2 >= 0)
            # Note for log scale, both values must be > 0
            # linear = (0, 0)
            # log    = (f1 > 0, f2 > 0)
            if 'p%de' % n in self.metadata:
                (decades, log0) = [
                    float(x) for x in self.metadata['p%de' % n].split(',')
                ]
                if log0 == 0 and decades != 0:
                    log0 = 1.0  # FCS std states to use 1.0 for invalid 0 value
                channel_lin_log.append((decades, log0))
            else:
                channel_lin_log.append((0.0, 0.0))

            # Determine fluoro vs scatter vs time channels
            # Null channels are excluded from any category.
            if channel_label in self.null_channels:
                pass
            elif channel_label.lower()[:4] not in ['fsc-', 'ssc-', 'time']:
                self.fluoro_indices.append(n - 1)
            elif channel_label.lower()[:4] in ['fsc-', 'ssc-']:
                self.scatter_indices.append(n - 1)
            elif channel_label.lower() == 'time':
                self.time_index = n - 1

            if 'PnS' in tmp_channels[str(n)]:
                self.pns_labels.append(tmp_channels[str(n)]['PnS'])
            else:
                self.pns_labels.append('')

        self._flowjo_pnn_labels = [label.replace('/', '_') for label in self.pnn_labels]

        # build the self.channels DataFrame
        self.channels = pd.DataFrame()
        self.channels['channel_number'] = sorted([int(k) for k in tmp_channels.keys()])
        self.channels['pnn'] = self.pnn_labels
        self.channels['pns'] = self.pns_labels
        self.channels['png'] = channel_gain
        self.channels['pne'] = channel_lin_log
        self.channels['pnr'] = channel_range

        # Start processing the event data. First, we'll get the unprocessed events
        # This is the main entry point for event data into FlowKit, and we ensure
        # the events are double precision because of the pre-processing that will
        # make even integer FCS data types floating point. The precision is needed
        # for accurate gating results.
        tmp_orig_events = np.reshape(
            np.array(flow_data.events, dtype=np.float64),
            (-1, flow_data.channel_count)
        )

        if cache_original_events:
            self._orig_events = tmp_orig_events
        else:
            self._orig_events = None

        # Event data must be scaled according to channel gain, as well
        # as corrected for proper lin/log display, and the time channel
        # scaled by the 'timestep' keyword value (if present).
        # This is the only pre-processing we will do on raw events
        raw_events = copy.deepcopy(tmp_orig_events)

        # Note: The time channel is scaled by the timestep (if present),
        # but should not be scaled by any gain value present in PnG.
        # It seems common for cytometers to include a gain value for the
        # time channel that matches the fluoro channels. Not sure why
        # they do this, but it makes no sense to have an amplifier gain
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
        self._include_scatter_option = False  # stores user option from transform_events method

        if compensation is not None:
            self.apply_compensation(compensation)

        # if filtering any events, save those in case they want to be retrieved
        self.negative_scatter_indices = None
        self.flagged_indices = None
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

        if sample_id is None:
            self.id = self.original_filename
        else:
            self.id = sample_id

        # finally, store initial sub-sampled event indices
        self.subsample_events(subsample)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'v{self.version}, {self.id}, '
            f'{len(self.pnn_labels)} channels, {self.event_count} events)'
        )

    def __lt__(self, other):
        return self.id < other.id

    def __eq__(self, other):
        return self.id == other.id

    def filter_negative_scatter(self, reapply_subsample=True):
        """
        Determines indices of negative scatter events, optionally re-subsample the Sample events afterward.

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

    def set_flagged_events(self, event_indices):
        """
        Flags the given event indices. Can be useful for flagging anomalous time events for quality control or
        for any other purpose. Flagged indices do not affect analysis, it is only used as an option when
        exporting Sample event data.

        :param event_indices: list of event indices to flag
        :return: None
        """
        self.flagged_indices = event_indices

    def get_index_sorted_locations(self):
        """
        Retrieve well locations for index sorted data (if present in metadata)

        :return: list of 2-D tuples
        """
        # Index sorted FCS files contain data where each event corresponds
        # to a location on a well plate (i.e. a 16x24 array for a 384 well plate).
        # It appears that, by convention, these FCS files contain extra
        # metadata keywords with values that list the well location (row & column)
        # for each event. The keywords have the format "index sorting locations_1"
        # and the value is a string of semicolon delimited coordinates that
        # are comma delimited.
        #
        # We'll start by checking for the existence of any keywords
        # starting with the string "index sorting locations"
        idx_sort_keyword_base = "index sorting locations"
        idx_sort_keywords = [k for k in self.metadata.keys() if k.startswith(idx_sort_keyword_base)]

        # loop through in order, these will be indexed at 1
        idx_sort_keyword_count = len(idx_sort_keywords)

        idx_sort_data = []

        if idx_sort_keyword_count == 0:
            warnings.warn("Sample does not contain index sorted data")
            return idx_sort_data

        idx_locations_string = ''

        for i in range(1, idx_sort_keyword_count + 1):
            idx_sort_keyword = "_".join([idx_sort_keyword_base, str(i)])
            idx_locations_string += self.metadata[idx_sort_keyword]

        # strip any leading or trailing delimiters (there seems to always be a trailing one)
        idx_locations_string = idx_locations_string.strip(';')

        # finally, loop through and save the coordinates as tuples
        for idx_loc_string in idx_locations_string.split(';'):
            idx_sort_data.append(tuple(idx_loc_string.split(',')))

        return idx_sort_data

    def subsample_events(
            self,
            subsample_count=10000,
            random_seed=1
    ):
        """
        Stores a set of sub-sampled indices for event data. Sub-sampled events
        can be accessed via the `get_events` method by setting the keyword
        argument `subsample=True`. The sub-sampled indices are available via
        the `subsample_indices` attribute.

        :param subsample_count: Number of events to use as a sub-sample. If the number of
            events in the Sample is less than the requested sub-sample count, then the
            maximum number of available events is used for the sub-sample.
        :param random_seed: Random seed used for sub-sampling events
        :return: None
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

        if self.flagged_indices is not None:
            bad_idx = np.unique(np.concatenate([bad_idx, self.flagged_indices]))

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
        transformation has been applied, it will be re-applied after
        compensation. Compensated events can be retrieved afterward
        by calling `get_events` with `source='comp'`. Note, if the
        sample specifies null channels then these must not be present
        in the compensation matrix.

        :param compensation: Compensation matrix, which can be a:

                - Matrix instance
                - NumPy array
                - CSV file path
                - pathlib Path object to a CSV or TSV file
                - string of CSV text

            If a string, both multi-line traditional CSV, and the single
            line FCS spill formats are supported. If a NumPy array, we
            assume the columns are in the same order as the channel labels.
        :param comp_id: text ID for identifying compensation matrix (not used if compensation was a Matrix instance)
        :return: None
        """
        if isinstance(compensation, Matrix):
            tmp_matrix = compensation
        elif compensation is not None:
            detectors = [self.pnn_labels[i] for i in self.fluoro_indices]
            fluorochromes = [self.pns_labels[i] for i in self.fluoro_indices]
            tmp_matrix = Matrix(comp_id, compensation, detectors, fluorochromes)
        else:
            # compensation must be None, we'll clear any stored comp events later
            tmp_matrix = None

        if tmp_matrix is not None:
            # We don't check null channels b/c Matrix.apply will catch them.
            # tmp_matrix ensures we don't store comp events or compensation
            # unless apply succeeds.
            self._comp_events = tmp_matrix.apply(self)
            self.compensation = tmp_matrix
        else:
            # compensation given was None, clear any matrix and comp events
            self._comp_events = None
            self.compensation = None

        # Re-apply transform if set
        if self.transform is not None:
            self._transformed_events = self._transform(self.transform, self._include_scatter_option)

    def get_metadata(self):
        """
        Retrieve FCS metadata.

        :return: Dictionary of FCS metadata
        """
        return self.metadata

    def _get_orig_events(self):
        """
        Returns 'original' events, i.e. not pre-processed, compensated,
        or transformed.

        :return: NumPy array of original events
        """
        if self._orig_events is None:
            raise ValueError(
                "Original events were not cached, to retrieve them create a "
                "Sample instance with cache_original_events=True"
            )

        return self._orig_events

    def _get_raw_events(self):
        """
        Returns 'raw' events that have been pre-processed to adjust for channel
        gain and lin/log display, but have not been compensated or transformed.

        :return: NumPy array of raw events
        """
        return self._raw_events

    def _get_comp_events(self):
        """
        Returns compensated events, (not transformed)

        :return: NumPy array of compensated events or None if no compensation
            matrix has been applied.
        """
        if self._comp_events is None:
            raise AttributeError(
                "Compensated events were requested but do not exist.\n"
                "Call a apply_compensation method prior to retrieving compensated events."
            )

        return self._comp_events

    def _get_transformed_events(self):
        """
        Returns transformed events. Note, if a compensation matrix has been
        applied then the events returned will be compensated and transformed.

        :return: NumPy array of transformed events or None if no transform
            has been applied.
        """
        if self._transformed_events is None:
            raise AttributeError(
                "Transformed events were requested but do not exist.\n"
                "Call a transform method prior to retrieving transformed events."
            )

        return self._transformed_events

    def get_events(self, source='xform', subsample=False):
        """
        Returns a NumPy array of event data.

        Note: This method returns the array directly, not a copy of the array. Be careful if you
        are planning to modify returned event data, and make a copy of the array when appropriate.

        :param source: 'orig', 'raw', 'comp', 'xform' for whether the original (no gain applied),
            raw (orig + gain), compensated (raw + comp), or transformed (comp + xform) events will
            be returned
        :param subsample: Whether to return all events or just the sub-sampled
            events. Default is False (all events)
        :return: NumPy array of event data
        """
        if source == 'xform':
            events = self._get_transformed_events()
        elif source == 'comp':
            events = self._get_comp_events()
        elif source == 'raw':
            events = self._get_raw_events()
        elif source == 'orig':
            events = self._get_orig_events()
        else:
            raise ValueError("source must be one of 'orig', 'raw', 'comp', or 'xform'")
        
        if subsample:
            events = events[self.subsample_indices]

        return events

    def as_dataframe(self, source='xform', subsample=False, col_order=None, col_names=None):
        """
        Returns a pandas DataFrame of event data.

        :param source: 'orig', 'raw', 'comp', 'xform' for whether the original (no gain applied),
            raw (orig + gain), compensated (raw + comp), or transformed (comp + xform) events will
            be returned
        :param subsample: Whether to return all events or just the sub-sampled
            events. Default is False (all events)
        :param col_order: list of PnN labels. Determines the order of columns
            in the output DataFrame. If None, the column order will match the FCS file.
        :param col_names: list of new column labels. If None (default), the DataFrame
            columns will be a MultiIndex of the PnN / PnS labels.
        :return: pandas DataFrame of event data
        """
        events = self.get_events(source=source, subsample=subsample)

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
        channel number, as defined in the FCS data (not the channel index), so
        the 1st channel's number is 1 (not 0).

        :param label: PnN channel label
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

    def get_channel_events(self, channel_index, source='xform', subsample=False):
        """
        Returns a NumPy array of event data for the specified channel index.

        Note: This method returns the array directly, not a copy of the array. Be careful if you
        are planning to modify returned event data, and make a copy of the array when appropriate.

        :param channel_index: channel index for which data is returned
        :param source: 'raw', 'comp', 'xform' for whether the raw, compensated
            or transformed events will be returned
        :param subsample: Whether to return all events or just the sub-sampled
            events. Default is False (all events)
        :return: NumPy array of event data for the specified channel index
        """
        events = self.get_events(source=source, subsample=subsample)
        events = events[:, channel_index]

        return events

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
        By default, only the fluorescent channels are transformed (and excludes null channels).
        For fully customized transformations per channel, the `transform` can be specified as a
        dictionary mapping PnN labels to an instance of the Transform subclass. If a dictionary
        of transforms is specified, the `include_scatter` option is ignored and only the channels
        explicitly included in the transform dictionary will be transformed.

        :param transform: an instance of a Transform subclass or a dictionary where the keys correspond
            to the PnN labels and the value is an instance of a Transform subclass.
        :param include_scatter: Whether to transform the scatter channel in addition to the
            fluorescent channels. Default is False.
        """
        self._transformed_events = self._transform(transform, include_scatter=include_scatter)
        self._include_scatter_option = include_scatter
        self.transform = transform

    def plot_channel(
            self,
            channel_label_or_number,
            source='xform',
            subsample=True,
            color_density=True,
            bin_width=4,
            event_mask=None,
            highlight_mask=None,
            x_min=None,
            x_max=None,
            y_min=None,
            y_max=None
    ):
        """
        Plot a 2-D histogram of the specified channel data with the x-axis as the event index.
        This is similar to plotting a channel vs Time, except the events are equally
        distributed along the x-axis.

        :param channel_label_or_number: A channel's PnN label or number
        :param source: 'raw', 'comp', 'xform' for whether the raw, compensated
            or transformed events are used for plotting
        :param subsample: Whether to use all events for plotting or just the
            sub-sampled events. Default is True (sub-sampled events). Plotting
            sub-sampled events is much faster.
        :param color_density: Whether to color the events by density, similar
            to a heat map. Default is True.
        :param bin_width: Bin size to use for the color density, in units of
            event point size. Larger values produce smoother gradients.
            Default is 4 for a 4x4 grid size.
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
        :return: A Bokeh Figure object containing the interactive channel plot.
        """
        channel_index = self.get_channel_index(channel_label_or_number)
        channel_data = self.get_channel_events(channel_index, source=source, subsample=subsample)

        if subsample:
            x_idx = self.subsample_indices
        else:
            x_idx = np.arange(self.event_count)

        if self.pns_labels[channel_index] != '':
            channel_label = '%s (%s)' % (self.pns_labels[channel_index], self.pnn_labels[channel_index])
        else:
            channel_label = self.pnn_labels[channel_index]

        fig = plot_utils.plot_scatter(
            x_idx,
            channel_data,
            x_label='Events',
            y_label=channel_label,
            color_density=color_density,
            bin_width=bin_width,
            event_mask=event_mask,
            highlight_mask=highlight_mask,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max
        )

        fig.aspect_ratio = 3
        fig.width = 1000

        return fig

    def plot_contour(
            self,
            x_label_or_number,
            y_label_or_number,
            source='xform',
            subsample=True,
            plot_events=False,
            fill=False,
            x_min=None,
            x_max=None,
            y_min=None,
            y_max=None
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
            sub-sampled events. Default is True (sub-sampled events). Running
            with all events is not recommended, as the Kernel Density
            Estimation is computationally demanding.
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
        :param fill: Whether to fill in color between contour lines. D default
            is False.
        :return: A Bokeh figure of the contour plot
        """
        x_index = self.get_channel_index(x_label_or_number)
        y_index = self.get_channel_index(y_label_or_number)

        x = self.get_channel_events(x_index, source=source, subsample=subsample)
        y = self.get_channel_events(y_index, source=source, subsample=subsample)

        if self.pns_labels[x_index] != '':
            x_label = '%s (%s)' % (self.pns_labels[x_index], self.pnn_labels[x_index])
        else:
            x_label = self.pnn_labels[x_index]

        if self.pns_labels[y_index] != '':
            y_label = '%s (%s)' % (self.pns_labels[y_index], self.pnn_labels[y_index])
        else:
            y_label = self.pnn_labels[y_index]

        fig = plot_utils.plot_contours(
            x,
            y,
            x_label=x_label,
            y_label=y_label,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            plot_events=plot_events,
            fill=fill
        )

        return fig

    def plot_scatter(
            self,
            x_label_or_number,
            y_label_or_number,
            source='xform',
            subsample=True,
            color_density=True,
            bin_width=4,
            event_mask=None,
            highlight_mask=None,
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
            sub-sampled events. Default is True (sub-sampled events). Plotting
            sub-sampled events is much faster.
        :param color_density: Whether to color the events by density, similar
            to a heat map. Default is True.
        :param bin_width: Bin size to use for the color density, in units of
            event point size. Larger values produce smoother gradients.
            Default is 4 for a 4x4 grid size.
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
        :return: A Bokeh Figure object containing the interactive scatter plot.
        """
        x_index = self.get_channel_index(x_label_or_number)
        y_index = self.get_channel_index(y_label_or_number)

        x = self.get_channel_events(x_index, source=source, subsample=subsample)
        y = self.get_channel_events(y_index, source=source, subsample=subsample)
        if highlight_mask is not None and subsample:
            highlight_mask = highlight_mask[self.subsample_indices]
        if event_mask is not None:
            if subsample:
                event_mask = event_mask[self.subsample_indices]

            # Verify event_mask has events to show
            if event_mask.sum() == 0:
                raise FlowKitException("There are no events to plot for the specified options")

        if self.pns_labels[x_index] != '':
            x_label = '%s (%s)' % (self.pns_labels[x_index], self.pnn_labels[x_index])
        else:
            x_label = self.pnn_labels[x_index]

        if self.pns_labels[y_index] != '':
            y_label = '%s (%s)' % (self.pns_labels[y_index], self.pnn_labels[y_index])
        else:
            y_label = self.pnn_labels[y_index]

        p = plot_utils.plot_scatter(
            x,
            y,
            x_label=x_label,
            y_label=y_label,
            event_mask=event_mask,
            highlight_mask=highlight_mask,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            color_density=color_density,
            bin_width=bin_width
        )

        p.title = Title(text=self.id, align='center')

        return p

    def plot_scatter_matrix(
            self,
            channel_labels_or_numbers=None,
            source='xform',
            subsample=True,
            event_mask=None,
            highlight_mask=None,
            color_density=False,
            plot_height=256,
            plot_width=256
    ):
        """
        Returns an interactive scatter plot matrix for all channel combinations
        except for the Time channel.

        :param channel_labels_or_numbers: List of channel PnN labels or channel
            numbers to use for the scatter plot matrix. If None, then all
            channels will be plotted (except Time).
        :param source: 'raw', 'comp', 'xform' for whether the raw, compensated
            or transformed events are used for plotting
        :param subsample: Whether to use all events for plotting or just the
            sub-sampled events. Default is True (sub-sampled events). Plotting
            sub-sampled events is much faster.
        :param event_mask: Boolean array of events to plot. Takes precedence
            over highlight_mask (i.e. events marked False in event_mask will
            never be plotted).
        :param highlight_mask: Boolean array of event indices to highlight
            in color. Non-highlighted events will be light grey.
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

                # check if we're at the diagonal,
                # don't plot past to avoid duplicates
                if channel_x == channel_y:
                    # plot histogram instead of scatter plot
                    plot = self.plot_histogram(
                        channel_x, source=source, subsample=subsample
                    )
                else:
                    plot = self.plot_scatter(
                        channel_x,
                        channel_y,
                        source=source,
                        subsample=subsample,
                        event_mask=event_mask,
                        highlight_mask=highlight_mask,
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
            data_min=None,
            data_max=None,
            x_range=None
    ):
        """
        Returns a histogram plot of the specified channel events

        :param channel_label_or_number:  A channel's PnN label or number to use
            for plotting the histogram
        :param source: 'raw', 'comp', 'xform' for whether the raw, compensated
            or transformed events are used for plotting
        :param subsample: Whether to use all events for plotting or just the
            sub-sampled events. Default is False (all events).
        :param bins: Number of bins to use for the histogram or a string compatible
            with the NumPy histogram function. If None, the number of bins is
            determined by the square root rule.
        :param data_min: filter event data, removing events below specified value
        :param data_max: filter event data, removing events above specified value
        :param x_range: Tuple of lower & upper bounds of x-axis. Used for modifying
            plot view, doesn't filter event data.
        :return: Bokeh figure of the histogram plot.
        """

        channel_index = self.get_channel_index(channel_label_or_number)
        channel_data = self.get_channel_events(channel_index, source=source, subsample=subsample)

        if data_min is not None:
            channel_data = channel_data[channel_data >= data_min]

        if data_max is not None:
            channel_data = channel_data[channel_data <= data_max]

        p = plot_utils.plot_histogram(
            channel_data,
            x_label=self.pnn_labels[channel_index],
            bins=bins
        )

        p.title = Title(text=self.id, align='center')

        if x_range is not None:
            x_range = Range1d(x_range[0], x_range[1])
            p.x_range = x_range

        return p

    def _get_metadata_for_export(self, source, include_all=False):
        metadata_dict = {}
        ignore_keywords = ['timestep']

        # If the original events were requested, need to make sure
        # some metadata is included and has certain values. This
        # ensures the events can be interpreted correctly.
        # For non-orig sources, the event values have already been
        # processed to account for the metadata.
        if source == 'orig':
            if 'timestep' in self.metadata and self.time_index is not None:
                metadata_dict['TIMESTEP'] = self.metadata['timestep']

            # Grab channel scale & gain values from self.channels
            # This seems safer than
            for _, channel_row in self.channels.iterrows():
                chan_num = channel_row['channel_number']

                gain_keyword = 'p%dg' % chan_num
                gain_value = str(channel_row['png'])

                scale_keyword = 'p%de' % chan_num
                decades, log0 = channel_row['pne']

                # in Python 3.6+, this seems safe to do
                scale_value = ",".join([str(decades), str(log0)])

                range_keyword = 'p%dr' % chan_num
                range_value = str(channel_row['pnr'])

                metadata_dict[gain_keyword] = gain_value
                metadata_dict[scale_keyword] = scale_value
                metadata_dict[range_keyword] = range_value

                ignore_keywords.extend([gain_keyword, scale_keyword, range_keyword])
        else:
            # for 'raw', 'comp', or 'xform' cases, set data type to float
            metadata_dict['datatype'] = 'F'

            # And set proper values for channel metadata
            for _, channel_row in self.channels.iterrows():
                chan_num = channel_row['channel_number']

                gain_keyword = 'p%dg' % chan_num
                scale_keyword = 'p%de' % chan_num
                range_keyword = 'p%dr' % chan_num

                metadata_dict[gain_keyword] = '1.0'
                metadata_dict[scale_keyword] = '0,0'
                metadata_dict[range_keyword] = '262144'

                ignore_keywords.extend([gain_keyword, scale_keyword, range_keyword])

        # Certain metadata fields are set automatically in FlowIO,
        # but FlowIO will ignore them if present, so it's fine to
        # include them.
        # However, we do want to avoid any of the above keywords that
        # we had to set.
        if include_all:
            for k, v in self.metadata.items():
                if k in ignore_keywords or k in metadata_dict:
                    # keyword has already been added or isn't needed
                    continue

                metadata_dict[k] = v

        return metadata_dict

    def export(
            self,
            filename,
            source='xform',
            exclude_neg_scatter=False,
            exclude_flagged=False,
            exclude_normal=False,
            subsample=False,
            include_metadata=False,
            directory=None
    ):
        """
        Export Sample event data to either a new FCS file or a CSV file. Format determined by filename extension.

        :param filename: Text string to use for the exported file name. File type is determined by
            the filename extension (supported types are .fcs & .csv).
        :param source: 'orig', 'raw', 'comp', 'xform' for whether the original (no gain applied),
            raw (orig + gain), compensated (raw + comp), or transformed (comp + xform) events  are
            used for exporting
        :param exclude_neg_scatter: Whether to exclude negative scatter events. Default is False.
        :param exclude_flagged: Whether to exclude flagged events. Default is False.
        :param exclude_normal: Whether to exclude "normal" events. This is useful for retrieving all
             the "bad" events (neg scatter and/or flagged events). Default is False.
        :param subsample: Whether to export all events or just the sub-sampled events.
            Default is False (all events).
        :param include_metadata: Whether to include all key/value pairs from the metadata attribute
            in the output FCS file. Only valid for .fcs file extension. If False, only the minimum
            amount of metadata will be included in the output FCS file. Default is False.
        :param directory: Directory path where the exported file will be saved. If None, the file
            will be saved in the current working directory.
        :return: None
        """
        # get the requested file type (either .fcs or .csv)
        ext = os.path.splitext(filename)[-1].lower()

        # Next, check if exporting as CSV, and issue a warning if so.
        # Exporting original events to CSV doesn't allow for the
        # inclusion of the proper metadata (PnG, PnE, PnR) for the
        # exported event values to be interpreted correctly.
        if ext == '.csv' and source == 'orig':
            warnings.warn(
                "Exporting original events as CSV will not include the metadata (gain, timestep, etc.) "
                "to properly interpret the exported event values."
            )
        elif ext == '.fcs' and source == 'orig':
            # Related to above: If exporting original events as an FCS file,
            # verify the data type is float ('F'). FlowIO doesn't support
            # creating non-float FCS files
            data_type = self.metadata['datatype']
            if data_type != 'F':
                raise NotImplementedError(
                    "Exporting original events is not supported for FCS files with data type %s." % data_type
                )

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

        if exclude_flagged:
            idx[self.flagged_indices] = False
        if exclude_neg_scatter:
            idx[self.negative_scatter_indices] = False
        if exclude_normal:
            # start with all events marked normal
            normal_idx = np.zeros(self.event_count, bool)

            # set neg scatter and flagged events to False
            normal_idx[self.negative_scatter_indices] = False
            normal_idx[self.flagged_indices] = False

            # then filter out the inverse normal indices
            idx = np.logical_and(idx, ~normal_idx)

        events = self.get_events(source=source)
        events = events[idx, :]

        if ext == '.csv':
            np.savetxt(
                output_path,
                events,
                delimiter=',',
                header=",".join(self.pnn_labels),
                comments=''
            )
        elif ext == '.fcs':
            metadata_dict = self._get_metadata_for_export(source=source, include_all=include_metadata)

            fh = open(output_path, 'wb')

            flowio.create_fcs(
                fh,
                events.flatten().tolist(),
                channel_names=self.pnn_labels,
                opt_channel_names=self.pns_labels,
                metadata_dict=metadata_dict
            )
            fh.close()
