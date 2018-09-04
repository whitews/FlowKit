import flowio
import flowutils
import os
from pathlib import Path
import io
from tempfile import TemporaryFile
import numpy as np
from flowkit import utils
from scipy import stats
import matplotlib.pyplot as plt
import seaborn


class Sample(object):
    def __init__(
            self,
            fcs_path_or_data,
            channel_labels=None,
            compensation=None,
            subsample_count=10000,
            random_seed=1,
            filter_negative_scatter=True,
            filter_anomalous_events=True
    ):
        """
        Represents a single FCS sample.

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

        self.event_count = self._flow_data.event_count
        self.channels = self._flow_data.channels
        self.pnn_labels = list()
        self.pns_labels = list()
        self.fluoro_indices = list()

        for n in sorted([int(k) for k in self.channels.keys()]):
            chan_label = self.channels[str(n)]['PnN']
            self.pnn_labels.append(chan_label)

            if chan_label.lower()[:4] not in ['fsc-', 'ssc-', 'time']:
                self.fluoro_indices.append(n - 1)

            if 'PnS' in self.channels[str(n)]:
                self.pns_labels.append(self.channels[str(n)]['PnS'])
            else:
                self.pns_labels.append('')

        self._raw_events = np.reshape(
            self._flow_data.events,
            (-1, self._flow_data.channel_count)
        )
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
                ref_set_count=3
            )
            self.anomalous_indices = anomalous_idx

        bad_idx = np.unique(np.concatenate([neg_scatter_idx, anomalous_idx]))
        bad_count = bad_idx.shape[0]
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
        Applies given compensation matrix to Sample events. If any transformation has been
        applied, those events will be deleted

        :param compensation: a compensation matrix NumPy array, CSV file or string file path
        :return: None
        """
        if compensation is not None:
            self.compensation = utils.parse_compensation_matrix(compensation, self.pnn_labels)
        else:
            self.compensation = None

        self._transformed_events = None
        self._compensate()

    def get_metadata(self):
        return self._flow_data.text

    def get_raw_events(self, subsample=False):
        if subsample:
            return self._raw_events[self.subsample_indices]
        else:
            return self._raw_events

    def get_comp_events(self, subsample=False):
        if self._comp_events is None:
            # TODO: should issue warning instructing user to call compensate
            return None

        if subsample:
            return self._comp_events[self.subsample_indices]
        else:
            return self._comp_events

    def get_transformed_events(self, subsample=False):
        if self._transformed_events is None:
            # TODO: should issue warning instructing user to call a transform
            return None

        if subsample:
            return self._transformed_events[self.subsample_indices]
        else:
            return self._transformed_events

    def get_channel_number_by_label(self, label):
        return self.pnn_labels.index(label) + 1

    def get_channel_index(self, channel_label_or_number):
        if isinstance(channel_label_or_number, str):
            index = self.get_channel_number_by_label(channel_label_or_number) - 1
        elif isinstance(channel_label_or_number, int):
            index = channel_label_or_number - 1
        else:
            raise ValueError("x_label_or_number must be a label string or channel number")

        return index

    def get_channel_data(self, channel_index, source='xform', subsample=False):
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

        Saves transformed data to self._transformed_events
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

        Saves transformed data to self._transformed_events
        """
        # only transform fluorescent channels
        self._transformed_events = flowutils.transforms.asinh(
            self._comp_events,
            self.fluoro_indices,
            pre_scale=pre_scale
        )

    def plot_scatter(
            self,
            x_label_or_number,
            y_label_or_number,
            source='xform',
            subsample=False,
            contours=False,
            x_min=None,
            x_max=None,
            y_min=None,
            y_max=None,
            fig_size=(8, 8)
    ):
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

        values = np.vstack([x, y])
        kernel = stats.gaussian_kde(values, bw_method='silverman')
        h = kernel.silverman_factor() / 2.0
        kernel.set_bandwidth(h)
        z = kernel.evaluate(values)

        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_xlabel(self.pnn_labels[x_index])
        ax.set_ylabel(self.pnn_labels[y_index])
        scatter_alpha = 1.0

        if contours:
            scatter_alpha = 0.4
            seaborn.kdeplot(
                x,
                y,
                bw='scott',
                cmap=utils.new_jet,
                linewidths=2,
                alpha=1
            )

        seaborn.scatterplot(
            x,
            y,
            hue=z,
            palette=utils.new_jet,
            legend=False,
            s=5,
            linewidth=0,
            alpha=scatter_alpha
        )

        plt.show()

    def plot_scatter_matrix(self, source='xform'):
        pass

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
        channel_index = self.get_channel_index(channel_label_or_number)
        channel_data = self.get_channel_data(channel_index, source=source, subsample=subsample)

        _, ax = plt.subplots(figsize=fig_size)
        ax.set_xlim([x_min, x_max])

        seaborn.distplot(
            channel_data,
            hist_kws=dict(edgecolor="w", linewidth=1),
            label=self.pnn_labels[channel_index],
            bins=bins
        )

        plt.show()

    def export_csv(self, source='xform', subsample=False, filename=None, directory=None):
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
                output_path, self._transformed_events[idx, :], delimiter=',', header=header
            )
        elif source == 'comp':
            np.savetxt(output_path, self._comp_events[idx, :], delimiter=',', header=header)
        elif source == 'raw':
            np.savetxt(output_path, self._raw_events[idx, :], delimiter=',', header=header)
        else:
            raise ValueError("source must be one of 'raw', 'comp', or 'xform'")

    def export_fcs(self, source='xform', subsample=False, filename=None, directory=None):
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
