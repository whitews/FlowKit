import os
from glob import glob
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from MulticoreTSNE import MulticoreTSNE
from bokeh.plotting import figure, show
import seaborn
from matplotlib import cm
import matplotlib.pyplot as plt
from flowio.create_fcs import create_fcs
from flowkit import Sample, GatingStrategy, _utils
import warnings

try:
    import multiprocessing as mp
    multi_proc = True
except ImportError:
    mp = None
    multi_proc = False


def get_samples_from_paths(sample_paths):
    if multi_proc:
        if len(sample_paths) < mp.cpu_count():
            proc_count = len(sample_paths)
        else:
            proc_count = mp.cpu_count() - 1  # leave a CPU free just to be nice

        pool = mp.Pool(processes=proc_count)
        samples = pool.map(Sample, sample_paths)
    else:
        samples = []
        for path in sample_paths:
            samples.append(Sample(path))

    return samples


def load_samples(fcs_samples):
    sample_list = []

    if isinstance(fcs_samples, list):
        # TODO: should we check that all Samples have the same channels?
        # and if so, should we determine the common channels and continue?

        # 'fcs_samples' is a list of either file paths or Sample instances
        sample_types = set()

        for sample in fcs_samples:
            sample_types.add(type(sample))

        if len(sample_types) > 1:
            raise ValueError(
                "Each item in 'fcs_sample' list must be a FCS file path or Sample instance"
            )

        if Sample in sample_types:
            sample_list = fcs_samples
        elif str in sample_types:
            sample_list = get_samples_from_paths(fcs_samples)
    elif isinstance(fcs_samples, Sample):
        # 'fcs_samples' is a single Sample instance
        sample_list = [fcs_samples]
    elif isinstance(fcs_samples, str):
        # 'fcs_samples' is a str to either a single FCS file or a directory
        # If directory, search non-recursively for files w/ .fcs extension
        if os.path.isdir(fcs_samples):
            fcs_paths = glob(os.path.join(fcs_samples, '*.fcs'))
            if len(fcs_paths) > 0:
                sample_list = get_samples_from_paths(fcs_paths)

    return sample_list


# TODO: gate_sample & gate_samples should be in the GatingStrategy class
def gate_sample(data):
    gating_strategy = data[0]
    sample = data[1]
    verbose = data[2]
    return gating_strategy.gate_sample(sample, verbose=verbose)


def gate_samples(gating_strategy, samples, verbose):
    if multi_proc:
        if len(samples) < mp.cpu_count():
            proc_count = len(samples)
        else:
            proc_count = mp.cpu_count() - 1  # leave a CPU free just to be nice

        pool = mp.Pool(processes=proc_count)
        data = [(gating_strategy, sample, verbose) for sample in samples]
        all_results = pool.map(gate_sample, data)
    else:
        all_results = []
        for sample in samples:
            results = gating_strategy.gate_sample(sample, verbose=verbose)
            all_results.append(results)

    return all_results


class Session(object):
    def __init__(self, fcs_samples=None, comp_bead_samples=None, gating_strategy=None, subsample_count=10000):
        self.samples = []
        self.bead_samples = []
        self.bead_lut = {}
        self.report = None
        self._results = None

        if comp_bead_samples == fcs_samples and isinstance(fcs_samples, str):
            # there's a mix of regular FCS files with bead files in the same directory,
            # which isn't supported at this time. Raise error
            raise ValueError("Specify bead samples as a list of paths if in the same directory as other samples")

        self.samples = load_samples(fcs_samples)

        for s in self.samples:
            s.subsample_events(subsample_count)

        if comp_bead_samples is not None:
            self.bead_samples = load_samples(comp_bead_samples)
            self.process_bead_samples()

        if isinstance(gating_strategy, GatingStrategy):
            self.gating_strategy = gating_strategy
        elif isinstance(gating_strategy, str):
            # assume a path to a GatingML XML file
            self.gating_strategy = GatingStrategy(gating_strategy)
        elif gating_strategy is None:
            self.gating_strategy = GatingStrategy()
        else:
            raise ValueError(
                "'gating_strategy' must be either a GatingStrategy instance or a path to a GatingML document"
            )

    @property
    def gates(self):
        return self.gating_strategy.gates

    def get_sample(self, sample_id):
        for s in self.samples:
            if s.original_filename == sample_id:
                return s

    def process_bead_samples(self):
        # do nothing if there are no bead samples
        bead_sample_count = len(self.bead_samples)
        if bead_sample_count == 0:
            warnings.warn("No bead samples were loaded")
            return

        # all the bead samples must have the same panel, use the 1st one to
        # determine the fluorescence channels
        fluoro_indices = self.bead_samples[0].fluoro_indices

        # 1st check is to make sure the # of bead samples matches the #
        # of fluorescence channels
        if bead_sample_count != len(fluoro_indices):
            raise ValueError("Number of bead samples must match the number of fluorescence channels")

        # get PnN channel names from 1st bead sample
        for f_idx in fluoro_indices:
            pnn_label = self.bead_samples[0].pnn_labels[f_idx]
            if pnn_label not in self.bead_lut:
                self.bead_lut[f_idx] = {'channel_label': pnn_label}
            else:
                raise ValueError("Duplicate channel labels are not supported")

        # now, determine which bead file goes with which channel, and make sure
        # they all have the same channels
        for i, bs in enumerate(self.bead_samples):
            # check file name for a match with a channel
            if bs.fluoro_indices != fluoro_indices:
                raise ValueError("All bead samples must have the same channel labels")

            for chan_idx, lut in self.bead_lut.items():
                # file names typically don't have the "-A", "-H', or "-W" sub-strings
                pnn_label = lut['channel_label'].replace("-A", "")

                if pnn_label in bs.original_filename:
                    lut['bead_index'] = i

    def calculate_compensation_from_beads(self):
        if len(self.bead_lut) == 0:
            warnings.warn("No bead samples were loaded")
            return

        header = []
        comp_values = []
        for chan_idx in sorted(self.bead_lut.keys()):
            header.append(self.bead_lut[chan_idx]['channel_label'])
            bead_idx = self.bead_lut[chan_idx]['bead_index']

            x = self.bead_samples[bead_idx].get_raw_events()[:, chan_idx]
            good_events = x < (2 ** 18) - 1
            x = x[good_events]

            comp_row_values = []
            for chan_idx2 in sorted(self.bead_lut.keys()):
                if chan_idx == chan_idx2:
                    comp_row_values.append('1.0')
                else:
                    y = self.bead_samples[bead_idx].get_raw_events()[:, chan_idx2]
                    y = y[good_events]
                    rlm_res = sm.RLM(y, x).fit()

                    comp_row_values.append("%.8f" % rlm_res.params[0])

            comp_values.append(",".join(comp_row_values))

        header = ",".join(header)
        comp_values = "\n".join(comp_values)

        comp = "\n".join([header, comp_values])

        # TODO: this should return a Matrix instance
        return comp

    def analyze_samples(self, verbose=False):
        # Don't save just the DataFrame report, save the entire
        # GatingResults objects for each sample, since we'll need the gate
        # indices for each sample. Add convenience functions inside
        # GatingResults class for conveniently extracting information from
        # the DataFrame
        results = gate_samples(self.gating_strategy, self.samples, verbose)

        all_reports = [res.report for res in results]
        self.report = pd.concat(all_reports)

        self._results = {}
        for r in results:
            self._results[r.sample_id] = r

    def calculate_tsne(
            self,
            n_dims=2,
            ignore_scatter=True,
            scale_scatter=True,
            transform=None,
            subsample=True
    ):
        """
        Performs dimensional reduction using the TSNE algorithm
        :param n_dims: Number of dimensions to which the source data is reduced
        :param ignore_scatter: If True, the scatter channels are excluded
        :param scale_scatter: If True, the scatter channel data is scaled to be
            in the same range as the fluorescent channel data. If
            ignore_scatter is True, this option has no effect.
        :param transform: A Transform instance to apply to events
        :param subsample: Whether to sub-sample events from FCS files (default: True)
        :return: Dictionary of TSNE results where the keys are the FCS sample
            IDs and the values are the TSNE data for events with n_dims
        """
        tsne_events = None
        sample_events_lut = {}

        for s in self.samples:
            # Determine channels to include for TSNE analysis
            if ignore_scatter:
                tsne_indices = s.fluoro_indices
            else:
                # need to get all channel indices except time
                tsne_indices = list(range(len(self.samples[0].channels)))
                tsne_indices.remove(s.get_channel_index('Time'))

                # TODO: implement scale_scatter option
                if scale_scatter:
                    pass

            s_events = s.get_raw_events(subsample=subsample)

            if transform is not None:
                fluoro_indices = s.fluoro_indices
                xform_events = transform.apply(s_events[:, fluoro_indices])
                s_events[:, fluoro_indices] = xform_events

            s_events = s_events[:, tsne_indices]

            # Concatenate events for all samples, keeping track of the indices
            # belonging to each sample
            if tsne_events is None:
                sample_events_lut[s.original_filename] = {
                    'start': 0,
                    'end': len(s_events),
                    'channel_indices': tsne_indices,
                    'events': s_events
                }
                tsne_events = s_events
            else:
                sample_events_lut[s.original_filename] = {
                    'start': len(tsne_events),
                    'end': len(tsne_events) + len(s_events),
                    'channel_indices': tsne_indices,
                    'events': s_events
                }
                tsne_events = np.vstack([tsne_events, s_events])

        # Scale data & run TSNE
        tsne_events = StandardScaler().fit(tsne_events).transform(tsne_events)
        tsne_results = MulticoreTSNE(n_components=n_dims, n_jobs=8).fit_transform(tsne_events)

        # Split TSNE results back into individual samples as a dictionary
        for k, v in sample_events_lut.items():
            v['tsne_results'] = tsne_results[v['start']:v['end'], :]

        # Return split results
        return sample_events_lut

    def plot_tsne(
            self,
            tsne_results,
            x_min=None,
            x_max=None,
            y_min=None,
            y_max=None,
            fig_size=(8, 8)
    ):
        for s_id, s_results in tsne_results.items():
            sample = self.get_sample(s_id)
            tsne_events = s_results['tsne_results']

            for i, chan_idx in enumerate(s_results['channel_indices']):
                labels = sample.channels[str(chan_idx + 1)]

                x = tsne_events[:, 0]
                y = tsne_events[:, 1]

                # determine padding to keep min/max events off the edge,
                # but only if user didn't specify the limits
                x_min, x_max = _utils.calculate_extent(x, d_min=x_min, d_max=x_max, pad=0.02)
                y_min, y_max = _utils.calculate_extent(y, d_min=y_min, d_max=y_max, pad=0.02)

                z = s_results['events'][:, i]
                z_sort = np.argsort(z)
                z = z[z_sort]
                x = x[z_sort]
                y = y[z_sort]

                fig, ax = plt.subplots(figsize=fig_size)
                ax.set_title(" - ".join([s_id, labels['PnN'], labels['PnS']]))

                ax.set_xlim([x_min, x_max])
                ax.set_ylim([y_min, y_max])

                seaborn.scatterplot(
                    x,
                    y,
                    hue=z,
                    palette=cm.get_cmap('rainbow'),
                    legend=False,
                    s=11,
                    linewidth=0,
                    alpha=0.7
                )

                file_name = s_id
                file_name = file_name.replace(".fcs", "")
                file_name = "_".join([file_name, labels['PnN'], labels['PnS']])
                file_name = file_name.replace("/", "_")
                file_name += ".png"
                plt.savefig(file_name)

    @staticmethod
    def plot_tsne_difference(
            tsne_results1,
            tsne_results2,
            x_min=None,
            x_max=None,
            y_min=None,
            y_max=None,
            fig_size=(16, 16),
            export_fcs=False,
            export_cnt=20000,
            fcs_export_dir=None
    ):
        # fit an array of size [Ndim, Nsamples]
        kde1 = gaussian_kde(
            np.vstack(
                [
                    tsne_results1[:, 0],
                    tsne_results1[:, 1]
                ]
            )
        )
        kde2 = gaussian_kde(
            np.vstack(
                [
                    tsne_results2[:, 0],
                    tsne_results2[:, 1]
                ]
            )
        )

        # evaluate on a regular grid
        x_grid = np.linspace(x_min, x_max, 250)
        y_grid = np.linspace(y_min, y_max, 250)
        x_grid, y_grid = np.meshgrid(x_grid, y_grid)
        xy_grid = np.vstack([x_grid.ravel(), y_grid.ravel()])

        z1 = kde1.evaluate(xy_grid)
        z2 = kde2.evaluate(xy_grid)

        z = z2 - z1

        if export_fcs:
            z_g2 = z.copy()
            z_g2[z_g2 < 0] = 0
            z_g1 = z.copy()
            z_g1[z_g1 > 0] = 0
            z_g1 = np.abs(z_g1)

            z_g2_norm = [float(i) / sum(z_g2) for i in z_g2]
            z_g1_norm = [float(i) / sum(z_g1) for i in z_g1]

            cdf = np.cumsum(z_g2_norm)
            cdf = cdf / cdf[-1]
            values = np.random.rand(export_cnt)
            value_bins = np.searchsorted(cdf, values)
            new_g2_events = np.array([xy_grid[:, i] for i in value_bins])

            cdf = np.cumsum(z_g1_norm)
            cdf = cdf / cdf[-1]
            values = np.random.rand(export_cnt)
            value_bins = np.searchsorted(cdf, values)
            new_g1_events = np.array([xy_grid[:, i] for i in value_bins])

            pnn_labels = ['tsne_0', 'tsne_1']

            fh = open(os.path.join(fcs_export_dir, "tsne_group_1.fcs"), 'wb')
            create_fcs(new_g1_events.flatten(), pnn_labels, fh)
            fh.close()

            fh = open(os.path.join(fcs_export_dir, "tsne_group_2.fcs"), 'wb')
            create_fcs(new_g2_events.flatten(), pnn_labels, fh)
            fh.close()

        # Plot the result as an image
        _, _ = plt.subplots(figsize=fig_size)
        plt.imshow(z.reshape(x_grid.shape),
                   origin='lower', aspect='auto',
                   extent=[x_min, x_max, y_min, y_max],
                   cmap='bwr')
        plt.show()

    def get_gate_indices(self, sample_id, gate_id):
        gating_result = self._results[sample_id]
        return gating_result.get_gate_indices(gate_id)

    def plot_gate(self, sample_id, gate_id):
        in_gate = self.get_gate_indices(sample_id, gate_id)
        gate = self.gates[gate_id]
        sample = self.get_sample(sample_id)
        events, dim_idx, dim_min, dim_max, new_dims = gate.preprocess_sample_events(sample)

        dim_labels = [dim.label for dim in gate.dimensions]

        z_colors = []
        for e in in_gate:
            if e:
                z_colors.append("#0000ffff")
            else:
                z_colors.append("#99999999")

        tools = "crosshair,pan,zoom_in,zoom_out,box_zoom,undo,redo,reset,save,"
        p = figure(
            tools=tools,
            x_range=(dim_min[0], dim_max[0]),
            y_range=(dim_min[1], dim_max[1]),
            title=sample_id
        )
        p.title.align = 'center'

        p.xaxis.axis_label = dim_labels[0]
        p.yaxis.axis_label = dim_labels[1]

        if dim_max[1] > dim_max[0]:
            radius_dimension = 'y'
            radius = 0.003 * dim_max[1]
        else:
            radius_dimension = 'x'
            radius = 0.003 * dim_max[0]

        p.scatter(
            events[:, dim_idx[0]],
            events[:, dim_idx[1]],
            radius=radius,
            radius_dimension=radius_dimension,
            fill_color=z_colors,
            fill_alpha=0.4,
            line_color=None
        )

        show(p)

        return p
