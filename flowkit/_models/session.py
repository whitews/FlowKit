import io
import os
import copy
from glob import glob
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from MulticoreTSNE import MulticoreTSNE
from bokeh.models import Span, BoxAnnotation
import seaborn
from matplotlib import cm
import matplotlib.pyplot as plt
from flowio.create_fcs import create_fcs
from flowkit import Sample, GatingStrategy, Matrix, gates, _xml_utils, _plot_utils
import warnings

try:
    import multiprocessing as mp
    multi_proc = False
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
        elif os.path.isfile(fcs_samples):
            sample_list = get_samples_from_paths([fcs_samples])

    return sample_list


# gate_sample & gate_samples are multi-proc wrappers for GatingStrategy gate_sample method
# These are functions external to GatingStrategy as mp doesn't work well for class methods
def gate_sample(data):
    gating_strategy = data[0]
    sample = data[1]
    verbose = data[2]
    return gating_strategy.gate_sample(sample, verbose=verbose)


def gate_samples(gating_strategies, samples, verbose):
    # TODO: Looks like multiprocessing can fail for very large workloads (lots of gates), maybe due
    #       to running out of memory. Will investigate further, but for now maybe provide an option
    #       for turning off multiprocessing so end user can avoid this issue if it occurs.
    if multi_proc:
        if len(samples) < mp.cpu_count():
            proc_count = len(samples)
        else:
            proc_count = mp.cpu_count() - 1  # leave a CPU free just to be nice

        pool = mp.Pool(processes=proc_count)
        data = [(gating_strategies[i], sample, verbose) for i, sample in enumerate(samples)]
        all_results = pool.map(gate_sample, data)
    else:
        all_results = []
        for i, sample in enumerate(samples):
            results = gating_strategies[i].gate_sample(sample, verbose=verbose)
            all_results.append(results)

    return all_results


class Session(object):
    def __init__(self, fcs_samples=None, subsample_count=10000):
        self.subsample_count = subsample_count
        self.sample_lut = {}
        self._results_lut = {}
        self._sample_group_lut = {}
        self.report = None

        self.add_sample_group('default')

        self.add_samples(fcs_samples)

    def add_sample_group(self, group_name, gating_strategy=None):
        if group_name in self._sample_group_lut:
            warnings.warn("A sample group with this name already exists...skipping")
            return

        if isinstance(gating_strategy, GatingStrategy):
            gating_strategy = gating_strategy
        elif isinstance(gating_strategy, str) or isinstance(gating_strategy, io.IOBase):
            # assume a path to an XML file representing either a GatingML document or FlowJo workspace
            gating_strategy = _xml_utils.parse_gating_xml(gating_strategy)
        elif gating_strategy is None:
            gating_strategy = GatingStrategy()
        else:
            raise ValueError(
                "'gating_strategy' must be either a GatingStrategy instance or a path to a GatingML document"
            )

        self._sample_group_lut[group_name] = {
            'template': gating_strategy,
            'samples': {}
        }

    def import_flowjo_workspace(self, workspace_file_or_path):
        wsp_sample_groups = _xml_utils.parse_wsp(workspace_file_or_path)
        for group_name, sample_data in wsp_sample_groups.items():
            for sample, data_dict in sample_data.items():
                if sample not in self.sample_lut:
                    self.sample_lut[sample] = None
                    msg = "Sample %s has not been added to the session. \n" % sample
                    msg += "A GatingStrategy was loaded for this sample ID, but the file needs to be added " \
                           "to the Session prior to running the analyze_samples method."
                    warnings.warn(msg)

                gs = GatingStrategy()

                gs.gates = {gate.id: gate for gate in data_dict['gates']}
                matrix = data_dict['compensation']
                if isinstance(matrix, Matrix):
                    gs.comp_matrices[matrix.id] = matrix
                gs.transformations = {xform.id: xform for xform in data_dict['transforms']}

                if group_name not in self._sample_group_lut:
                    self.add_sample_group(group_name, gs)

                self._sample_group_lut[group_name]['samples'][sample] = gs

    def add_samples(self, samples):
        new_samples = load_samples(samples)
        for s in new_samples:
            s.subsample_events(self.subsample_count)
            if s.original_filename in self.sample_lut:
                # sample ID may have been added via a FlowJo workspace,
                # check if Sample value is None
                if self.sample_lut[s.original_filename] is not None:
                    warnings.warn("A sample with this ID already exists...skipping")
                    continue
            self.sample_lut[s.original_filename] = s

            # all samples get added to the 'default' group
            self.assign_sample(s.original_filename, 'default')

    def assign_sample(self, sample_id, group_name):
        group = self._sample_group_lut[group_name]
        if sample_id in group['samples']:
            warnings.warn("Sample %s is already assigned to the group %s...nothing changed" % (sample_id, group_name))
            return
        template = group['template']
        group['samples'][sample_id] = copy.deepcopy(template)

    def get_sample_ids(self):
        return list(self.sample_lut.keys())

    def get_sample_groups(self):
        return list(self._sample_group_lut.keys())

    def get_group_samples(self, sample_group):
        sample_ids = self._sample_group_lut[sample_group]['samples'].keys()

        samples = []
        for s_id in sample_ids:
            samples.append(self.sample_lut[s_id])

        return samples

    def get_gate_ids(self, sample_group):
        group = self._sample_group_lut[sample_group]
        template = group['template']
        return list(template.gates.keys())

    # start pass through methods for GatingStrategy class
    def add_gate(self, gate, group_name='default'):
        # TODO: allow adding multiple gates at once, while still allowing a single gate. Check if list or Gate instance
        group = self._sample_group_lut[group_name]
        template = group['template']
        s_members = group['samples']

        # first, add gate to template, then add a copy to each group sample gating strategy
        template.add_gate(copy.deepcopy(gate))
        for s_id, s_strategy in s_members.items():
            s_strategy.add_gate(copy.deepcopy(gate))

    def add_transform(self, transform, group_name='default'):
        group = self._sample_group_lut[group_name]
        template = group['template']
        s_members = group['samples']

        # first, add gate to template, then add a copy to each group sample gating strategy
        template.add_transform(copy.deepcopy(transform))
        for s_id, s_strategy in s_members.items():
            s_strategy.add_transform(copy.deepcopy(transform))

    def add_comp_matrix(self, matrix, group_name='default'):
        # TODO: GatingStrategy method add_comp_matrix accepts only Matrix instances, so this pass through does as well.
        #       Consider adding a pass through Session method to parse comp matrices for conveniently getting a
        #       Matrix instance from a comp source (npy, CSV, str, etc.)
        group = self._sample_group_lut[group_name]
        template = group['template']
        s_members = group['samples']

        # first, add gate to template, then add a copy to each group sample gating strategy
        template.add_comp_matrix(copy.deepcopy(matrix))
        for s_id, s_strategy in s_members.items():
            s_strategy.add_comp_matrix(copy.deepcopy(matrix))

    def get_parent_gate_id(self, group_name, gate_id):
        group = self._sample_group_lut[group_name]
        template = group['template']
        gate = template.get_gate_by_reference(gate_id)
        return gate.parent

    def get_gate_by_reference(self, group_name, sample_id, gate_id):
        group = self._sample_group_lut[group_name]
        gating_strategy = group['samples'][sample_id]
        gate = gating_strategy.get_gate_by_reference(gate_id)
        return gate

    def get_gate_hierarchy(self, sample_group, output='ascii'):
        return self._sample_group_lut[sample_group]['template'].get_gate_hierarchy(output)
    # end pass through methods for GatingStrategy

    def export_gml(self, file_handle, group_name, sample_id=None):
        group = self._sample_group_lut[group_name]
        if sample_id is not None:
            gating_strategy = group['samples'][sample_id]
        else:
            gating_strategy = group['template']
        _xml_utils.export_gatingml(gating_strategy, file_handle)

    def get_sample(self, sample_id):
        return self.sample_lut[sample_id]

    @staticmethod
    def _process_bead_samples(bead_samples):
        # do nothing if there are no bead samples
        bead_sample_count = len(bead_samples)
        if bead_sample_count == 0:
            warnings.warn("No bead samples were loaded")
            return

        bead_lut = {}

        # all the bead samples must have the same panel, use the 1st one to
        # determine the fluorescence channels
        fluoro_indices = bead_samples[0].fluoro_indices

        # 1st check is to make sure the # of bead samples matches the #
        # of fluorescence channels
        if bead_sample_count != len(fluoro_indices):
            raise ValueError("Number of bead samples must match the number of fluorescence channels")

        # get PnN channel names from 1st bead sample
        pnn_labels = []
        for f_idx in fluoro_indices:
            pnn_label = bead_samples[0].pnn_labels[f_idx]
            if pnn_label not in pnn_labels:
                pnn_labels.append(pnn_label)
                bead_lut[f_idx] = {'pnn_label': pnn_label}
            else:
                raise ValueError("Duplicate channel labels are not supported")

        # now, determine which bead file goes with which channel, and make sure
        # they all have the same channels
        for i, bs in enumerate(bead_samples):
            # check file name for a match with a channel
            if bs.fluoro_indices != fluoro_indices:
                raise ValueError("All bead samples must have the same channel labels")

            for chan_idx, lut in bead_lut.items():
                # file names typically don't have the "-A", "-H', or "-W" sub-strings
                pnn_label = lut['pnn_label'].replace("-A", "")

                if pnn_label in bs.original_filename:
                    lut['bead_index'] = i
                    lut['pns_label'] = bs.pns_labels[chan_idx]

        return bead_lut

    def calculate_compensation_from_beads(self, comp_bead_samples, matrix_id='comp_bead'):
        bead_samples = load_samples(comp_bead_samples)
        bead_lut = self._process_bead_samples(bead_samples)
        if len(bead_lut) == 0:
            warnings.warn("No bead samples were loaded")
            return

        detectors = []
        fluorochromes = []
        comp_values = []
        for chan_idx in sorted(bead_lut.keys()):
            detectors.append(bead_lut[chan_idx]['pnn_label'])
            fluorochromes.append(bead_lut[chan_idx]['pns_label'])
            bead_idx = bead_lut[chan_idx]['bead_index']

            x = bead_samples[bead_idx].get_raw_events()[:, chan_idx]
            good_events = x < (2 ** 18) - 1
            x = x[good_events]

            comp_row_values = []
            for chan_idx2 in sorted(bead_lut.keys()):
                if chan_idx == chan_idx2:
                    comp_row_values.append(1.0)
                else:
                    y = bead_samples[bead_idx].get_raw_events()[:, chan_idx2]
                    y = y[good_events]
                    rlm_res = sm.RLM(y, x).fit()

                    # noinspection PyUnresolvedReferences
                    comp_row_values.append(rlm_res.params[0])

            comp_values.append(comp_row_values)

        return Matrix(matrix_id, np.array(comp_values), detectors, fluorochromes)

    def analyze_samples(self, sample_group='default', verbose=False):
        # Don't save just the DataFrame report, save the entire
        # GatingResults objects for each sample, since we'll need the gate
        # indices for each sample.
        samples = self.get_group_samples(sample_group)
        if len(samples) == 0:
            warnings.warn("No samples have been assigned to sample group %s" % sample_group)
            return
        gating_strategies = []
        samples_to_run = []
        for s in samples:
            if s is None:
                # sample hasn't been added to Session
                continue
            gating_strategies.append(self._sample_group_lut[sample_group]['samples'][s.original_filename])
            samples_to_run.append(s)

        results = gate_samples(gating_strategies, samples_to_run, verbose)

        all_reports = [res.report for res in results]

        self._results_lut[sample_group] = {
            'report': pd.concat(all_reports),
            'samples': {}  # dict will have sample ID keys and results values
        }
        for r in results:
            self._results_lut[sample_group]['samples'][r.sample_id] = r

    def get_gating_results(self, sample_group, sample_id):
        gating_result = self._results_lut[sample_group]['samples'][sample_id]
        return copy.deepcopy(gating_result)

    def get_gate_indices(self, sample_group, sample_id, gate_id):
        gating_result = self._results_lut[sample_group]['samples'][sample_id]
        return gating_result.get_gate_indices(gate_id)

    def calculate_tsne(
            self,
            sample_group,
            n_dims=2,
            ignore_scatter=True,
            scale_scatter=True,
            transform=None,
            subsample=True
    ):
        """
        Performs dimensional reduction using the TSNE algorithm
        :param sample_group: The sample group on which to run TSNE
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
        samples = self.get_group_samples(sample_group)

        for s in samples:
            # Determine channels to include for TSNE analysis
            if ignore_scatter:
                tsne_indices = s.fluoro_indices
            else:
                # need to get all channel indices except time
                tsne_indices = list(range(len(samples[0].channels)))
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
                x_min, x_max = _plot_utils.calculate_extent(x, d_min=x_min, d_max=x_max, pad=0.02)
                y_min, y_max = _plot_utils.calculate_extent(y, d_min=y_min, d_max=y_max, pad=0.02)

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

    def plot_gate(
            self,
            sample_group,
            sample_id,
            gate_id,
            x_min=None,
            x_max=None,
            y_min=None,
            y_max=None,
            color_density=True
    ):
        group = self._sample_group_lut[sample_group]
        gating_strategy = group['samples'][sample_id]
        gate = gating_strategy.get_gate_by_reference(gate_id)

        # dim count determines if we need a histogram, scatter, or multi-scatter
        dim_count = len(gate.dimensions)
        if dim_count == 1:
            gate_type = 'hist'
        elif dim_count == 2:
            gate_type = 'scatter'
        else:
            raise NotImplementedError("Plotting of gates with >2 dimensions is not yet supported")

        sample_to_plot = self.get_sample(sample_id)
        events, dim_idx, dim_min, dim_max, new_dims = gate.preprocess_sample_events(
            sample_to_plot,
            copy.deepcopy(gating_strategy)
        )
        if len(new_dims) > 0:
            raise NotImplementedError("Plotting of RatioDimenstions is not yet supported.")
        if isinstance(gate, gates.QuadrantGate):
            raise NotImplementedError("Plotting of quadrant gates is not supported in this version of FlowKit")
        x = events[sample_to_plot.subsample_indices, dim_idx[0]]

        dim_labels = [dim.label for dim in gate.dimensions]
        plot_title = "%s - %s - %s" % (sample_id, sample_group, gate_id)

        if gate_type == 'scatter':
            y = events[sample_to_plot.subsample_indices, dim_idx[1]]

            p = _plot_utils.plot_scatter(
                x,
                y,
                dim_labels,
                title=plot_title,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                color_density=color_density
            )
        elif gate_type == 'hist':
            p = _plot_utils.plot_histogram(x, dim_labels[0], title=plot_title)
        else:
            raise NotImplementedError("Only histograms and scatter plots are supported in this version of FlowKit")

        g_line_color = "#1F77B4"
        g_line_width = 3
        g_fill_color = 'lime'
        g_fill_alpha = 0.05

        if isinstance(gate, gates.PolygonGate):
            x_coords, y_coords = list(zip(*[v.coordinates for v in gate.vertices]))
            p.patch(
                x_coords,
                y_coords,
                fill_color=g_fill_color,
                fill_alpha=g_fill_alpha,
                line_width=g_line_width
            )
        elif isinstance(gate, gates.RectangleGate):
            # rectangle gates in GatingML may not actually be rectangles, as the min/max for the dimensions
            # are options. So, if any of the dim min/max values are missing it is essentially a set of ranges.

            if None in dim_min or None in dim_max or dim_count == 1:
                renderers = []
                left = None
                right = None
                bottom = None
                top = None

                if dim_min[0] is not None:
                    left = dim_min[0]
                    renderers.append(
                        Span(location=left, dimension='height', line_width=g_line_width, line_color=g_line_color)
                    )
                if dim_max[0] is not None:
                    right = dim_max[0]
                    renderers.append(
                        Span(location=right, dimension='height', line_width=g_line_width, line_color=g_line_color)
                    )
                if dim_count > 1:
                    if dim_min[1] is not None:
                        bottom = dim_min[1]
                        renderers.append(
                            Span(location=bottom, dimension='width', line_width=g_line_width, line_color=g_line_color)
                        )
                    if dim_max[1] is not None:
                        top = dim_max[1]
                        renderers.append(
                            Span(location=top, dimension='width', line_width=g_line_width, line_color=g_line_color)
                        )

                mid_box = BoxAnnotation(
                    left=left,
                    right=right,
                    bottom=bottom,
                    top=top,
                    fill_alpha=g_fill_alpha,
                    fill_color=g_fill_color
                )
                renderers.append(mid_box)

                p.renderers.extend(renderers)
            else:
                # a true rectangle
                x_center = (dim_min[0] + dim_max[0]) / 2.0
                y_center = (dim_min[1] + dim_max[1]) / 2.0
                x_width = dim_max[0] - dim_min[0]
                y_height = dim_max[1] - dim_min[1]
                p.rect(
                    x_center,
                    y_center,
                    x_width,
                    y_height,
                    fill_color=g_fill_color,
                    fill_alpha=g_fill_alpha,
                    line_width=g_line_width
                )
        else:
            raise NotImplementedError("Only polygon and rectangle gates are supported in this version of FlowKit")

        return p
