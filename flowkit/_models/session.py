"""
Session class
"""
import io
import os
import copy
from glob import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from MulticoreTSNE import MulticoreTSNE
import seaborn
from bokeh.models import Title
from matplotlib import cm
import matplotlib.pyplot as plt
from .._models.sample import Sample
from .._models.gating_strategy import GatingStrategy
from .._models.transforms._matrix import Matrix
from .._models import gates
from .._utils import plot_utils, xml_utils
import warnings

try:
    import multiprocessing as mp
    multi_proc = True
except ImportError:
    mp = None
    multi_proc = False


def get_samples_from_paths(sample_paths):
    sample_count = len(sample_paths)
    if multi_proc and sample_count > 1:
        if sample_count < mp.cpu_count():
            proc_count = sample_count
        else:
            proc_count = mp.cpu_count() - 1  # leave a CPU free just to be nice

        try:
            pool = mp.Pool(processes=proc_count)
            samples = pool.map(Sample, sample_paths)
        except Exception as e:
            # noinspection PyUnboundLocalVariable
            pool.close()
            raise e
        pool.close()
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
    sample_count = len(samples)
    if multi_proc and sample_count > 1:
        if sample_count < mp.cpu_count():
            proc_count = sample_count
        else:
            proc_count = mp.cpu_count() - 1  # leave a CPU free just to be nice

        try:
            pool = mp.Pool(processes=proc_count)
            data = [(gating_strategies[i], sample, verbose) for i, sample in enumerate(samples)]
            all_results = pool.map(gate_sample, data)
        except Exception as e:
            # noinspection PyUnboundLocalVariable
            pool.close()
            raise e
        pool.close()
    else:
        all_results = []
        for i, sample in enumerate(samples):
            results = gating_strategies[i].gate_sample(sample, verbose=verbose)
            all_results.append(results)

    return all_results


class Session(object):
    """
    The Session class is intended as the main interface in FlowKit for complex flow cytometry analysis.
    A Session represents a collection of gating strategies and FCS samples. FCS samples are added and assigned to sample
    groups, and each sample group has a single gating strategy template. The gates in a template can be customized
    per sample.

    :param fcs_samples: a list of either file paths or Sample instances
    :param subsample_count: Number of events to use as a sub-sample. If the number of
        events in the Sample is less than the requested sub-sample count, then the
        maximum number of available events is used for the sub-sample.
    """
    def __init__(self, fcs_samples=None, subsample_count=10000):
        self.subsample_count = subsample_count
        self.sample_lut = {}
        self._results_lut = {}
        self._sample_group_lut = {}

        self.add_sample_group('default')

        self.add_samples(fcs_samples)

    def add_sample_group(self, group_name, gating_strategy=None):
        """
        Create a new sample group to the session. The group name must be unique to the session.

        :param group_name: a text string representing the sample group
        :param gating_strategy: a gating strategy instance to use for the group template. If None, then a new, blank
            gating strategy will be created.
        :return: None
        """
        if group_name in self._sample_group_lut:
            warnings.warn("A sample group with this name already exists...skipping")
            return

        if isinstance(gating_strategy, GatingStrategy):
            gating_strategy = gating_strategy
        elif isinstance(gating_strategy, str) or isinstance(gating_strategy, io.IOBase):
            # assume a path to an XML file representing either a GatingML document or FlowJo workspace
            gating_strategy = xml_utils.parse_gating_xml(gating_strategy)
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

    def import_flowjo_workspace(self, workspace_file_or_path, ignore_missing_files=False):
        """
        Imports a FlowJo workspace (version 10+) into the Session. Each sample group in the workspace will
        be a sample group in the FlowKit session. Referenced samples in the workspace will be imported as
        references in the session. Ideally, these samples should have already been loaded into the session,
        and a warning will be issued for each sample reference that has not yet been loaded.
        Support for FlowJo workspaces is limited to the following
        features:

          - Transformations:

            - linear
            - log
            - logicle
          - Gates:

            - rectangle
            - polygon
            - ellipse
            - quadrant
            - range

        :param workspace_file_or_path: WSP workspace file as a file name/path, file object, or file-like object
        :param ignore_missing_files: Controls whether UserWarning messages are issued for FCS files found in the
            workspace that have not yet been loaded in the Session. Default is False, displaying warnings.
        :return: None
        """
        wsp_sample_groups = xml_utils.parse_wsp(workspace_file_or_path)
        for group_name, sample_data in wsp_sample_groups.items():
            for sample, data_dict in sample_data.items():
                if sample not in self.sample_lut:
                    self.sample_lut[sample] = None
                    if not ignore_missing_files:
                        msg = "Sample %s has not been added to the session. \n" % sample
                        msg += "A GatingStrategy was loaded for this sample ID, but the file needs to be added " \
                               "to the Session prior to running the analyze_samples method."
                        warnings.warn(msg)

                gs = GatingStrategy()

                # TODO: change keys to tuple of gate ID, parent ID so gates can be "reused" for different branches
                for gate_node in data_dict['gates'].descendants:
                    gs.add_gate(gate_node.gate)

                matrix = data_dict['compensation']
                if isinstance(matrix, Matrix):
                    gs.comp_matrices[matrix.id] = matrix
                gs.transformations = {xform.id: xform for xform in data_dict['transforms']}

                if group_name not in self._sample_group_lut:
                    self.add_sample_group(group_name, gs)

                self._sample_group_lut[group_name]['samples'][sample] = gs

    def add_samples(self, samples):
        """
        Adds FCS samples to the session. All added samples will be added to the 'default' sample group.

        :param samples: a list of Sample instances
        :return: None
        """
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
        """
        Assigns a sample ID to a sample group. Samples can belong to more than one sample group.

        :param sample_id: Sample ID to assign to the specified sample group
        :param group_name: name of sample group to which the sample will be assigned
        :return: None
        """
        group = self._sample_group_lut[group_name]
        if sample_id in group['samples']:
            warnings.warn("Sample %s is already assigned to the group %s...nothing changed" % (sample_id, group_name))
            return
        template = group['template']
        group['samples'][sample_id] = copy.deepcopy(template)

    def get_sample_ids(self):
        """
        Retrieve the list of Sample IDs that have been loaded or referenced in the Session
        :return: list of Sample ID strings
        """
        return list(self.sample_lut.keys())

    def get_sample_groups(self):
        """
        Retrieve the list of sample group labels defined in the Session
        :return: list of sample group ID strings
        """
        return list(self._sample_group_lut.keys())

    def get_group_sample_ids(self, sample_group):
        """
        Retrieve the list of Sample IDs belonging to the specified sample group
        :param sample_group: a text string representing the sample group
        :return: list of Sample IDs
        """
        return self._sample_group_lut[sample_group]['samples'].keys()

    def get_group_samples(self, sample_group):
        """
        Retrieve the list of Sample instances belonging to the specified sample group
        :param sample_group: a text string representing the sample group
        :return: list of Sample instances
        """
        sample_ids = self.get_group_sample_ids(sample_group)

        samples = []
        for s_id in sample_ids:
            samples.append(self.sample_lut[s_id])

        return samples

    def get_gate_ids(self, sample_group):
        """
        Retrieve the list of gate IDs defined in the specified sample group
        :param sample_group: a text string representing the sample group
        :return: list of gate ID strings
        """
        group = self._sample_group_lut[sample_group]
        template = group['template']
        return template.get_gate_ids()

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

    def get_transform(self, group_name, sample_id, transform_id):
        group = self._sample_group_lut[group_name]
        gating_strategy = group['samples'][sample_id]
        xform = gating_strategy.get_transform(transform_id)
        return xform

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

    def get_comp_matrix(self, group_name, sample_id, matrix_id):
        group = self._sample_group_lut[group_name]
        gating_strategy = group['samples'][sample_id]
        comp_mat = gating_strategy.get_comp_matrix(matrix_id)
        return comp_mat

    def get_parent_gate_id(self, group_name, gate_id):
        group = self._sample_group_lut[group_name]
        template = group['template']
        gate = template.get_gate(gate_id)
        return gate.parent

    def get_gate(self, group_name, sample_id, gate_id, gate_path=None):
        group = self._sample_group_lut[group_name]
        gating_strategy = group['samples'][sample_id]
        gate = gating_strategy.get_gate(gate_id, gate_path=gate_path)
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
        xml_utils.export_gatingml(gating_strategy, file_handle)

    def get_sample(self, sample_id):
        """
        Retrieve a Sample instance from the Session
        :param sample_id: a text string representing the sample
        :return: Sample instance
        """
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

    def analyze_samples(self, sample_group='default', sample_id=None, verbose=False):
        """
        Process gates for samples in a sample group. After running, results can be
        retrieved using the `get_gating_results`, `get_group_report`, and  `get_gate_indices`,
        methods.

        :param sample_group: a text string representing the sample group
        :param sample_id: optional sample ID, if specified only this sample will be processed
        :param verbose: if True, print a line for every gate processed (default is False)
        :return: None
        """
        # Don't save just the DataFrame report, save the entire
        # GatingResults objects for each sample, since we'll need the gate
        # indices for each sample.
        samples = self.get_group_samples(sample_group)
        if len(samples) == 0:
            warnings.warn("No samples have been assigned to sample group %s" % sample_group)
            return

        if sample_id is not None:
            sample_ids = self.get_group_sample_ids(sample_group)
            if sample_id not in sample_ids:
                warnings.warn("%s is not assigned to sample group %s" % (sample_id, sample_group))
                return

            samples = [self.get_sample(sample_id)]

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

    def get_group_report(self, sample_group):
        return self._results_lut[sample_group]['report']

    def get_gate_indices(self, sample_group, sample_id, gate_id, gate_path=None):
        gating_result = self._results_lut[sample_group]['samples'][sample_id]
        return gating_result.get_gate_indices(gate_id, gate_path=gate_path)

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
                x_min, x_max = plot_utils.calculate_extent(x, d_min=x_min, d_max=x_max, pad=0.02)
                y_min, y_max = plot_utils.calculate_extent(y, d_min=y_min, d_max=y_max, pad=0.02)

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

    def plot_gate(
            self,
            sample_group,
            sample_id,
            gate_id,
            gate_path=None,
            x_min=None,
            x_max=None,
            y_min=None,
            y_max=None,
            color_density=True
    ):
        """
        Returns an interactive plot for the specified gate. The type of plot is determined by the number of
         dimensions used to define the gate: single dimension gates will be histograms, 2-D gates will be returned
         as a scatter plot.

        :param sample_group: The sample group containing the sample ID (and, optionally the gate ID)
        :param sample_id: The sample ID for the FCS sample to plot
        :param gate_id: Gate ID to filter events (only events within the given gate will be plotted)
        :param gate_path: list of gate IDs for full set of gate ancestors. Required if gate_id is ambiguous
        :param x_min: Lower bound of x-axis. If None, channel's min value will
            be used with some padding to keep events off the edge of the plot.
        :param x_max: Upper bound of x-axis. If None, channel's max value will
            be used with some padding to keep events off the edge of the plot.
        :param y_min: Lower bound of y-axis. If None, channel's min value will
            be used with some padding to keep events off the edge of the plot.
        :param y_max: Upper bound of y-axis. If None, channel's max value will
            be used with some padding to keep events off the edge of the plot.
        :param color_density: Whether to color the events by density, similar
            to a heat map. Default is True.
        :return: A Bokeh Figure object containing the interactive scatter plot.
        """
        group = self._sample_group_lut[sample_group]
        gating_strategy = group['samples'][sample_id]
        gate = gating_strategy.get_gate(gate_id, gate_path)

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

        # get parent gate results to display only those events
        if gate.parent is not None:
            parent_results = gating_strategy.gate_sample(sample_to_plot, gate.parent)
            is_parent_event = parent_results.get_gate_indices(gate.parent)
            is_subsample = np.zeros(sample_to_plot.event_count, dtype=np.bool)
            is_subsample[sample_to_plot.subsample_indices] = True
            idx_to_plot = np.logical_and(is_parent_event, is_subsample)
        else:
            idx_to_plot = sample_to_plot.subsample_indices

        if len(new_dims) > 0:
            raise NotImplementedError("Plotting of RatioDimensions is not yet supported.")

        x = events[idx_to_plot, dim_idx[0]]

        dim_labels = []

        x_index = dim_idx[0]
        x_pnn_label = sample_to_plot.pnn_labels[x_index]
        y_pnn_label = None

        if sample_to_plot.pns_labels[x_index] != '':
            dim_labels.append('%s (%s)' % (sample_to_plot.pns_labels[x_index], x_pnn_label))
        else:
            dim_labels.append(sample_to_plot.pnn_labels[x_index])

        if len(dim_idx) > 1:
            y_index = dim_idx[1]
            y_pnn_label = sample_to_plot.pnn_labels[y_index]

            if sample_to_plot.pns_labels[y_index] != '':
                dim_labels.append('%s (%s)' % (sample_to_plot.pns_labels[y_index], y_pnn_label))
            else:
                dim_labels.append(sample_to_plot.pnn_labels[y_index])

        if gate_type == 'scatter':
            y = events[idx_to_plot, dim_idx[1]]

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
        elif gate_type == 'hist':
            p = plot_utils.plot_histogram(x, dim_labels[0])
        else:
            raise NotImplementedError("Only histograms and scatter plots are supported in this version of FlowKit")

        if isinstance(gate, gates.PolygonGate):
            source, glyph = plot_utils.render_polygon(gate.vertices)
            p.add_glyph(source, glyph)
        elif isinstance(gate, gates.EllipsoidGate):
            ellipse = plot_utils.calculate_ellipse(
                gate.coordinates[0],
                gate.coordinates[1],
                gate.covariance_matrix,
                gate.distance_square
            )
            p.add_glyph(ellipse)
        elif isinstance(gate, gates.RectangleGate):
            # rectangle gates in GatingML may not actually be rectangles, as the min/max for the dimensions
            # are options. So, if any of the dim min/max values are missing it is essentially a set of ranges.

            if None in dim_min or None in dim_max or dim_count == 1:
                renderers = plot_utils.render_ranges(dim_min, dim_max)

                p.renderers.extend(renderers)
            else:
                # a true rectangle
                rect = plot_utils.render_rectangle(dim_min, dim_max)
                p.add_glyph(rect)
        elif isinstance(gate, gates.QuadrantGate):
            x_locations = []
            y_locations = []

            for div in gate.dimensions:
                if div.dimension_ref == x_pnn_label:
                    x_locations.extend(div.values)
                elif div.dimension_ref == y_pnn_label and y_pnn_label is not None:
                    y_locations.extend(div.values)

            renderers = plot_utils.render_dividers(x_locations, y_locations)
            p.renderers.extend(renderers)
        else:
            raise NotImplementedError(
                "Plotting of %s gates is not supported in this version of FlowKit" % gate.__class__
            )

        if gate_path is not None:
            full_gate_path = gate_path[1:]  # omit 'root'
            full_gate_path.append(gate_id)
            sub_title = ' > '.join(full_gate_path)

            # truncate beginning of long gate paths
            if len(sub_title) > 72:
                sub_title = '...' + sub_title[-69:]
            p.add_layout(
                Title(text=sub_title, text_font_style="italic", text_font_size="1em", align='center'),
                'above'
            )
        else:
            p.add_layout(
                Title(text=gate_id, text_font_style="italic", text_font_size="1em", align='center'),
                'above'
            )

        plot_title = "%s (%s)" % (sample_id, sample_group)
        p.add_layout(
            Title(text=plot_title, text_font_size="1.1em", align='center'),
            'above'
        )

        return p

    def plot_scatter(
            self,
            sample_id,
            x_dim,
            y_dim,
            sample_group='default',
            gate_id=None,
            subsample=False,
            color_density=True,
            x_min=None,
            x_max=None,
            y_min=None,
            y_max=None
    ):
        """
        Returns an interactive scatter plot for the specified channel data.

        :param sample_id: The sample ID for the FCS sample to plot
        :param x_dim:  Dimension instance to use for the x-axis data
        :param y_dim: Dimension instance to use for the y-axis data
        :param sample_group: The sample group containing the sample ID (and, optionally the gate ID)
        :param gate_id: Gate ID to filter events (only events within the given gate will be plotted)
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
        sample = self.get_sample(sample_id)
        group = self._sample_group_lut[sample_group]
        gating_strategy = group['samples'][sample_id]

        x_index = sample.get_channel_index(x_dim.label)
        y_index = sample.get_channel_index(y_dim.label)

        x_comp_ref = x_dim.compensation_ref
        x_xform_ref = x_dim.transformation_ref

        y_comp_ref = y_dim.compensation_ref
        y_xform_ref = y_dim.transformation_ref

        if x_comp_ref is not None and x_comp_ref != 'uncompensated':
            x_comp = gating_strategy.get_comp_matrix(x_dim.compensation_ref)
            comp_events = x_comp.apply(sample)
            x = comp_events[:, x_index]
        else:
            # not doing sub-sample here, will do later with bool AND
            x = sample.get_channel_data(x_index, source='raw', subsample=False)

        if y_comp_ref is not None and x_comp_ref != 'uncompensated':
            # this is likely unnecessary as the x & y comp should be the same,
            # but requires more conditionals to cover
            y_comp = gating_strategy.get_comp_matrix(x_dim.compensation_ref)
            comp_events = y_comp.apply(sample)
            y = comp_events[:, y_index]
        else:
            # not doing sub-sample here, will do later with bool AND
            y = sample.get_channel_data(y_index, source='raw', subsample=False)

        if x_xform_ref is not None:
            x_xform = gating_strategy.get_transform(x_xform_ref)
            x = x_xform.apply(x.reshape(-1, 1))[:, 0]
        if y_xform_ref is not None:
            y_xform = gating_strategy.get_transform(y_xform_ref)
            y = y_xform.apply(y.reshape(-1, 1))[:, 0]

        if gate_id is not None:
            gate_results = gating_strategy.gate_sample(sample, gate_id)
            is_gate_event = gate_results.get_gate_indices(gate_id)
            if subsample:
                is_subsample = np.zeros(sample.event_count, dtype=np.bool)
                is_subsample[sample.subsample_indices] = True
            else:
                is_subsample = np.ones(sample.event_count, dtype=np.bool)

            idx_to_plot = np.logical_and(is_gate_event, is_subsample)
            x = x[idx_to_plot]
            y = y[idx_to_plot]

        dim_labels = []

        if sample.pns_labels[x_index] != '':
            dim_labels.append('%s (%s)' % (sample.pns_labels[x_index], sample.pnn_labels[x_index]))
        else:
            dim_labels.append(sample.pnn_labels[x_index])

        if sample.pns_labels[y_index] != '':
            dim_labels.append('%s (%s)' % (sample.pns_labels[y_index], sample.pnn_labels[y_index]))
        else:
            dim_labels.append(sample.pnn_labels[y_index])

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

        p.title = Title(text=sample.original_filename, align='center')

        return p
