"""
Session class
"""
import gc
import io
import copy
import numpy as np
import pandas as pd
from bokeh.models import Title
from .._conf import debug
from .._models.gating_strategy import GatingStrategy
from .._utils import plot_utils, xml_utils, gml_write, wsp_utils, sample_utils, gating_utils
from ..exceptions import FlowKitException, GateReferenceError
import warnings


class Session(object):
    """
    The Session class enables the programmatic creation of a gating strategy or for importing
    GatingML compliant documents. A Session combines multiple Sample instances with a single
    GatingStrategy. The gates in the gating strategy can be customized per sample.

    :param fcs_samples: str or list. If given a string, it can be a directory path or a file path.
            If a directory, any .fcs files in the directory will be loaded. If a list, then it must
            be a list of file paths or a list of Sample instances. Lists of mixed types are not
            supported.
    """
    def __init__(self, gating_strategy=None, fcs_samples=None):
        self.sample_lut = {}
        self._results_lut = {}

        if isinstance(gating_strategy, GatingStrategy):
            gating_strategy = gating_strategy
        elif isinstance(gating_strategy, str) or isinstance(gating_strategy, io.IOBase):
            # assume a path to an XML file representing a GatingML document
            gating_strategy = xml_utils.parse_gating_xml(gating_strategy)
        elif gating_strategy is None:
            gating_strategy = GatingStrategy()
        else:
            raise ValueError(
                "'gating_strategy' must be a GatingStrategy instance, GatingML document path, or None"
            )

        self.gating_strategy = gating_strategy

        self.add_samples(fcs_samples)

    def __repr__(self):
        sample_count = len(self.sample_lut)

        return (
            f'{self.__class__.__name__}('
            f'{sample_count} samples)'
        )

    def add_samples(self, fcs_samples):
        """
        Adds FCS samples to the session.

        :param fcs_samples: str or list. If given a string, it can be a directory path or a file path.
            If a directory, any .fcs files in the directory will be loaded. If a list, then it must
            be a list of file paths or a list of Sample instances. Lists of mixed types are not
            supported.
        :return: None
        """
        new_samples = sample_utils.load_samples(fcs_samples)
        for s in new_samples:
            if s.id in self.sample_lut:
                warnings.warn("A sample with ID %s already exists...skipping" % s.id)
                continue
            self.sample_lut[s.id] = s

    def get_sample_ids(self):
        """
        Retrieve the list of Sample IDs in the Session.

        :return: list of Sample ID strings
        """
        return list(self.sample_lut.keys())

    def get_gate_ids(self):
        """
        Retrieve the list of gate IDs defined for the Session's gating
        strategy. The gate ID is a 2-item tuple where the first item
        is a string representing the gate name and the second item is
        a tuple of the gate path.

        :return: list of gate ID tuples
        """
        return self.gating_strategy.get_gate_ids()

    def add_gate(self, gate, gate_path, sample_id=None):
        """
        Add a Gate instance to the gating strategy. The gate ID and gate path
        must be unique in the gating strategy. Custom sample gates may be added
        by specifying an optional sample ID. Note, the gate & gate path must
        already exist prior to adding custom sample gates.

        :param gate: an instance of a Gate subclass
        :param gate_path: complete tuple of gate IDs for unique set of gate ancestors
        :param sample_id: text string for specifying given gate as a custom Sample gate
        :return: None
        """
        self.gating_strategy.add_gate(copy.deepcopy(gate), gate_path=gate_path, sample_id=sample_id)

    def rename_gate(self, gate_name, new_gate_name, gate_path=None):
        """
        Rename a gate in the gating strategy. Any custom sample gates associated with the gate
        will also be renamed.

        :param gate_name: text string of existing gate name
        :param new_gate_name: text string for new gate name
        :param gate_path: complete ordered tuple of gate names for unique set of gate ancestors.
            Required if gate_name is ambiguous
        :return: None
        """
        self.gating_strategy.rename_gate(gate_name, new_gate_name, gate_path=gate_path)

    def remove_gate(self, gate_name, gate_path=None, sample_id=None, keep_children=False):
        """
        Remove a gate from the gate tree. Any descendant gates will also be removed
        unless keep_children=True. In all cases, if a BooleanGate exists that references
        the gate to remove, a GateTreeError will be thrown indicating the BooleanGate
        must be removed prior to removing the gate.

        :param gate_name: text string of a gate name
        :param gate_path: complete tuple of gate IDs for unique set of gate ancestors.
            Required if gate_name is ambiguous
        :param sample_id: text string for Sample ID to remove only its custom Sample gate and
            retain the template gate (and other custom gates if they exist).
        :param keep_children: Whether to keep child gates. If True, the child gates will be
            remapped to the removed gate's parent. Default is False, which will delete all
            descendant gates.
        :return: None
        """
        self.gating_strategy.remove_gate(
            gate_name, gate_path=gate_path, sample_id=sample_id, keep_children=keep_children
        )

    def add_transform(self, transform_id, transform):
        """
        Add a Transform instance to use in the gating strategy.

        :param transform_id: A string identifying the transform
        :param transform: an instance of a Transform subclass
        :return: None
        """
        self.gating_strategy.add_transform(transform_id, copy.deepcopy(transform))

    def get_transforms(self):
        """
        Retrieve a dictionary LUT of transformations stored in the GatingStrategy.
        Keys are the transform IDs and values are Transform instances.

        :return: a dictionary LUT of transform IDs: Transform instances
        """
        return self.gating_strategy.transformations

    def get_transform(self, transform_id):
        """
        Retrieve a Transform stored in the gating strategy by its ID.

        :param transform_id: a text string representing a Transform ID
        :return: an instance of a Transform subclass
        """
        return self.gating_strategy.get_transform(transform_id)

    def add_comp_matrix(self, matrix_id, matrix):
        """
        Add a Matrix instance to use in the gating strategy.

        :param matrix_id: A string identifying the matrix
        :param matrix: an instance of the Matrix class
        :return: None
        """
        self.gating_strategy.add_comp_matrix(matrix_id, copy.deepcopy(matrix))

    def get_comp_matrices(self):
        """
        Retrieve a dictionary LUT of compensation Matrix instances stored in the gating strategy.

        :return: a dictionary LUT of matrix IDs: Matrix instances
        """
        return self.gating_strategy.comp_matrices

    def get_comp_matrix(self, matrix_id):
        """
        Retrieve a compensation Matrix instance stored in the gating strategy by its ID.

        :param matrix_id: a text string representing a Matrix ID
        :return: a Matrix instance
        """
        return self.gating_strategy.get_comp_matrix(matrix_id)

    def find_matching_gate_paths(self, gate_name):
        """
        Find all gate paths in the gating strategy matching the given gate name.

        :param gate_name: text string of a gate name
        :return: list of gate paths (list of tuples)
        """
        return self.gating_strategy.find_matching_gate_paths(gate_name)

    def get_child_gate_ids(self, gate_name, gate_path=None):
        """
        Retrieve list of child gate IDs given the parent gate name (and path if ambiguous)
        in the gating strategy.

        :param gate_name: text string of a gate name
        :param gate_path: complete tuple of gate IDs for unique set of gate ancestors.
            Required if gate.gate_name is ambiguous
        :return: list of Gate IDs (tuple of gate name plus gate path). Returns an empty
            list if no child gates exist.
        :raises GateReferenceError: if gate ID is not found in gating strategy or if gate
            name is ambiguous
        """
        return self.gating_strategy.get_child_gate_ids(gate_name, gate_path)

    def get_gate(self, gate_name, gate_path=None, sample_id=None):
        """
        Retrieve a gate instance by its gate ID (and sample ID for custom sample gates).

        :param gate_name: text string of a gate ID
        :param gate_path: tuple of gate IDs for unique set of gate ancestors. Required if gate_name is ambiguous
        :param sample_id: a text string representing a Sample instance. If None, the template gate is returned.
        :return: Subclass of a Gate object
        """
        return self.gating_strategy.get_gate(gate_name, gate_path=gate_path, sample_id=sample_id)

    def get_sample_gates(self, sample_id):
        """
        Retrieve all gates for a sample in the gating strategy. This returns custom sample
        gates for the specified sample ID.

        :param sample_id: a text string representing a Sample instance
        :return: list of Gate subclass instances
        """
        gate_tuples = self.gating_strategy.get_gate_ids()

        sample_gates = []

        for gate_name, ancestors in gate_tuples:
            gate = self.gating_strategy.get_gate(gate_name, gate_path=ancestors, sample_id=sample_id)
            sample_gates.append(gate)

        return sample_gates

    def get_gate_hierarchy(self, output='ascii', **kwargs):
        """
        Retrieve the hierarchy of gates in the gating strategy. Output is available
        in several formats, including text, dictionary, or JSON. If output == 'json', extra
        keyword arguments are passed to json.dumps

        :param output: Determines format of hierarchy returned, either 'ascii',
            'dict', or 'JSON' (default is 'ascii')
        :return: gate hierarchy as a text string or a dictionary
        """
        return self.gating_strategy.get_gate_hierarchy(output, **kwargs)

    def export_gml(self, file_handle, sample_id=None):
        """
        Export a GatingML 2.0 file for the gating strategy. Specify the sample ID to use
        that sample's custom gates in the exported file, otherwise the template gates
        will be exported.

        :param file_handle: file handle for exporting data
        :param sample_id: an optional text string representing a Sample instance
        :return: None
        """
        gml_write.export_gatingml(self.gating_strategy, file_handle, sample_id=sample_id)

    def export_wsp(self, file_handle, group_name):
        """
        Export a FlowJo 10 workspace file (.wsp) for the gating strategy.

        :param file_handle: file handle for exporting data
        :param group_name: a text string representing the sample group to add to the WSP file
        :return: None
        """
        samples = self.sample_lut.values()

        wsp_utils.export_flowjo_wsp(self.gating_strategy, group_name, samples, file_handle)

    def get_sample(self, sample_id):
        """
        Retrieve a Sample instance from the Session.

        :param sample_id: a text string representing the Sample to retrieve
        :return: a Sample instance
        """
        return self.sample_lut[sample_id]

    def analyze_samples(self, sample_id=None, cache_events=False, use_mp=True, verbose=False):
        """
        Process gating strategy for samples. After running, results can be retrieved
        using the `get_gating_results`, `get_report`, and  `get_gate_membership`,
        methods.

        :param sample_id: optional sample ID, if specified only this sample will be processed
        :param cache_events: Whether to cache pre-processed events (compensated and transformed). This can
            be useful to speed up processing of gates that share the same pre-processing instructions for
            the same channel data, but can consume significantly more memory space. See the related
            clear_cache method for additional information. Default is False.
        :param use_mp: Controls whether multiprocessing is used to gate samples (default is True).
            Multiprocessing can fail for large workloads (lots of samples & gates) due to running out of
            memory. If encountering memory errors, set use_mp to False (processing will take longer,
            but will use significantly less memory).
        :param verbose: if True, print a line for every gate processed (default is False)
        :return: None
        """
        # Don't save just the DataFrame report, save the entire
        # GatingResults objects for each sample, since we'll need the gate
        # indices for each sample.
        samples = self.sample_lut.values()
        if len(samples) == 0:
            warnings.warn("No samples have been loaded in the Session")
            return

        if sample_id is not None:
            samples = [self.get_sample(sample_id)]

        sample_data_to_run = []
        for s in samples:
            sample_data_to_run.append(
                {
                    'gating_strategy': self.gating_strategy,
                    'sample': s
                }
            )

            # clear any existing results
            if sample_id in self._results_lut:
                del self._results_lut[sample_id]
                gc.collect()

        results = gating_utils.gate_samples(
            sample_data_to_run,
            cache_events,
            verbose,
            use_mp=False if debug else use_mp
        )

        for r in results:
            self._results_lut[r.sample_id] = r

    def get_gating_results(self, sample_id):
        """
        Retrieve analyzed gating results gates for a sample.

        :param sample_id: a text string representing a Sample instance
        :return: GatingResults instance
        """
        try:
            gating_result = self._results_lut[sample_id]
        except KeyError:
            raise KeyError(
                "No results for %s. Have you run `analyze_samples`?" % sample_id
            )
        return copy.deepcopy(gating_result)

    def get_analysis_report(self):
        """
        Retrieve the report for the analyzed samples as a pandas DataFrame.

        :return: pandas DataFrame
        """
        all_reports = []

        for s_id, result in self._results_lut.items():
            # avoid Pandas warning about concatenating empty DataFrame instances
            if len(result.report) == 0:
                continue

            all_reports.append(result.report)

        # Explicitly setting copy=True even though the default in case
        # it ever changes. Used to do our own deep copy but Pandas was
        # already doing this, so it was getting deep copied twice.
        return pd.concat(all_reports, ignore_index=True, copy=True)

    def get_gate_membership(self, sample_id, gate_name, gate_path=None):
        """
        Retrieve a boolean array indicating gate membership for the events in the
        specified sample. Note, the same gate ID may be found in multiple gate paths,
        i.e. the gate ID can be ambiguous. In this case, specify the full gate path
        to retrieve gate indices.

        :param sample_id: a text string representing a Sample instance
        :param gate_name: text string of a gate name
        :param gate_path: complete tuple of gate IDs for unique set of gate ancestors.
            Required if gate_name is ambiguous
        :return: NumPy boolean array (length of sample event count)
        """
        gating_result = self._results_lut[sample_id]
        return gating_result.get_gate_membership(gate_name, gate_path=gate_path)

    def get_gate_events(self, sample_id, gate_name=None, gate_path=None, matrix=None, transform=None):
        """
        Retrieve a pandas DataFrame containing only the events within the specified gate.
        If an optional compensation matrix and/or a transform is provided, the returned
        event data will be compensated or transformed. If both a compensation matrix and
        a transform is provided the event data will be both compensated and transformed.

        :param sample_id: a text string representing a Sample instance
        :param gate_name: text string of a gate name. If None, all Sample events will be returned (i.e. un-gated)
        :param gate_path: complete tuple of gate IDs for unique set of gate ancestors.
            Required if gate_name is ambiguous
        :param matrix: an instance of the Matrix class
        :param transform: an instance of a Transform subclass
        :return: pandas DataFrame containing only the events within the specified gate
        """
        # TODO: re-evaluate whether this method should be removed or modified...the
        #   ambiguous transforms per channel make this tricky to implement.
        sample = self.get_sample(sample_id)
        sample = copy.deepcopy(sample)

        # default is 'raw' events
        event_source = 'raw'

        if matrix is not None:
            sample.apply_compensation(matrix)
            event_source = 'comp'
        if transform is not None:
            sample.apply_transform(transform)
            event_source = 'xform'

        events_df = sample.as_dataframe(source=event_source)

        if gate_name is not None:
            gate_idx = self.get_gate_membership(sample_id, gate_name, gate_path)
            events_df = events_df[gate_idx]

        return events_df

    def plot_gate(
            self,
            sample_id,
            gate_name,
            gate_path=None,
            subsample_count=10000,
            random_seed=1,
            x_min=None,
            x_max=None,
            y_min=None,
            y_max=None,
            color_density=True,
            bin_width=4,
            hist_bins=None
    ):
        """
        Returns an interactive plot for the specified gate. The type of plot is
        determined by the number of dimensions used to define the gate: single
        dimension gates will be histograms, 2-D gates will be returned as a
        scatter plot.

        :param sample_id: The sample ID for the FCS sample to plot
        :param gate_name: Gate name to filter events (only events within the given gate will be plotted)
        :param gate_path: tuple of gate names for full set of gate ancestors.
            Required if gate_name is ambiguous
        :param subsample_count: Number of events to use as a subsample. If the number of
            events in the Sample is less than the requested subsample count, then the
            maximum number of available events is used for the subsample.
        :param random_seed: Random seed used for subsampling events
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
        :param bin_width: Bin size to use for the color density, in units of
            event point size. Larger values produce smoother gradients.
            Default is 4 for a 4x4 grid size.
        :param hist_bins: If the gate is only in 1 dimension, this option
            controls the number of bins to use for the histogram. If None,
            the number of bins is determined by the square root rule. This
            option is ignored for any gates in more than 1 dimension.
        :return: A Bokeh Figure object containing the interactive scatter plot.
        """
        if gate_path is None:
            # verify the gate_name isn't ambiguous
            gate_paths = self.find_matching_gate_paths(gate_name)
            if len(gate_paths) > 1:
                raise GateReferenceError(
                    "Multiple gates exist with gate name '%s'. Specify a gate_path to disambiguate." % gate_name
                )
            gate_path = gate_paths[0]

        gate_id = (gate_name, gate_path)
        parent_gate_name = gate_path[-1]
        parent_gate_path = gate_path[:-1]

        sample = self.get_sample(sample_id)

        # get parent gate results to display only those events
        if parent_gate_name != 'root':
            # TODO:  make it clear to call analyze_samples prior to calling this method
            parent_event_mask = self.get_gate_membership(sample_id, parent_gate_name, parent_gate_path)
        else:
            parent_event_mask = None

        p = plot_utils.plot_gate(
            gate_id,
            self.gating_strategy,
            sample,
            subsample_count=subsample_count,
            random_seed=random_seed,
            event_mask=parent_event_mask,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            color_density=color_density,
            bin_width=bin_width,
            hist_bins=hist_bins
        )

        return p

    def plot_scatter(
            self,
            sample_id,
            x_dim,
            y_dim,
            gate_name=None,
            gate_path=None,
            subsample_count=10000,
            random_seed=1,
            color_density=True,
            bin_width=4,
            x_min=None,
            x_max=None,
            y_min=None,
            y_max=None,
            height=600,
            width=600
    ):
        """
        Returns an interactive scatter plot for the specified channel data.

        :param sample_id: The sample ID for the FCS sample to plot
        :param x_dim:  Dimension instance to use for the x-axis data
        :param y_dim: Dimension instance to use for the y-axis data
        :param gate_name: Gate name to filter events (only events within the given gate will be plotted)
        :param gate_path: tuple of gate names for full set of gate ancestors.
            Required if gate_name is ambiguous
        :param subsample_count: Number of events to use as a subsample. If the number of
            events in the Sample is less than the requested subsample count, then the
            maximum number of available events is used for the subsample.
        :param random_seed: Random seed used for subsampling events
        :param color_density: Whether to color the events by density, similar
            to a heat map. Default is True.
        :param bin_width: Bin size to use for the color density, in units of
            event point size. Larger values produce smoother gradients.
            Default is 4 for a 4x4 grid size.
        :param x_min: Lower bound of x-axis. If None, channel's min value will
            be used with some padding to keep events off the edge of the plot.
        :param x_max: Upper bound of x-axis. If None, channel's max value will
            be used with some padding to keep events off the edge of the plot.
        :param y_min: Lower bound of y-axis. If None, channel's min value will
            be used with some padding to keep events off the edge of the plot.
        :param y_max: Upper bound of y-axis. If None, channel's max value will
            be used with some padding to keep events off the edge of the plot.
        :param height: Height of plot in pixels. Default is 600.
        :param width: Width of plot in pixels. Default is 600.
        :return: A Bokeh Figure object containing the interactive scatter plot.
        """
        # Get Sample instance and apply requested subsampling
        sample = self.get_sample(sample_id)
        sample.subsample_events(subsample_count=subsample_count, random_seed=random_seed)

        # Factoring out this plot_scatter to common code shared by
        # the Sample & Workspace class is tricky because GatingML
        # technically allows mixed spill matrices for 2 channels
        # in the same sample. Further, a simple difference of comp
        # reference strings doesn't mean there are mixed matrices,
        # as a scatter channel can have the value 'uncompensated'
        # and a fluoro channel 'some_spill_id'.
        x_comp_ref = x_dim.compensation_ref
        y_comp_ref = y_dim.compensation_ref

        x_xform_ref = x_dim.transformation_ref
        y_xform_ref = y_dim.transformation_ref

        x_index = sample.get_channel_index(x_dim.id)
        y_index = sample.get_channel_index(y_dim.id)

        if x_comp_ref is not None and x_comp_ref != 'uncompensated':
            x_comp = self.gating_strategy.get_comp_matrix(x_dim.compensation_ref)
            comp_events = x_comp.apply(sample)
            x = comp_events[:, x_index]
        else:
            # not doing subsample here, will do later with bool AND
            # get channel events using the label
            x = sample.get_channel_events(x_dim.id, source='raw', subsample=False)

        if y_comp_ref is not None and y_comp_ref != 'uncompensated':
            # this is likely unnecessary as the x & y comp should be the same
            # for fluoro channels, but requires more conditionals to cover
            y_comp = self.gating_strategy.get_comp_matrix(y_dim.compensation_ref)
            comp_events = y_comp.apply(sample)
            y = comp_events[:, y_index]
        else:
            # not doing subsample here, will do later with bool AND
            y = sample.get_channel_events(y_dim.id, source='raw', subsample=False)

        if x_xform_ref is not None:
            x_xform = self.gating_strategy.get_transform(x_xform_ref)
            x = x_xform.apply(x)
        if y_xform_ref is not None:
            y_xform = self.gating_strategy.get_transform(y_xform_ref)
            y = y_xform.apply(y)

        if gate_name is not None:
            gate_results = self.get_gating_results(sample_id=sample_id)
            is_gate_event = gate_results.get_gate_membership(gate_name, gate_path)
        else:
            is_gate_event = np.ones(sample.event_count, dtype=bool)

        is_subsample = np.zeros(sample.event_count, dtype=bool)
        is_subsample[sample.subsample_indices] = True

        idx_to_plot = np.logical_and(is_gate_event, is_subsample)

        # check if there are any events to plot
        if idx_to_plot.sum() == 0:
            raise FlowKitException("There are no events to plot for the specified options")

        x = x[idx_to_plot]
        y = y[idx_to_plot]

        if sample.pns_labels[x_index] != '':
            x_label = '%s (%s)' % (sample.pns_labels[x_index], sample.pnn_labels[x_index])
        else:
            x_label = sample.pnn_labels[x_index]

        if sample.pns_labels[y_index] != '':
            y_label = '%s (%s)' % (sample.pns_labels[y_index], sample.pnn_labels[y_index])
        else:
            y_label = sample.pnn_labels[y_index]

        p = plot_utils.plot_scatter(
            x,
            y,
            x_label=x_label,
            y_label=y_label,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            color_density=color_density,
            bin_width=bin_width,
            height=height,
            width=width
        )

        p.title = Title(text=sample.id, align='center')

        return p
