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
from .._models import gates, dimension
from .._utils import plot_utils, xml_utils, wsp_utils, sample_utils, gating_utils
from ..exceptions import GateReferenceError
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

    def remove_gate(self, gate_name, gate_path=None, keep_children=False):
        """
        Remove a gate from the gate tree. Any descendant gates will also be removed
        unless keep_children=True. In all cases, if a BooleanGate exists that references
        the gate to remove, a GateTreeError will be thrown indicating the BooleanGate
        must be removed prior to removing the gate.

        :param gate_name: text string of a gate name
        :param gate_path: complete tuple of gate IDs for unique set of gate ancestors.
            Required if gate_name is ambiguous
        :param keep_children: Whether to keep child gates. If True, the child gates will be
            remapped to the removed gate's parent. Default is False, which will delete all
            descendant gates.
        :return: None
        """
        self.gating_strategy.remove_gate(gate_name, gate_path=gate_path, keep_children=keep_children)

    def add_transform(self, transform):
        """
        Add a Transform instance to use in the gating strategy.

        :param transform: an instance of a Transform subclass
        :return: None
        """
        self.gating_strategy.add_transform(copy.deepcopy(transform))

    def get_transforms(self):
        """
        Retrieve the list of Transform instances stored in the gating strategy.

        :return: list of Transform instances
        """

        return list(self.gating_strategy.transformations.values())

    def get_transform(self, transform_id):
        """
        Retrieve a Transform stored in the gating strategy by its ID.

        :param transform_id: a text string representing a Transform ID
        :return: an instance of a Transform subclass
        """
        return self.gating_strategy.get_transform(transform_id)

    def add_comp_matrix(self, matrix):
        """
        Add a Matrix instance to use in the gating strategy.

        :param matrix: an instance of the Matrix class
        :return: None
        """
        self.gating_strategy.add_comp_matrix(copy.deepcopy(matrix))

    def get_comp_matrices(self):
        """
        Retrieve the list of compensation Matrix instances stored in the gating strategy.

        :return: list of Matrix instances
        """
        return list(self.gating_strategy.comp_matrices.values())

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
        xml_utils.export_gatingml(self.gating_strategy, file_handle, sample_id=sample_id)

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

        :param sample_id: a text string representing the sample
        :return: Sample instance
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
            all_reports.append(result.report)

        return copy.deepcopy(pd.concat(all_reports))

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
            bin_width=4
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
        :param subsample_count: Number of events to use as a sub-sample. If the number of
            events in the Sample is less than the requested sub-sample count, then the
            maximum number of available events is used for the sub-sample.
        :param random_seed: Random seed used for sub-sampling events
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

        gate = self.gating_strategy.get_gate(gate_name, gate_path=gate_path, sample_id=sample_id)

        # check for a boolean gate, there's no reasonable way to plot these
        if isinstance(gate, gates.BooleanGate):
            raise TypeError("Plotting Boolean gates is not allowed (gate %s)" % gate.gate_name)

        parent_gate_name = gate_path[-1]
        parent_gate_path = gate_path[:-1]

        dim_ids_ordered = []
        dim_is_ratio = []
        dim_comp_refs = []
        dim_min = []
        dim_max = []
        for i, dim in enumerate(gate.dimensions):
            if isinstance(dim, dimension.RatioDimension):
                dim_ids_ordered.append(dim.ratio_ref)
                tmp_dim_min = dim.min
                tmp_dim_max = dim.max
                is_ratio = True
            elif isinstance(dim, dimension.QuadrantDivider):
                dim_ids_ordered.append(dim.dimension_ref)
                tmp_dim_min = None
                tmp_dim_max = None
                is_ratio = False
            else:
                dim_ids_ordered.append(dim.id)
                tmp_dim_min = dim.min
                tmp_dim_max = dim.max
                is_ratio = False

            dim_min.append(tmp_dim_min)
            dim_max.append(tmp_dim_max)
            dim_is_ratio.append(is_ratio)
            dim_comp_refs.append(dim.compensation_ref)

        # dim count determines if we need a histogram, scatter, or multi-scatter
        dim_count = len(dim_ids_ordered)
        if dim_count == 1:
            gate_type = 'hist'
        elif dim_count == 2:
            gate_type = 'scatter'
        elif dim_count > 2:
            raise NotImplementedError("Plotting of gates with >2 dimensions is not supported")
        else:
            # there are no dimensions
            raise ValueError("Gate %s appears to not reference any dimensions" % gate_name)

        # Get Sample instance and apply requested subsampling
        sample_to_plot = self.get_sample(sample_id)
        sample_to_plot.subsample_events(subsample_count=subsample_count, random_seed=random_seed)
        # noinspection PyProtectedMember
        events = self.gating_strategy._preprocess_sample_events(
            sample_to_plot,
            gate
        )

        # get parent gate results to display only those events
        if parent_gate_name != 'root':
            # TODO:  make it clear to call analyze_samples prior to calling this method
            is_parent_event = self.get_gate_membership(sample_id, parent_gate_name, parent_gate_path)
            is_subsample = np.zeros(sample_to_plot.event_count, dtype=bool)
            is_subsample[sample_to_plot.subsample_indices] = True
            idx_to_plot = np.logical_and(is_parent_event, is_subsample)
        else:
            idx_to_plot = sample_to_plot.subsample_indices

        x = events.loc[idx_to_plot, dim_ids_ordered[0]].values

        dim_ids = []

        if dim_is_ratio[0]:
            dim_ids.append(dim_ids_ordered[0])
            x_pnn_label = None
        else:
            try:
                x_index = sample_to_plot.get_channel_index(dim_ids_ordered[0])
            except ValueError:
                # might be a label reference in the comp matrix
                matrix = self.gating_strategy.get_comp_matrix(dim_comp_refs[0])
                try:
                    matrix_dim_idx = matrix.fluorochomes.index(dim_ids_ordered[0])
                except ValueError:
                    raise ValueError("%s not found in list of matrix fluorochromes" % dim_ids_ordered[0])
                detector = matrix.detectors[matrix_dim_idx]
                x_index = sample_to_plot.get_channel_index(detector)

            x_pnn_label = sample_to_plot.pnn_labels[x_index]

            if sample_to_plot.pns_labels[x_index] != '':
                dim_ids.append('%s (%s)' % (sample_to_plot.pns_labels[x_index], x_pnn_label))
            else:
                dim_ids.append(sample_to_plot.pnn_labels[x_index])

        y_pnn_label = None

        if dim_count > 1:
            if dim_is_ratio[1]:
                dim_ids.append(dim_ids_ordered[1])

            else:
                try:
                    y_index = sample_to_plot.get_channel_index(dim_ids_ordered[1])
                except ValueError:
                    # might be a label reference in the comp matrix
                    matrix = self.gating_strategy.get_comp_matrix(dim_comp_refs[1])
                    try:
                        matrix_dim_idx = matrix.fluorochomes.index(dim_ids_ordered[1])
                    except ValueError:
                        raise ValueError("%s not found in list of matrix fluorochromes" % dim_ids_ordered[1])
                    detector = matrix.detectors[matrix_dim_idx]
                    y_index = sample_to_plot.get_channel_index(detector)

                y_pnn_label = sample_to_plot.pnn_labels[y_index]

                if sample_to_plot.pns_labels[y_index] != '':
                    dim_ids.append('%s (%s)' % (sample_to_plot.pns_labels[y_index], y_pnn_label))
                else:
                    dim_ids.append(sample_to_plot.pnn_labels[y_index])

        if gate_type == 'scatter':
            y = events.loc[idx_to_plot, dim_ids_ordered[1]].values

            p = plot_utils.plot_scatter(
                x,
                y,
                dim_ids,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                color_density=color_density,
                bin_width=bin_width
            )
        elif gate_type == 'hist':
            p = plot_utils.plot_histogram(x, dim_ids[0])
        else:
            raise NotImplementedError("Only histograms and scatter plots are supported in this version of FlowKit")

        if isinstance(gate, gates.PolygonGate):
            source, glyph = plot_utils.render_polygon(gate.vertices)
            p.add_glyph(source, glyph)
        elif isinstance(gate, gates.EllipsoidGate):
            ellipse = plot_utils.render_ellipse(
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
            full_gate_path = full_gate_path + (gate_name,)
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
                Title(text=gate_name, text_font_style="italic", text_font_size="1em", align='center'),
                'above'
            )

        plot_title = "%s" % sample_id
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
            gate_name=None,
            subsample_count=10000,
            random_seed=1,
            color_density=True,
            bin_width=4,
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
        :param gate_name: Gate name to filter events (only events within the given gate will be plotted)
        :param subsample_count: Number of events to use as a sub-sample. If the number of
            events in the Sample is less than the requested sub-sample count, then the
            maximum number of available events is used for the sub-sample.
        :param random_seed: Random seed used for sub-sampling events
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
        :return: A Bokeh Figure object containing the interactive scatter plot.
        """
        # Get Sample instance and apply requested subsampling
        sample = self.get_sample(sample_id)
        sample.subsample_events(subsample_count=subsample_count, random_seed=random_seed)

        x_index = sample.get_channel_index(x_dim.id)
        y_index = sample.get_channel_index(y_dim.id)

        x_comp_ref = x_dim.compensation_ref
        x_xform_ref = x_dim.transformation_ref

        y_comp_ref = y_dim.compensation_ref
        y_xform_ref = y_dim.transformation_ref

        if x_comp_ref is not None and x_comp_ref != 'uncompensated':
            x_comp = self.gating_strategy.get_comp_matrix(x_dim.compensation_ref)
            comp_events = x_comp.apply(sample)
            x = comp_events[:, x_index]
        else:
            # not doing sub-sample here, will do later with bool AND
            x = sample.get_channel_events(x_index, source='raw', subsample=False)

        if y_comp_ref is not None and x_comp_ref != 'uncompensated':
            # this is likely unnecessary as the x & y comp should be the same,
            # but requires more conditionals to cover
            y_comp = self.gating_strategy.get_comp_matrix(x_dim.compensation_ref)
            comp_events = y_comp.apply(sample)
            y = comp_events[:, y_index]
        else:
            # not doing sub-sample here, will do later with bool AND
            y = sample.get_channel_events(y_index, source='raw', subsample=False)

        if x_xform_ref is not None:
            x_xform = self.gating_strategy.get_transform(x_xform_ref)
            x = x_xform.apply(x.reshape(-1, 1))[:, 0]
        if y_xform_ref is not None:
            y_xform = self.gating_strategy.get_transform(y_xform_ref)
            y = y_xform.apply(y.reshape(-1, 1))[:, 0]

        if gate_name is not None:
            gate_results = self.get_gating_results(sample_id=sample_id)
            is_gate_event = gate_results.get_gate_membership(gate_name)
        else:
            is_gate_event = np.ones(sample.event_count, dtype=bool)

        is_subsample = np.zeros(sample.event_count, dtype=bool)
        is_subsample[sample.subsample_indices] = True

        idx_to_plot = np.logical_and(is_gate_event, is_subsample)
        x = x[idx_to_plot]
        y = y[idx_to_plot]

        dim_ids = []

        if sample.pns_labels[x_index] != '':
            dim_ids.append('%s (%s)' % (sample.pns_labels[x_index], sample.pnn_labels[x_index]))
        else:
            dim_ids.append(sample.pnn_labels[x_index])

        if sample.pns_labels[y_index] != '':
            dim_ids.append('%s (%s)' % (sample.pns_labels[y_index], sample.pnn_labels[y_index]))
        else:
            dim_ids.append(sample.pnn_labels[y_index])

        p = plot_utils.plot_scatter(
            x,
            y,
            dim_ids,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            color_density=color_density,
            bin_width=bin_width
        )

        p.title = Title(text=sample.id, align='center')

        return p
