"""
Workspace class
"""
import gc
import copy
import os
import numpy as np
import pandas as pd
from bokeh.models import Title
from .._conf import debug
from .._utils import plot_utils, wsp_utils, sample_utils, gating_utils
from ..exceptions import FlowKitException, GateReferenceError
import warnings


class Workspace(object):
    """
    A Workspace represents an imported FlowJo workspace (.wsp file).

    :param wsp_file_path: FlowJo WSP file as a file name/path, file object, or file-like object
    :param fcs_samples: str or list. If given a string, it can be a directory path or a file path.
        If a directory, any .fcs files in the directory will be found. If a list, then it must
        be a list of file paths or a list of Sample instances. Lists of mixed types are not
        supported. Note that only FCS files matching the ones referenced in the .wsp file will
        be retained in the Workspace.
    :param load_missing_file_data: Controls behavior for missing FCS files. If True, gate data for
        missing FCS files (i.e. not in fcs_samples arg) will still be loaded. If False, warnings
        are issued for FCS files found in the WSP file that were not loaded in the Workspace and
        gate data for these missing files will not be retained. Default is False.
    :param find_fcs_files_from_wsp: Controls whether to search for FCS files based on `URI` params
        within the FlowJo workspace file.
    :param filename_as_id: Boolean option for using the file name (as it exists on the filesystem)
        as Sample ID, default is False. Only applies to file paths given to the 'fcs_samples'
        argument.
    """
    def __init__(
            self,
            wsp_file_path,
            fcs_samples=None,
            load_missing_file_data=False,
            find_fcs_files_from_wsp=False,
            filename_as_id=False
    ):
        # The sample LUT holds sample IDs (keys) only for loaded samples.
        # The values are the Sample instances
        self._sample_lut = {}

        # The sample data LUT holds sample IDs (keys) for all samples
        # that have WSP gating data whether the sample is loaded or not.
        # The value is a dict containing the following items:
        #   - 'keywords': dict of FCS metadata found in WSP file
        #   - 'compensation' dict of spill info found in WSP file
        #   - 'transforms': dict of channel transforms found in WSP file
        #   - 'custom_gate_ids': set of gate paths for custom gates
        #   - 'gating_strategy': GatingStrategy assembled from WSP file
        self._sample_data_lut = {}

        # The group LUT keys are the available sample group names.
        # The values are dicts with keys 'gates' & 'samples'.
        # 'gates' has the common group Gate instances (not a
        # full GatingStrategy though) and 'samples' is a list
        # of the sample IDs belonging to the group.
        self._group_lut = {}

        # For storing GatingResults for analyzed samples.
        # Keys are sample IDs. We could put these in the
        # sample_lut but sample_lut is meant to be created
        # and complete within this constructor. Plus, this
        # makes it easier to determine which samples have
        # been analyzed.
        self._results_lut = {}
        
        # load samples we were given, we'll cross-reference against wsp below
        tmp_sample_lut = {
            s.id: s for s in sample_utils.load_samples(
                fcs_samples, filename_as_id=filename_as_id, use_flowjo_labels=True
            )
        }
        self._sample_lut = {}

        wsp_data = wsp_utils.parse_wsp(wsp_file_path)

        # find samples in wsp file. in wsp_data['samples'], each item is a dict which has a key `sample_uri`
        if find_fcs_files_from_wsp:
            if fcs_samples is not None:
                warnings.warn("When `find_fcs_files_from_wsp` is True, `fcs_samples` will be ignored.")

            tmp_sample_lut = {}
    
            for sample_name in wsp_data['samples']:
                sample_data = wsp_data['samples'][sample_name]
                sample_uri = sample_data['sample_uri']

                # Convert the URI to a path
                # noinspection PyProtectedMember
                path = wsp_utils._uri_to_path(sample_uri, wsp_file_path)

                # Test whether file exists at path and if not present
                # warn user with message indicating the path.
                if not os.path.exists(path):
                    warnings.warn("Sample file not found at path: {}".format(path))
                    continue

                # Read in the sample file
                sample_filedata = sample_utils.load_samples(path, filename_as_id=filename_as_id)[0]

                # Update the ID of the loaded data (otherwise analysis breaks)
                sample_filedata.id = sample_name

                tmp_sample_lut[sample_name] = sample_filedata

        # save group sample membership, we'll filter by loaded samples next
        group_lut = wsp_data['groups']

        # save sample data, including the GatingStrategy & Sample instance
        for sample_id, sample_dict in wsp_data['samples'].items():
            if sample_id in tmp_sample_lut:
                # retain sample and add to sample data
                self._sample_lut[sample_id] = tmp_sample_lut[sample_id]
                self._sample_data_lut[sample_id] = sample_dict
            else:
                # we have gating info for a sample that wasn't loaded
                if load_missing_file_data:
                    # We were instructed to load data for missing files, so
                    # we'll still save the gate info for retrieval purposes
                    self._sample_data_lut[sample_id] = sample_dict
                else:
                    # we won't load data for missing files, issue a warning
                    # and remove any references to the sample
                    msg = "WSP references %s, but sample was not loaded." % sample_id
                    warnings.warn(msg)

                    # search for this missing sample ID in group data & remove
                    for group_name, group_dict in group_lut.items():
                        if sample_id in group_dict['samples']:
                            group_dict['samples'].remove(sample_id)

        self._group_lut = group_lut

    def __repr__(self):
        sample_count = len(self._sample_lut)
        sample_group_count = len(self._group_lut)

        return (
            f'{self.__class__.__name__}('
            f'{sample_count} samples loaded, '
            f'{sample_group_count} sample groups)'
        )

    def summary(self):
        """
        Retrieve a summary of Workspace information, including a list of
        sample groups defined, along with the sample and gate counts
        for those sample groups.

        :return: Pandas DataFrame containing Workspace summary information
        """
        sg_list = []

        for group_name, group_dict in self._group_lut.items():
            group_sample_ids = group_dict['samples']
            sample_count = len(group_sample_ids)

            loaded_sample_count = 0
            for g_sample_id in group_sample_ids:
                if g_sample_id in self._sample_lut:
                    loaded_sample_count += 1

            if sample_count > 0:
                # There's at least one sample, so grab the first one
                # as the prototype for the gating strategy. A WSP
                # group isn't guaranteed to have template gates for
                # the complete gate tree, this is the best we can do.
                example_sample_id = group_dict['samples'][0]
                gs = self.get_gating_strategy(example_sample_id)

                gate_count = len(gs.get_gate_ids())
                gate_depth = gs.get_max_depth()
            else:
                gate_count = 0
                gate_depth = 0

            sg_info = {
                'group_name': group_name,
                'samples': sample_count,
                'loaded_samples': loaded_sample_count,
                'gates': gate_count,
                'max_gate_depth': gate_depth
            }

            sg_list.append(sg_info)

        df = pd.DataFrame(sg_list)
        df.set_index('group_name', inplace=True)

        return df

    def get_sample_ids(self, group_name=None, loaded_only=True):
        """
        Retrieve the list of Sample IDs that in the Workspace, optionally
        filtered by sample group and/or loaded status. Default is all loaded
        samples.

        :param group_name: Filter returned sample IDs by a sample group. If None, all sample IDs are returned
        :param loaded_only: Filter returned sample IDs for only loaded samples. If False, all the samples will
            be returned, including any missing sample IDs referenced in the workspace. Default is True for
            returning only loaded sample IDs.
        :return: list of Sample ID strings
        """
        if group_name is not None:
            # group LUT contains all group sample IDs incl. missing ones
            sample_ids = set(self._group_lut[group_name]['samples'])
        else:
            # No group name specified so give user all sample IDs
            # sample data LUT contains all sample IDs, incl. missing IDs
            # referenced in the wsp (if load_missing_file_data was True)
            sample_ids = set(self._sample_data_lut.keys())

        # check if only loaded samples were requested
        if loaded_only:
            # sample LUT contains all the loaded sample IDs
            loaded_sample_ids = set(self._sample_lut.keys())

            # cross-reference sample_ids with loaded_sample_ids
            sample_ids = sample_ids.intersection(loaded_sample_ids)

        return sorted(list(sample_ids))

    def get_sample(self, sample_id):
        """
        Retrieve a Sample instance from the Workspace.

        :param sample_id: a text string representing the sample
        :return: a Sample instance
        """
        return self._sample_lut[sample_id]

    def get_samples(self, group_name=None):
        """
        Retrieve list of Sample instances, optionally filtered by sample group.

        :param group_name: Filter returned samples by a sample group. If None, all samples are returned
        :return: list of Sample instances
        """
        # don't return samples that haven't been loaded
        sample_ids = self.get_sample_ids(group_name=group_name)

        samples = []
        for s_id in sample_ids:
            samples.append(self._sample_lut[s_id])

        return samples

    def get_keywords(self, sample_id):
        """
        Retrieve a dictionary metadata keywords stored in the workspace
        for a specific sample. Generally, these will be the same as the
        metadata attribute in the Sample instance but can contain additional
        keywords set by the user within FlowJo. Note the keys here will
        have the '$' character prefix for FCS-defined keywords.

        :param sample_id: a text string representing a Sample instance
        :return: a dictionary of metadata stored in the FlowJo workspace
        """
        sample_dict = self._sample_data_lut[sample_id]

        if sample_dict['keywords'] is not None:
            keywords = copy.deepcopy(sample_dict['keywords'])
        else:
            keywords = None

        return keywords

    def get_sample_groups(self):
        """
        Retrieve the list of sample group names defined in the Workspace.

        :return: list of sample group ID strings
        """
        return list(self._group_lut.keys())

    def get_gate_ids(self, sample_id):
        """
        Retrieve the list of gate IDs defined for the specified sample.
        The gate ID is a 2-item tuple where the first item is a string
        representing the gate name and the second item is a tuple of
        the gate path.

        :param sample_id: a text string representing a Sample instance
        :return: list of gate ID tuples
        """
        gs = self._sample_data_lut[sample_id]['gating_strategy']
        return gs.get_gate_ids()

    def find_matching_gate_paths(self, sample_id, gate_name):
        """
        Find all gate paths in the gating strategy for the given Sample
        matching the given gate name.

        :param sample_id: a text string representing a Sample instance
        :param gate_name: text string of a gate name
        :return: list of gate paths (list of tuples)
        """
        gs = self._sample_data_lut[sample_id]['gating_strategy']
        return gs.find_matching_gate_paths(gate_name)

    def get_child_gate_ids(self, sample_id, gate_name, gate_path=None):
        """
        Retrieve list of child gate IDs for a sample given the parent
        gate name (and path if ambiguous) in the gating strategy.

        :param sample_id: a text string representing a Sample instance
        :param gate_name: text string of a gate name
        :param gate_path: complete tuple of gate IDs for unique set of gate ancestors.
            Required if gate.gate_name is ambiguous
        :return: list of Gate IDs (tuple of gate name plus gate path). Returns an empty
            list if no child gates exist.
        :raises GateReferenceError: if gate ID is not found in gating strategy or if gate
            name is ambiguous
        """
        gs = self._sample_data_lut[sample_id]['gating_strategy']
        child_gate_ids = gs.get_child_gate_ids(gate_name, gate_path)

        return child_gate_ids

    def get_gate_hierarchy(self, sample_id, output='ascii', **kwargs):
        """
        Retrieve the hierarchy of gates in the sample's gating strategy. Output is available
        in several formats, including text, dictionary, or JSON. If output == 'json', extra
        keyword arguments are passed to json.dumps

        :param sample_id: a text string representing a Sample instance
        :param output: Determines format of hierarchy returned, either 'ascii',
            'dict', or 'JSON' (default is 'ascii')
        :return: gate hierarchy as a text string or a dictionary
        """
        gs = self._sample_data_lut[sample_id]['gating_strategy']
        return gs.get_gate_hierarchy(output, **kwargs)

    def get_gating_strategy(self, sample_id):
        """
        Retrieve a copy of the GatingStrategy for a specific sample. sample_id is required as
        each sample may have customized gates

        :param sample_id: a text string representing a Sample instance
        :return: a copy of the GatingStrategy instance
        """
        return copy.deepcopy(self._sample_data_lut[sample_id]['gating_strategy'])

    def get_comp_matrix(self, sample_id):
        """
        Retrieve the compensation matrix for a specific sample.

        :param sample_id: a text string representing a Sample instance
        :return: a copy of a Matrix instance
        """
        sample_dict = self._sample_data_lut[sample_id]

        if sample_dict['compensation'] is not None:
            comp_matrix = copy.deepcopy(sample_dict['compensation']['matrix'])
        else:
            comp_matrix = None

        return comp_matrix

    def get_transform(self, sample_id, transform_id):
        """
        Retrieve a single transform for a sample using the transform ID. Transform
        IDs in the Workspace class correspond to a channel label in the sample.

        :param sample_id: a text string representing a Sample instance
        :param transform_id: a text string representing a Transform instance
        :return:
        """
        sample_dict = self._sample_data_lut[sample_id]

        if sample_dict['transforms'] is not None:
            xform = copy.deepcopy(sample_dict['transforms'][transform_id])
        else:
            xform = None

        return xform

    def get_transforms(self, sample_id):
        """
        Retrieve a dictionary LUT of transformations for a specific sample.
        Keys are the transform IDs and values are Transform instances.

        :param sample_id: a text string representing a Sample instance
        :return: a dictionary LUT of transform IDs: Transform instances
        """
        sample_dict = self._sample_data_lut[sample_id]

        if sample_dict['transforms'] is not None:
            xform_lut = copy.deepcopy(sample_dict['transforms'])
        else:
            xform_lut = None

        return xform_lut

    def get_gate(self, sample_id, gate_name, gate_path=None):
        """
        Retrieve a gate instance for a sample by its gate ID.

        :param sample_id: a text string representing a Sample instance.
        :param gate_name: text string of a gate ID
        :param gate_path: tuple of gate IDs for unique set of gate ancestors. Required if gate_name is ambiguous
        :return: Subclass of a Gate object
        """
        gs = self._sample_data_lut[sample_id]['gating_strategy']
        return gs.get_gate(gate_name, gate_path=gate_path)

    def analyze_samples(self, group_name=None, sample_id=None, cache_events=False, use_mp=True, verbose=False):
        """
        Process gates for samples. Samples to analyze can be filtered by group name or sample ID.
        After running, results can be retrieved using the `get_gating_results`, `get_group_report`,
        and  `get_gate_membership`, methods.

        :param group_name: optional group name, if specified only samples in this group will be processed
        :param sample_id: optional sample ID, if specified only this sample will be processed (overrides group filter)
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
        if sample_id is not None:
            sample_ids = [sample_id]
        else:
            # If group name is specified, get_sample_ids will return the
            # group sample IDs. If not then it will return all sample IDs.
            sample_ids = self.get_sample_ids(group_name=group_name)

        if len(sample_ids) == 0:
            warnings.warn("No samples were found to analyze")
            return

        sample_data_to_run = []
        for s_id in sample_ids:
            if s_id not in self._sample_data_lut:
                # sample ID provided isn't present in Workspace
                # or was referenced but has no gate data.
                warnings.warn("Sample %s has no gate data" % s_id)
                continue

            sample = self._sample_lut[s_id]
            gating_strategy = self._sample_data_lut[s_id]['gating_strategy']

            sample_data_to_run.append(
                {
                    'gating_strategy': gating_strategy,
                    'sample': sample
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

        # save the results in results LUT
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
                "No results found for %s. Have you run `analyze_samples`?" % sample_id
            )
        return copy.deepcopy(gating_result)

    def get_analysis_report(self, group_name=None):
        """
        Retrieve the report for the analyzed samples as a pandas DataFrame.

        :param group_name: optional group name, if specified only results
            from samples in this group will be processed, otherwise results
            from all analyzed samples will be returned
        :return: pandas DataFrame
        """
        all_reports = []
        group_s_ids = self.get_sample_ids(group_name)

        for s_id in group_s_ids:
            try:
                result = self._results_lut[s_id]
            except KeyError:
                continue

            # avoid Pandas warning about concatenating empty DataFrame instances
            if len(result.report) == 0:
                continue

            all_reports.append(result.report)

        # Explicitly setting copy=True even though the default in case
        # it ever changes. Used to do our own deep copy but Pandas was
        # already doing this, so it was getting deep copied twice.
        return pd.concat(all_reports, ignore_index=True, copy=True)

    def _get_processed_events(self, sample_id, source=None):
        """
        Retrieve a pandas DataFrame containing processed events for specified sample.
        Compensation and transforms will be applied according to the WSP file.

        :param sample_id: a text string representing a Sample instance
        :param source: Determines how events are processed. Values include 'raw', 'comp',
            'xform', and None (default). If None, events are processed according to the
            workspace compensation and/or transforms. If 'raw', no processing is applied.
            If 'comp', only compensation is applied. If 'xform', then the available
            transformations will be applied (post compensation, if comp matrix exists).
        :return: pandas DataFrame containing the processed sample events
        """
        # get sample and accompanying comp & transform
        sample = self.get_sample(sample_id)
        comp_matrix = self.get_comp_matrix(sample_id)
        xform_lut = self.get_transforms(sample_id)
        event_source = 'raw'

        # compensate if source is None, comp, or xform AND a comp matrix exists
        if comp_matrix is not None and source in [None, 'comp', 'xform']:
            sample.apply_compensation(comp_matrix)
            event_source = 'comp'

        # transform if source is None or xform AND xforms exist
        if xform_lut is not None and source in [None, 'xform']:
            sample.apply_transform(xform_lut)
            event_source = 'xform'

        events_df = sample.as_dataframe(source=event_source)

        return events_df

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
        gating_result = self.get_gating_results(sample_id)
        return gating_result.get_gate_membership(gate_name, gate_path=gate_path)

    def get_gate_events(self, sample_id, gate_name=None, gate_path=None, source=None):
        """
        Retrieve gated events for a specific gate & sample as a pandas DataFrame.
        Gated events are processed according to the sample's compensation &
        channel transforms.

        :param sample_id: a text string representing a Sample instance
        :param gate_name: text string of a gate ID. If None, all Sample events will be returned (i.e. un-gated)
        :param gate_path: complete tuple of gate IDs for unique set of gate ancestors.
            Required if gate_name is ambiguous
        :param source: Determines how events are processed. Values include 'raw', 'comp',
            'xform', and None (default). If None, events are processed according to the
            workspace compensation and/or transforms. If 'raw', no processing is applied.
            If 'comp', only compensation is applied. If 'xform', then the available
            transformations will be applied (post compensation, if comp matrix exists).
        :return: a pandas DataFrames with the gated events, compensated & transformed according
            to the group's compensation matrix and transforms
        """
        df_events = self._get_processed_events(sample_id, source=source)

        if gate_name is not None:
            gate_idx = self.get_gate_membership(sample_id, gate_name, gate_path)
            df_events = df_events[gate_idx]

        # TODO: maybe make an optional kwarg to control column label format
        df_events.columns = [' '.join(col).strip() for col in df_events.columns]

        # TODO: make inclusion of sample_id column a kwarg as well
        df_events.insert(0, 'sample_id', sample_id)

        return df_events

    def archive_results(self, group_name, out_dir, output_prefix=None, df_sample_metadata=None, overwrite=False):
        """
        Archive group results to PyArrow Feather files for language-agnostic downstream analysis.
        Requires having previously run the `analyze_samples` method. The following archive
        files will be created:
            - Sample panel (CSV file)
            - Gate tree (text file)
            - Gate ID lookup table (Feather file)
            - Gating report (Feather file)
            - Gate membership (Feather file)
            - All preprocessed events (Feather file)

        :param group_name: Workspace group name to archive
        :param out_dir: string path for the output file directory
        :param output_prefix: Optional string prefix for the output file names
        :param df_sample_metadata: Optional pandas DataFrame of Sample metadata. Must have a
            'sample_id' column as an index, with distinct rows for each Sample ID. Sample metadata
            will be merged with the archived gating report, gate membership, and preprocessed
            events feather files.
        :param overwrite: force overwriting existing files (default=False)
        :return: None
        """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # set file write mode
        if overwrite:
            file_write_mode = 'w'
        else:
            file_write_mode = 'x'

        # parse output_prefix arg
        if output_prefix is not None:
            # add an underscore for separation with remaining file names
            output_prefix = output_prefix + '_'
        else:
            # no prefix, set to empty string
            output_prefix = ''

        # Check if any output files exist (unless overwrite=True)
        panel_file_path = os.path.join(out_dir, output_prefix + 'panel.csv')
        gate_tree_file_path = os.path.join(out_dir, output_prefix + 'gate_tree.txt')
        gate_lut_file_path = os.path.join(out_dir, output_prefix + 'gate_id_lut.feather')
        report_file_path = os.path.join(out_dir, output_prefix + 'gating_report.feather')
        events_file_path = os.path.join(out_dir, output_prefix + 'all_events.feather')
        member_file_path = os.path.join(out_dir, output_prefix + 'gate_membership.feather')
        if not overwrite:
            for fp in [panel_file_path, gate_lut_file_path, gate_lut_file_path, report_file_path, events_file_path, member_file_path]:
                if os.path.exists(fp):
                    raise FileExistsError(
                        "%s already exists, set `overwrite=True` to overwrite file" % os.path.basename(fp)
                    )

        # Get Sample IDs, normally they correspond to the FCS file names.
        sample_ids = self.get_sample_ids(group_name=group_name)

        # grab first sample to extract panel info (channels)
        proto_sample = self.get_sample(sample_ids[0])

        # get gate hierarchy from first sample
        gate_tree = self.get_gate_hierarchy(sample_id=sample_ids[0])

        # get gate IDs from first sample & save gate IDs as LUT with
        # full ID string joined with "/" and the gate name and gate path tuple
        gate_ids = self.get_gate_ids(sample_id=sample_ids[0])
        gate_lut_list = []  # list of dictionaries
        for gate_name, gate_path in gate_ids:
            gate_id_str = '/'.join(gate_path + (gate_name,))
            gate_lut_list.append(
                {
                    'gate_id': gate_id_str,
                    'gate_path': gate_path,
                    'gate_name': gate_name
                }
            )
        df_gate_lut = pd.DataFrame(gate_lut_list)

        # gate report
        report = self.get_analysis_report(group_name=group_name)

        # Export processed sample events
        df_s_events_list = []
        for s_id in sample_ids:
            # extract full array of pre-processed events for sample
            df_s_preproc_events = self.get_gate_events(sample_id=s_id)
            df_s_events_list.append(df_s_preproc_events)

        df_events_all = pd.concat(df_s_events_list, ignore_index=True)

        # Now extract the gate membership Boolean arrays for each gate in each sample
        gate_membership_all_samples = []

        for s_id in sample_ids:
            gate_membership_dict = {}

            for gate_name, gate_path in gate_ids:
                gate_id_str = '/'.join(gate_path + (gate_name,))

                gate_membership = self.get_gate_membership(s_id, gate_name, gate_path=gate_path)
                gate_membership_dict[gate_id_str] = gate_membership

            df_gate_membership = pd.DataFrame(gate_membership_dict)
            df_gate_membership.insert(0, 'sample_id', s_id)
            # df_gate_membership.insert(1, 'subject_id', subject_id)
            gate_membership_all_samples.append(df_gate_membership)

        df_gate_membership_all_samples = pd.concat(gate_membership_all_samples, ignore_index=True)

        # merge sample metadata if given
        if df_sample_metadata is not None:
            # Prior to merging, build the new column list so the metadata columns are first
            # but after the 'sample' column
            first_col = 'sample_id'
            meta_cols = df_sample_metadata.columns.tolist()

            remaining_report_cols = report.columns.tolist()
            remaining_report_cols.remove(first_col)
            new_report_col_order = [first_col] + meta_cols + remaining_report_cols

            # Now merge and rearrange the column order
            report = pd.merge(report, df_sample_metadata, left_on='sample_id', right_index=True)
            report = report.reindex(columns=new_report_col_order)

            remaining_events_cols = df_events_all.columns.tolist()
            remaining_events_cols.remove(first_col)
            new_events_col_order = [first_col] + meta_cols + remaining_events_cols

            # Now merge and rearrange the column order
            df_events_all = pd.merge(df_events_all, df_sample_metadata, left_on='sample_id', right_index=True)
            df_events_all = df_events_all.reindex(columns=new_events_col_order)

            remaining_membership_cols = df_gate_membership_all_samples.columns.tolist()
            remaining_membership_cols.remove(first_col)
            new_membership_col_order = [first_col] + meta_cols + remaining_membership_cols

            # Now merge and rearrange the column order
            df_gate_membership_all_samples = pd.merge(
                df_gate_membership_all_samples,
                df_sample_metadata,
                left_on='sample_id',
                right_index=True
            )
            df_gate_membership_all_samples = df_gate_membership_all_samples.reindex(columns=new_membership_col_order)

        # Save all data to filesystem
        # Save panel to file
        proto_sample.channels[['channel_number', 'pnn', 'pns']].to_csv(panel_file_path, index=False)
        # Save gate tree
        gate_tree_file = open(gate_tree_file_path, file_write_mode)
        gate_tree_file.write(gate_tree)
        gate_tree_file.close()
        # Save gate LUT
        df_gate_lut.to_feather(gate_lut_file_path)
        # Save gate report
        report.to_feather(report_file_path)
        # Save all events
        df_events_all.to_feather(events_file_path)
        # Save gate membership
        df_gate_membership_all_samples.to_feather(member_file_path)

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
            gate_paths = self.find_matching_gate_paths(sample_id, gate_name)
            if len(gate_paths) > 1:
                raise GateReferenceError(
                    "Multiple gates exist with gate name '%s'. Specify a gate_path to disambiguate." % gate_name
                )
            gate_path = gate_paths[0]

        gate_id = (gate_name, gate_path)
        parent_gate_name = gate_path[-1]
        parent_gate_path = gate_path[:-1]

        # Get Sample instance and its GatingStrategy
        sample = self.get_sample(sample_id)
        gating_strategy = self.get_gating_strategy(sample_id)

        # get parent gate results to display only those events
        if parent_gate_name != 'root':
            # TODO:  make it clear to call analyze_samples prior to calling this method
            parent_event_mask = self.get_gate_membership(sample_id, parent_gate_name, parent_gate_path)
        else:
            parent_event_mask = None

        p = plot_utils.plot_gate(
            gate_id,
            gating_strategy,
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
            x_label,
            y_label,
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
        :param x_label: channel label (PnN) to use for the x-axis data
        :param y_label: channel label (PnN) to use for the y-axis data
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

        # Build Dimension instances for the requested x & y labels from
        # the dedicated comp matrix & transform set for this sample.
        comp_matrix = self.get_comp_matrix(sample_id)
        x_xform = self.get_transform(sample_id, x_label)
        y_xform = self.get_transform(sample_id, y_label)

        x_index = sample.get_channel_index(x_label)
        y_index = sample.get_channel_index(y_label)

        if comp_matrix is not None:
            comp_events = comp_matrix.apply(sample)
            x = comp_events[:, x_index]
            y = comp_events[:, y_index]
        else:
            # not doing subsample here, will do later with bool AND
            x = sample.get_channel_events(x_label, source='raw', subsample=False)
            y = sample.get_channel_events(y_label, source='raw', subsample=False)

        if x_xform is not None:
            x = x_xform.apply(x)
        if y_xform is not None:
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
