"""
Utilities for meta & bulk Sample operations
"""
import os
from glob import glob
from pathlib import Path
import flowio
from .._models.sample import Sample


def _get_samples_from_paths(
        sample_paths,
        filename_as_id=False,
        compensation=None,
        null_channel_list=None,
        preprocess=True,
        use_flowjo_labels=False
):
    """
    Load multiple Sample instances from a list of file paths or Path-like objects.

    :param sample_paths: list of file paths containing FCS files
    :param filename_as_id: Boolean option for using the file name (as it exists on the
        filesystem) for the Sample's ID, default is False.
    :param compensation: Compensation matrix. The matrix must be applicable to all samples in
        'fcs_samples'. Acceptable types include a Matrix instance, NumPy array, CSV file path,
        pathlib Path object to a CSV or TSV file, or a string of CSV text
    :param null_channel_list: List of PnN labels for acquired channels that do not contain
        useful data. Note, this should only be used if no fluorochromes were used to target
        those detectors. Null channels do not contribute to compensation and should not be
        included in a compensation matrix for this sample. This option is ignored if
        `fcs_path_or_data` is a FlowData object. The null channel list must be applicable
        to all samples.
    :param preprocess: Controls whether preprocessing is applied to the 'raw' data (retrievable
        via Sample.get_events() with source='raw'). Binary events in an FCS file are stored
        unprocessed, meaning they have not been scaled according to channel gain, corrected for
        proper lin/log display, or had the time channel scaled by the 'timestep' keyword value
        (if present). Unprocessed event data is typically not useful for analysis, so the default
        is True. Preprocessing does not include compensation or transformation (e.g. biex, Logicle)
        which are separate operations.
    :param use_flowjo_labels: FlowJo converts forward slashes ('/') in PnN labels to underscores.
        This option matches that behavior. Default is False.
    :return: list of Sample instances
    """
    samples = []
    for path in sample_paths:
        samples.append(
            Sample(
                path,
                filename_as_id=filename_as_id,
                compensation=compensation,
                null_channel_list=null_channel_list,
                preprocess=preprocess,
                use_flowjo_labels=use_flowjo_labels
            )
        )

    return samples


def load_samples(
        fcs_samples,
        filename_as_id=False,
        compensation=None,
        null_channel_list=None,
        preprocess=True,
        use_flowjo_labels=False
):
    """
    Returns a list of Sample instances from a variety of input types (fcs_samples), such as a file or
        directory path string, Path object, a Sample instance, or a list of the previous types. Lists
        of mixed types are not supported.

    :param fcs_samples: Sample, str, or list. Allowed types: a Sample instance, list of Sample instances,
            a directory or file path, or a list of directory or file paths. If a directory, any .fcs
            files in the directory will be loaded. If a list, then it must be a list of file paths or a
            list of Sample instances. Lists of mixed types are not supported.
    :param filename_as_id: Boolean option for using the file name (as it exists on the
        filesystem) for the Sample's ID, default is False. Only applies to file paths given to the
        'fcs_samples' argument.
    :param compensation: Compensation matrix. The matrix must be applicable to all samples in
        'fcs_samples'. Acceptable types include a Matrix instance, NumPy array, CSV file path,
        pathlib Path object to a CSV or TSV file, or a string of CSV text
    :param null_channel_list: List of PnN labels for acquired channels that do not contain
        useful data. Note, this should only be used if no fluorochromes were used to target
        those detectors. Null channels do not contribute to compensation and should not be
        included in a compensation matrix for this sample. This option is ignored if
        `fcs_path_or_data` is a FlowData object. The null channel list must be applicable
        to all samples.
    :param preprocess: Controls whether preprocessing is applied to the 'raw' data (retrievable
        via Sample.get_events() with source='raw'). Binary events in an FCS file are stored
        unprocessed, meaning they have not been scaled according to channel gain, corrected for
        proper lin/log display, or had the time channel scaled by the 'timestep' keyword value
        (if present). Unprocessed event data is typically not useful for analysis, so the default
        is True. Preprocessing does not include compensation or transformation (e.g. biex, Logicle)
        which are separate operations.
    :param use_flowjo_labels: FlowJo converts forward slashes ('/') in PnN labels to underscores.
        This option matches that behavior. Default is False.
    :return: list of Sample instances
    """
    sample_list = []
    load_from_paths = None

    if isinstance(fcs_samples, list):
        # 'fcs_samples' is a list of either file paths or Sample instances
        sample_types = set()

        for sample in fcs_samples:
            sample_types.add(type(sample))

        if len(sample_types) > 1:
            raise ValueError(
                "Found multiple object types for 'fcs_samples' option. " 
                "Each item in 'fcs_samples' list must be of the same type (FCS file path or Sample instance)."
            )

        # Returning a list of Sample instances given that same list of samples
        # seems pointless. This is here for other classes (Session, Workspace)
        # to load samples without having to do any type checking.
        if Sample in sample_types:
            sample_list = fcs_samples
        elif str in sample_types or Path in sample_types:
            load_from_paths = fcs_samples
    elif isinstance(fcs_samples, Sample):
        # 'fcs_samples' is a single Sample instance
        sample_list = [fcs_samples]
    elif isinstance(fcs_samples, (str, Path)):
        # 'fcs_samples' is a str to either a single FCS file or a directory
        # If directory, search non-recursively for files w/ .fcs extension
        if os.path.isdir(fcs_samples):
            fcs_paths = glob(os.path.join(fcs_samples, '*.fcs'))
            if len(fcs_paths) > 0:
                load_from_paths = fcs_paths
        else:
            # assume a path to a single FCS file
            load_from_paths = [fcs_samples]

    if load_from_paths is not None:
        sample_list = _get_samples_from_paths(
            load_from_paths,
            filename_as_id=filename_as_id,
            compensation=compensation,
            null_channel_list=null_channel_list,
            preprocess=preprocess,
            use_flowjo_labels=use_flowjo_labels
        )

    return sorted(sample_list)


def read_multi_dataset_fcs(
        fcs_file,
        ignore_offset_error=False,
        ignore_offset_discrepancy=False,
        use_header_offsets=False,
        preprocess=True
):
    """
    Utility function for reading all data sets in an FCS file containing multiple data sets.

    :param fcs_file: a file path string, Path instance, or file handle to an FCS file
    :param ignore_offset_error: option to ignore data offset error (see notes in Sample class docstring),
        default is False
    :param ignore_offset_discrepancy: option to ignore discrepancy between the HEADER
        and TEXT values for the DATA byte offset location, default is False
    :param use_header_offsets: use the HEADER section for the data offset locations, default is False.
        Setting this option to True also suppresses an error in cases of an offset discrepancy.
    :param preprocess: Original events are the unprocessed events as stored in the FCS binary,
        meaning they have not been scaled according to channel gain, corrected for proper lin/log display,
        or had the time channel scaled by the 'timestep' keyword value (if present). This option controls
        whether to perform these preprocessing steps. Unprocessed event data is typically not useful
        for analysis, so the default is True.
    :return: list of Sample instances
    """
    flow_data_list = flowio.read_multiple_data_sets(
        fcs_file,
        ignore_offset_error=ignore_offset_error,
        ignore_offset_discrepancy=ignore_offset_discrepancy,
        use_header_offsets=use_header_offsets
    )

    samples = []

    for fd in flow_data_list:
        s = Sample(fd, preprocess=preprocess)
        samples.append(s)

    return samples


def extract_fcs_metadata(fcs_file):
    """
    Extract only the metadata from an FCS file without parsing the event data. This
    significantly speeds up parsing FCS files for the use case of retrieving just
    the metadata.

    :param fcs_file: a file path string, Path instance, or file handle to an FCS file
    :return:
    """
    # Using 'use_header_offsets' to avoid and DATA offset discrepancies, we don't
    # need to worry about those for just getting the TEXT metadata
    fd = flowio.FlowData(fcs_file, only_text=True, use_header_offsets=True)

    return fd.text
