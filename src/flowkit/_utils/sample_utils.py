"""
Utilities for meta & bulk Sample operations
"""
import os
from glob import glob
import flowio
from .._models.sample import Sample


def _get_samples_from_paths(sample_paths):
    """
    Load multiple Sample instances from a list of file paths

    :param sample_paths: list of file paths containing FCS files
    :return: list of Sample instances
    """
    samples = []
    for path in sample_paths:
        samples.append(Sample(path))

    return samples


def load_samples(fcs_samples):
    """
    Returns a list of Sample instances from a variety of input types (fcs_samples), such as file or
        directory paths, a Sample instance, or lists of the previous types.

    :param fcs_samples: Sample, str, or list. Allowed types: a Sample instance, list of Sample instances,
            a directory or file path, or a list of directory or file paths. If a directory, any .fcs
            files in the directory will be loaded. If a list, then it must be a list of file paths or a
            list of Sample instances. Lists of mixed types are not supported.
    :return: list of Sample instances
    """
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
            sample_list = _get_samples_from_paths(fcs_samples)
    elif isinstance(fcs_samples, Sample):
        # 'fcs_samples' is a single Sample instance
        sample_list = [fcs_samples]
    elif isinstance(fcs_samples, str):
        # 'fcs_samples' is a str to either a single FCS file or a directory
        # If directory, search non-recursively for files w/ .fcs extension
        if os.path.isdir(fcs_samples):
            fcs_paths = glob(os.path.join(fcs_samples, '*.fcs'))
            if len(fcs_paths) > 0:
                sample_list = _get_samples_from_paths(fcs_paths)
        elif os.path.isfile(fcs_samples):
            sample_list = _get_samples_from_paths([fcs_samples])

    return sorted(sample_list)


def read_multi_dataset_fcs(
        filename_or_handle,
        ignore_offset_error=False,
        ignore_offset_discrepancy=False,
        use_header_offsets=False,
        cache_original_events=False
):
    """
    Utility function for reading all data sets in an FCS file containing multiple data sets.

    :param filename_or_handle: a path string or a file handle for an FCS file
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
    :param cache_original_events: Original events are the unprocessed events as stored in the FCS binary,
        meaning they have not been scaled according to channel gain, corrected for proper lin/log display,
        or had the time channel scaled by the 'timestep' keyword value (if present). By default, these
        events are not retained by the Sample class as they are typically not useful. To retrieve the
        original events, set this to True and call the get_events() method with source='orig'.
    :return: list of Sample instances
    """
    flow_data_list = flowio.read_multiple_data_sets(
        filename_or_handle,
        ignore_offset_error=ignore_offset_error,
        ignore_offset_discrepancy=ignore_offset_discrepancy,
        use_header_offsets=use_header_offsets
    )

    samples = []

    for fd in flow_data_list:
        s = Sample(fd, cache_original_events=cache_original_events)
        samples.append(s)

    return samples
