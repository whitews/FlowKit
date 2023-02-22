"""
Utilities for meta & bulk Sample operations
"""
import os
from glob import glob
from .. import Sample


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
