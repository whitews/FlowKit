"""
Utilities for meta & bulk Sample operations
"""
import os
from glob import glob
import numpy as np
import statsmodels.api as sm
import warnings
from .. import Sample, Matrix


# FCS 3.1 reserves certain keywords as being part of the FCS standard. Some
# of these are required, and others are optional. However, all of these
# keywords shall be prefixed by the '$' character. No other keywords shall
# begin with the '$' character. All keywords are case-insensitive, however
# most cytometers use all uppercase for keyword strings. FlowKit follows
# the convention used in FlowIO and internally stores and references all
# FCS keywords as lowercase for more convenient typing by developers.
FCS_STANDARD_KEYWORDS = [
    'beginanalysis',
    'begindata',
    'beginstext',
    'byteord',
    'datatype',
    'endanalysis',
    'enddata',
    'endstext',
    'mode',
    'nextdata',
    'par',
    'tot',
    # start optional standard keywords
    'abrt',
    'btim',
    'cells',
    'com',
    'csmode',
    'csvbits',
    'cyt',
    'cytsn',
    'data',
    'etim',
    'exp',
    'fil',
    'gate',
    'inst',
    'last_modified',
    'last_modifier',
    'lost',
    'op',
    'originality',
    'plateid',
    'platename',
    'proj',
    'smno',
    'spillover',
    'src',
    'sys',
    'timestep',
    'tr',
    'vol',
    'wellid'
]


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

    :param fcs_samples: str or list. If given a string, it can be a directory path or a file path.
            If a directory, any .fcs files in the directory will be loaded. If a list, then it must
            be a list of file paths or a list of Sample instances. Lists of mixed types are not
            supported.
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

    return sample_list


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

        for channel_idx, lut in bead_lut.items():
            # file names typically don't have the "-A", "-H', or "-W" sub-strings
            pnn_label = lut['pnn_label'].replace("-A", "")

            if pnn_label in bs.original_filename:
                lut['bead_index'] = i
                lut['pns_label'] = bs.pns_labels[channel_idx]

    return bead_lut


def calculate_compensation_from_beads(comp_bead_samples, matrix_id='comp_bead'):
    """
    Calculates spillover from a list of FCS bead files.

    :param comp_bead_samples: str or list. If given a string, it can be a directory path or a file path.
        If a directory, any .fcs files in the directory will be loaded. If a list, then it must
        be a list of file paths or a list of Sample instances. Lists of mixed types are not
        supported.
    :param matrix_id: label for the calculated Matrix
    :return: a Matrix instance
    """
    bead_samples = load_samples(comp_bead_samples)
    bead_lut = _process_bead_samples(bead_samples)
    if len(bead_lut) == 0:
        warnings.warn("No bead samples were loaded")
        return

    detectors = []
    fluorochromes = []
    comp_values = []
    for channel_idx in sorted(bead_lut.keys()):
        detectors.append(bead_lut[channel_idx]['pnn_label'])
        fluorochromes.append(bead_lut[channel_idx]['pns_label'])
        bead_idx = bead_lut[channel_idx]['bead_index']

        x = bead_samples[bead_idx].get_events(source='raw')[:, channel_idx]
        good_events = x < (2 ** 18) - 1
        x = x[good_events]

        comp_row_values = []
        for channel_idx2 in sorted(bead_lut.keys()):
            if channel_idx == channel_idx2:
                comp_row_values.append(1.0)
            else:
                y = bead_samples[bead_idx].get_events(source='raw')[:, channel_idx2]
                y = y[good_events]
                rlm_res = sm.RLM(y, x).fit()

                # noinspection PyUnresolvedReferences
                comp_row_values.append(rlm_res.params[0])

        comp_values.append(comp_row_values)

    return Matrix(matrix_id, np.array(comp_values), detectors, fluorochromes)
