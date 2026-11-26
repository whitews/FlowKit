"""
Matrix class
"""
from copy import copy
import os
from pathlib import Path
import numpy as np
import pandas as pd
import flowutils
from ...exceptions import FlowKitException

_NON_FLUORO_CHANNELS = (
    'FSC-A',
    'FSC-H',
    'FSC-W',
    'SSC-A',
    'SSC-H',
    'SSC-W',
    'Time',
)


def _build_label_mismatch_error(header_labels, fluoro_labels):
    in_fcs_not_comp = [label for label in fluoro_labels if label not in header_labels]
    in_comp_not_fcs = [label for label in header_labels if label not in fluoro_labels]

    error_lines = ["Matrix labels do not match given fluorescent labels"]

    if in_fcs_not_comp:
        error_lines.extend(
            [
                "",
                "Labels in FCS file not found in comp matrix (null channels?):",
                ", ".join(in_fcs_not_comp),
                "",
                "Null channels can be specified when creating a Sample instance",
            ]
        )

    if in_comp_not_fcs:
        error_lines.extend(
            [
                "",
                "Labels in comp matrix not found in FCS file (wrong matrix chosen?):",
                ", ".join(in_comp_not_fcs),
            ]
        )

    return "\n".join(error_lines)


def _read_spill_text(spill_data_or_file):
    if isinstance(spill_data_or_file, Path):
        with spill_data_or_file.open('r') as fh:
            return fh.read()

    if isinstance(spill_data_or_file, str):
        try:
            is_file = os.path.isfile(spill_data_or_file)
        except (TypeError, ValueError, OSError):
            is_file = False

        if is_file:
            with open(spill_data_or_file, 'r') as fh:
                return fh.read()

        return spill_data_or_file

    return None


def _parse_spill_matrix(
        spill_data_or_file,
        detectors,
        null_channels=None,
        channel_labels=None,
        orig_channel_labels=None,
        fluoro_indices=None
):
    matrix_text = _read_spill_text(spill_data_or_file)

    if matrix_text is not None and not isinstance(detectors, str):
        lines = matrix_text.splitlines()
        if len(lines) == 1:
            tokens = lines[0].split(',')
            try:
                n = int(tokens[0].strip())
            except (IndexError, ValueError):
                n = None

            if n is not None and n > 0 and len(tokens) == 1 + n + n * n:
                headers = [item.strip() for item in tokens[1:n + 1]]

                non_fluoro = set(_NON_FLUORO_CHANNELS)
                if null_channels is not None:
                    non_fluoro.update(null_channels)
                expected_detectors = [
                    label for label in detectors if label not in non_fluoro
                ]

                # Exact $PnN label match always takes precedence over 1-based channel indices.
                exact_match = (
                    len(headers) == len(expected_detectors)
                    and set(headers) == set(expected_detectors)
                )

                if not exact_match and channel_labels is not None:
                    source_channel_labels = list(channel_labels)
                    known_channel_labels = set(source_channel_labels)
                    if orig_channel_labels is not None:
                        known_channel_labels.update(orig_channel_labels)
                    if null_channels is not None:
                        known_channel_labels.update(null_channels)

                    # Only fall back to 1-based channel indices when channel_labels is
                    # explicitly provided and zero header tokens match any current,
                    # original, or declared-null $PnN channel name.
                    # Otherwise, delegate to standard $PnN label validation to prevent
                    # silent channel mis-mapping (e.g., after Sample.rename_channel or
                    # when numeric $PnN names overlap 1-based indices).
                    if set(headers).isdisjoint(known_channel_labels):
                        channel_number_map = {
                            str(idx): idx - 1
                            for idx in range(1, len(source_channel_labels) + 1)
                        }

                        if all(h in channel_number_map for h in headers):
                            if len(set(headers)) != len(headers):
                                raise ValueError(
                                    "Spillover matrix contains duplicate channel indices: %s"
                                    % ", ".join(headers)
                                )

                            resolved_idx0 = [channel_number_map[h] for h in headers]
                            resolved_headers = [
                                source_channel_labels[idx0] for idx0 in resolved_idx0
                            ]
                            value_tokens = [item.strip() for item in tokens[n + 1:]]
                            matrix = np.reshape(
                                list(map(float, value_tokens)),
                                (n, n)
                            )

                            if fluoro_indices is not None:
                                fluoro_idx_list = list(fluoro_indices)
                                if (
                                    len(set(fluoro_idx_list)) != len(fluoro_idx_list)
                                    or any(
                                        not isinstance(i, (int, np.integer))
                                        or isinstance(i, bool)
                                        or i < 0
                                        or i >= len(source_channel_labels)
                                        for i in fluoro_idx_list
                                    )
                                ):
                                    raise ValueError(
                                        "Invalid fluoro_indices for channel_labels of length %d"
                                        % len(source_channel_labels)
                                    )

                                if (
                                    len(resolved_idx0) != len(fluoro_idx_list)
                                    or set(resolved_idx0) != set(fluoro_idx_list)
                                ):
                                    if set(resolved_headers) != set(expected_detectors):
                                        raise ValueError(
                                            _build_label_mismatch_error(
                                                resolved_headers,
                                                expected_detectors
                                            )
                                        )
                                    raise ValueError(
                                        "Spillover matrix channel indices do not match "
                                        "fluorescent channel indices"
                                    )

                            duplicate_labels = [
                                label for label in resolved_headers
                                if source_channel_labels.count(label) > 1
                            ]
                            if duplicate_labels:
                                raise ValueError(
                                    "Cannot resolve numeric spillover indices because "
                                    "channel labels are not unique: %s"
                                    % ", ".join(dict.fromkeys(duplicate_labels))
                                )

                            if (
                                any(label in non_fluoro for label in resolved_headers)
                                or len(resolved_headers) != len(expected_detectors)
                                or len(set(resolved_headers)) != len(resolved_headers)
                                or set(resolved_headers) != set(expected_detectors)
                            ):
                                raise ValueError(
                                    _build_label_mismatch_error(
                                        resolved_headers,
                                        expected_detectors
                                    )
                                )

                            idx_order = [
                                resolved_headers.index(label)
                                for label in expected_detectors
                            ]
                            return matrix[idx_order, :][:, idx_order]

    spill = flowutils.compensate.parse_compensation_matrix(
        spill_data_or_file,
        detectors,
        null_channels=null_channels
    )
    return spill[1:, :]


class Matrix(object):
    """
    Represents a single compensation matrix from a CSV/TSV file, NumPy array or pandas
    DataFrame.

    :param spill_data_or_file: matrix data array, can be either:
            - text string from FCS $SPILL or $SPILLOVER metadata value
            - a file path or file handle to a CSV/TSF file
            - a pathlib Path object to a CSV/TSF file
            - a NumPy array of spill data
    :param detectors: A list of strings to use for the detector labels.
    :param fluorochromes: A list of strings to use for the fluorochrome labels.
    :param null_channels: List of PnN labels for channels that were collected
        but do not contain useful data. Note, this should only be used if there were
        truly no fluorochromes used targeting those detectors and the channels
        do not contribute to compensation.
    :param channel_labels: Optional full ordered list of physical FCS $PnN channel
        labels (including scatter, time, and null channels). Required to resolve
        non-standard FCS $SPILLOVER headers that specify 1-based channel numbers
        instead of $PnN labels (such as Beckman Coulter Navios LMD/FCS files);
        when omitted (None), numeric headers are not resolved positionally and
        delegate directly to standard $PnN label validation. Positional resolution
        is also refused whenever any header token matches a current, original, or
        declared-null $PnN channel name in the file, or when any resolved display
        label appears more than once in channel_labels.
    :param orig_channel_labels: Optional original raw FCS $PnN channel labels
        prior to in-memory renames (e.g., Sample.rename_channel). Checked
        alongside channel_labels so renamed numeric $PnN headers are never
        silently reinterpreted as 1-based channel indices.
    :param fluoro_indices: Optional 0-based physical fluorescent channel indices
        into channel_labels. When supplied, 1-based numeric spillover headers must
        resolve to the exact set of physical fluorescent indices, rejecting scatter,
        time, or null channel indices even if their display labels alias a
        fluorescent channel.
    """
    def __init__(
            self,
            spill_data_or_file,
            detectors,
            fluorochromes=None,
            null_channels=None,
            channel_labels=None,
            orig_channel_labels=None,
            fluoro_indices=None
    ):
        # Copy detectors b/c it may be modified
        detectors = copy(detectors)

        # TODO: accept the same CSV w/ headers that is exported & take Pandas DataFrame

        if isinstance(spill_data_or_file, np.ndarray):
            spill = spill_data_or_file
        else:
            spill = _parse_spill_matrix(
                spill_data_or_file,
                detectors,
                null_channels=null_channels,
                channel_labels=channel_labels,
                orig_channel_labels=orig_channel_labels,
                fluoro_indices=fluoro_indices
            )

        self.matrix = spill

        # Remove any null channels from detector list
        if null_channels is not None:
            for nc in null_channels:
                if nc in detectors:
                    detectors.remove(nc)

        self.detectors = detectors

        # TODO: Should we use a different name other than 'fluorochromes'? They are typically antibodies or markers.
        # Note: fluorochromes attribute is required for compatibility with GatingML exports,
        #       as the GatingML 2.0 requires both the set of detectors and fluorochromes.
        if fluorochromes is None:
            fluorochromes = ['' for _ in detectors]

        self.fluorochromes = fluorochromes

    def __repr__(self):
        return f'{self.__class__.__name__}(dims: {len(self.detectors)})'

    def __eq__(self, other):
        """
        Tests where 2 matrices share the same attributes.

        :param other: Matrix instance to compare
        """
        if self.__class__ == other.__class__:
            this_attr = copy(self.__dict__)
            other_attr = copy(other.__dict__)

            # ignore 'private' attributes
            this_delete = [k for k in this_attr.keys() if k.startswith('_')]
            other_delete = [k for k in other_attr.keys() if k.startswith('_')]
            for k in this_delete:
                del this_attr[k]
            for k in other_delete:
                del other_attr[k]

            # pop matrix attribute, need to compare NumPy array differently
            this_matrix = this_attr.pop('matrix')
            other_matrix = other_attr.pop('matrix')

            if not np.array_equal(this_matrix, other_matrix):
                return False

            return this_attr == other_attr
        else:
            return False

    def apply(self, sample):
        """
        Apply compensation matrix to given Sample instance.

        :param sample: Sample instance with matching set of detectors
        :return: NumPy array of compensated events
        """
        # Check that sample fluoro channels match the
        # matrix detectors
        sample_fluoro_labels = [sample.pnn_labels[i] for i in sample.fluoro_indices]
        if not set(self.detectors).issubset(sample_fluoro_labels):
            raise FlowKitException("Detectors must be a subset of the Sample's fluorochromes")

        indices = [
            sample.get_channel_index(d) for d in self.detectors
        ]
        events = sample.get_events(source='raw')

        return flowutils.compensate.compensate(
            events,
            self.matrix,
            indices
        )

    def inverse(self, sample):
        """
        Apply compensation matrix to given Sample instance.

        :param sample: Sample instance with matching set of detectors
        :return: NumPy array of compensated events
        """
        # Check that sample fluoro channels match the
        # matrix detectors
        sample_fluoro_labels = [sample.pnn_labels[i] for i in sample.fluoro_indices]
        if not set(self.detectors).issubset(sample_fluoro_labels):
            raise FlowKitException("Detectors must be a subset of the Sample's fluorochromes")

        indices = [
            sample.get_channel_index(d) for d in self.detectors
        ]
        events = sample.get_events(source='comp')

        return flowutils.compensate.inverse_compensate(
            events,
            self.matrix,
            indices
        )

    def as_dataframe(self, fluoro_labels=False):
        """
        Returns the compensation matrix as a pandas DataFrame.

        :param fluoro_labels: If True, the fluorochrome names are used as the column headers & row indices, else
            the detector names are used (default).
        :return: pandas DataFrame
        """
        if fluoro_labels:
            labels = self.fluorochromes
        else:
            labels = self.detectors

        return pd.DataFrame(self.matrix, columns=labels, index=labels)


class SpectralMatrix(object):
    """
    Represents a spectral compensation matrix using the ordinary least squares
    method (OLS). Unlike conventional flow cytometry where one detector is
    used per fluorochrome, spectral flow cytometry utilizes a larger number of
    detectors than fluorochromes. The corresponding spectral matrix (M x N) is
    not square, with more columns than rows (N > M). The rows (M) in the matrix
    correspond to the "true" fluorochromes (i.e. those detectors with dedicated
    fluorochromes). Note, there is no inverse method for this method of
    compensation.

    :param spill_data_array: NumPy array of matrix data
    :param detectors: The full list of strings for all the detector labels.
    :param true_detectors: The list of strings for the "true" detector labels
        with dedicated fluorochromes.
    """
    def __init__(
            self,
            spill_data_array,
            detectors,
            true_detectors
    ):
        # Check that the order of the true detectors align with the first
        # elements in detectors
        for i, true_detector in enumerate(true_detectors):
            if true_detector != detectors[i]:
                raise FlowKitException("`true_detectors` must match the first elements of `detectors`")

        self.matrix = spill_data_array
        self.detectors = detectors
        self.true_detectors = true_detectors

    def __repr__(self):
        return f'{self.__class__.__name__}(detectors: {len(self.detectors)}, true_detectors: {len(self.true_detectors)})'

    def __eq__(self, other):
        """
        Tests where 2 matrices share the same attributes.

        :param other: SpectralMatrix instance to compare
        """
        if self.__class__ == other.__class__:
            this_attr = copy(self.__dict__)
            other_attr = copy(other.__dict__)

            # ignore 'private' attributes
            this_delete = [k for k in this_attr.keys() if k.startswith('_')]
            other_delete = [k for k in other_attr.keys() if k.startswith('_')]
            for k in this_delete:
                del this_attr[k]
            for k in other_delete:
                del other_attr[k]

            # pop matrix attribute, need to compare NumPy array differently
            this_matrix = this_attr.pop('matrix')
            other_matrix = other_attr.pop('matrix')

            if not np.array_equal(this_matrix, other_matrix):
                return False

            return this_attr == other_attr
        else:
            return False

    def apply(self, sample):
        """
        Apply ordinary least squares (OLS) spectral compensation matrix to
        given Sample instance.

        :param sample: Sample instance with matching set of detectors
        :return: NumPy array of compensated events
        """
        # Check that matrix detectors is a subset of the sample's fluoro
        # channels
        sample_fluoro_labels = [sample.pnn_labels[i] for i in sample.fluoro_indices]
        if not set(self.detectors).issubset(sample_fluoro_labels):
            raise FlowKitException("Detectors must be a subset of the Sample channels")

        indices = [
            sample.get_channel_index(d) for d in self.detectors
        ]
        events = sample.get_events(source='raw')

        return flowutils.compensate.compensate_spectral_ols(
            events,
            self.matrix,
            indices
        )

    def as_dataframe(self):
        """
        Returns the spectral compensation matrix as a pandas DataFrame. The columns
        are all the detectors, the row indices indicate the true detectors.

        :return: pandas DataFrame
        """
        detector_labels = self.detectors
        true_detector_index = self.true_detectors

        return pd.DataFrame(self.matrix, columns=detector_labels, index=true_detector_index)
