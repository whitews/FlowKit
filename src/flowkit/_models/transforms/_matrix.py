"""
Matrix class
"""
from copy import copy
import numpy as np
import pandas as pd
import flowutils
from ...exceptions import FlowKitException


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
    """
    def __init__(
            self,
            spill_data_or_file,
            detectors,
            fluorochromes=None,
            null_channels=None
    ):
        # Copy detectors b/c it may be modified
        detectors = copy(detectors)

        # TODO: accept the same CSV w/ headers that is exported & take Pandas DataFrame

        if isinstance(spill_data_or_file, np.ndarray):
            spill = spill_data_or_file
        else:
            spill = flowutils.compensate.parse_compensation_matrix(
                spill_data_or_file,
                detectors,
                null_channels=null_channels
            )
            spill = spill[1:, :]

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
