import numpy as np
import flowutils
from ... import _utils


class Matrix(object):
    """
    Represents a single compensation matrix from a CSV/TSV file, NumPy array or Pandas
    DataFrame.
    """
    def __init__(
            self,
            matrix_id,
            spill_data_or_file,
            detectors,
            fluorochromes,
            null_channels=None
    ):
        """
        Create a Sample instance

        :param matrix_id: Text string used to identify the matrix
        :param spill_data_or_file: matrix data array, can be either:
                - a file path or file handle to a CSV/TSF file
                - a pathlib Path object to a CSV/TSF file
                - a NumPy array of spill data
                - a Pandas DataFrame (channel labels as headers)
        :param detectors: A list of strings or a list of tuples to use for the detector
            labels.
        :param fluorochromes: A list of strings or a list of tuples to use for the detector
            labels.
        :param null_channels: List of PnN labels for channels that were collected
            but do not contain useful data. Note, this should only be used if there were
            truly no fluorochromes used targeting those detectors and the channels
            do not contribute to compensation.
        """
        if isinstance(spill_data_or_file, np.ndarray):
            spill = spill_data_or_file
        else:
            spill = _utils.parse_compensation_matrix(
                spill_data_or_file,
                detectors,
                null_channels=null_channels
            )
            spill = spill[1:, :]

        self.id = matrix_id
        self.matrix = spill
        self.detectors = detectors
        # Note: fluorochromes attribute is required for compatibility with GatingML exports,
        #       as the GatingML 2.0 requires both the set of detectors and fluorochromes.
        self.fluorochomes = fluorochromes

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.id}, dims: {len(self.detectors)})'
        )

    def apply(self, sample):
        indices = [
            sample.get_channel_index(d) for d in self.detectors
        ]
        events = sample.get_raw_events()
        events = events.copy()

        return flowutils.compensate.compensate(
            events,
            self.matrix,
            indices
        )
