import flowutils


class Matrix(object):
    def __init__(
        self,
        matrix_id,
        fluorochromes,
        detectors,
        matrix
    ):
        self.id = matrix_id
        self.fluorochomes = fluorochromes
        self.detectors = detectors
        self.matrix = matrix

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.id}, dims: {len(self.fluorochomes)})'
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
