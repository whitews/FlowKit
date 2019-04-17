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
