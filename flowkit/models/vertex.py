class Vertex(object):
    def __init__(self, coordinates):
        self.coordinates = coordinates

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.coordinates})'
        )
