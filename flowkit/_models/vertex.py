"""
Vertex class
"""


class Vertex(object):
    """
    Represents a single vertex of a polygon

    :param coordinates: tuple of floating point numbers of the vertex coordinates
    """
    def __init__(self, coordinates):
        self.coordinates = coordinates

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.coordinates})'
        )
