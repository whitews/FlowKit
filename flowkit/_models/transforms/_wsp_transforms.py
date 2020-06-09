"""
Transform classes compatible with FlowJo 10
"""
import numpy as np
from ._base_transform import Transform


class WSPLogTransform(Transform):
    """
    Logarithmic transform as implemented in FlowJo 10.

    :param transform_id: A string identifying the transform
    :param offset: A positive number used to offset event data
    :param decades: A positive number of for the desired number of decades
    """
    def __init__(
        self,
        transform_id,
        offset,
        decades
    ):
        Transform.__init__(self, transform_id)

        self.offset = offset
        self.decades = decades

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.id}, offset: {self.offset}, decades: {self.decades})'
        )

    def apply(self, events):
        """
        Apply transform to given events.

        :param events: NumPy array of FCS event data
        :return: NumPy array of transformed events
        """
        new_events = np.copy(events)
        new_events[new_events < self.offset] = self.offset
        new_events = 1. / self.decades * (np.log10(new_events) - np.log10(self.offset))

        return new_events
