"""
Abstract base class for Transform classes
"""

from abc import ABC, abstractmethod
from copy import copy


class Transform(ABC):
    """
    Abstract base class for all transformation classes
    """
    def __init__(self,):
        self.dimensions = []

    @abstractmethod
    def apply(self, events_or_sample):
        """
        Abstract method for applying the transform to a set of events.

        :param events_or_sample: A NumPy array of event data or, in some cases, a Sample instance.
          See subclass documentation for specific implementation.
        """
        return

    def __eq__(self, other):
        """Tests where 2 transforms share the same attributes."""
        if self.__class__ == other.__class__:
            this_attr = copy(self.__dict__)
            other_attr = copy(other.__dict__)

            # also ignore 'private' attributes
            this_delete = [k for k in this_attr.keys() if k.startswith('_')]
            other_delete = [k for k in other_attr.keys() if k.startswith('_')]
            for k in this_delete:
                del this_attr[k]
            for k in other_delete:
                del other_attr[k]

            return this_attr == other_attr
        else:
            return False
