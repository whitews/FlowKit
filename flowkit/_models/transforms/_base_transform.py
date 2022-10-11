"""
Abstract base class for Transform classes
"""

from abc import ABC, abstractmethod
from copy import copy


class Transform(ABC):
    """
    Abstract base class for all transformation classes

    :param transform_id: A string identifying the transform
    """
    def __init__(
            self,
            transform_id
    ):
        self.id = transform_id
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
        """Tests where 2 transforms share the same attributes, ignoring the 'id' attribute."""
        if self.__class__ == other.__class__:
            this_attr = copy(self.__dict__)
            other_attr = copy(other.__dict__)
            this_attr.pop('id')
            other_attr.pop('id')
            return this_attr == other_attr
        else:
            return False
