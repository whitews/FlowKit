"""
Abstract base class for Transform classes
"""

from abc import ABC, abstractmethod


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

        :param events_or_sample: A NumPy array of event data or, in some cases, a Sample instance. See sub-class
            documentation for specific implementation.
        """
        return
