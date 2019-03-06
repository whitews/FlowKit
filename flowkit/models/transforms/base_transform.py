from abc import ABC, abstractmethod


class Transform(ABC):
    def __init__(
            self,
            transform_id
    ):
        self.id = transform_id
        self.dimensions = []

    @abstractmethod
    def apply(self, events_or_sample):
        pass
