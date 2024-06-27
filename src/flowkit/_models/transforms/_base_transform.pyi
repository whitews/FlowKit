from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    import numpy as np

    from ..sample import Sample

class Transform(ABC):
    def __init__(self, transform_id: str): ...
    @abstractmethod
    def apply(self, events_or_sample: Union[np.ndarray, Sample]) -> Any: ...
    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
