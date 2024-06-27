from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    import pandas as pd

    from ..dimension import Dimension

class Gate(ABC):
    def __init__(self, gate_name: str, dimensions: Optional[List[Dimension]]): ...
    def get_dimension(self, dim_id: str) -> Dimension: ...
    def get_dimension_ids(self) -> List[str]: ...
    @abstractmethod
    def apply(self, df_events: pd.DataFrame) -> Any: ...
