from typing import TYPE_CHECKING, Any, Dict, List, Literal, Tuple, Union

if TYPE_CHECKING:
    import numpy as np

    from ..dimension import Dimension, QuadrantDivider
    from ._base_gate import Gate

class RectangleGate(Gate):
    def __init__(
        self, gate_name: str, dimensions: List[Dimension], use_complement: bool = False
    ): ...
    def __repr__(self) -> str: ...
    def apply(self, df_events) -> np.ndarray[bool]: ...

class PolygonGate(Gate):
    def __init__(
        self,
        gate_name: str,
        dimensions: List[Dimension],
        vertices: List[Tuple[float, float]],
        use_complement: bool = False,
    ): ...
    def __repr__(self) -> str: ...
    def apply(self, df_events) -> np.ndarray[bool]: ...

class EllipsoidGate(Gate):
    def __init__(
        self,
        gate_name: str,
        dimensions: List[Dimension],
        coordinates: Tuple[float, float],
        covariance_matrix: Union[List[List[float]], np.ndarray[float]],
        distance_square: int,
    ): ...
    def __repr__(self) -> str: ...
    def apply(self, df_events) -> np.ndarray[bool]: ...

class Quadrant(object):
    def __init__(
        self,
        quadrant_id: str,
        divider_refs: List[str],
        divider_ranges: List[Tuple[float, float]],
    ): ...
    def get_divider_range(self, div_ref: str) -> Tuple[float, float]: ...
    def __repr__(self) -> str: ...

class QuadrantGate(Gate):
    def __init__(
        self, gate_name: str, dividers: List[QuadrantDivider], quadrants: List[Quadrant]
    ): ...
    def __repr__(self) -> str: ...
    def apply(self, df_events) -> Dict[str, np.ndarray[bool]]: ...

class BooleanGate(Gate):
    def __init__(
        self,
        gate_name: str,
        bool_type: Literal["and", "or", "not"],
        gate_refs: List[Dict[str, Any]],
    ): ...
    def __repr__(self) -> str: ...
    def apply(self, df_events) -> np.ndarray[bool]: ...
