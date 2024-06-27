from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from lxml.etree import _Element

    from ..transforms._base_transform import Transform
    from ._base_gate import Gate
    from ._gates import PolygonGate

def _rotate_point_around_point(
    point: Tuple[float, float],
    cov_mat: np.ndarray,
    center_point: Tuple[float, float] = (0, 0),
): ...

class WSPEllipsoidGate(Gate):
    def __init__(
        self,
        gate_element: _Element,
        gating_namespace: str,
        data_type_namespace: str,
        use_complement: bool = False,
    ): ...
    @staticmethod
    def _parse_coordinate_elements(
        parent_element: _Element, gating_namespace: str, data_type_namespace: str
    ) -> List[Tuple[float, float]]: ...
    def _parse_foci_elements(
        self, gate_element: _Element, gating_namespace: str, data_type_namespace: str
    ) -> np.ndarray: ...
    def _parse_edge_elements(
        self, gate_element: _Element, gating_namespace: str, data_type_namespace: str
    ) -> np.ndarray: ...
    def convert_to_polygon_gate(
        self, transforms: List[Transform], n_vertices: int = 128
    ) -> PolygonGate: ...
    def apply(self, df_events: pd.DataFrame) -> None: ...
