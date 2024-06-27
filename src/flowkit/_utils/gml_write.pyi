from typing import IO, TYPE_CHECKING, Dict, Optional, Union

if TYPE_CHECKING:
    from pathlib import Path

    from lxml.etree import _Element

    from .._models.gates._base_gate import Gate
    from .._models.gating_strategy import GatingStrategy
    from .._models.transforms._base_transform import Transform
    from .._models.transforms._matrix import Matrix

def _add_matrix_to_gml(
    root: _Element, matrix: Matrix, ns_map: Dict[str, str]
) -> None: ...
def _add_transform_to_gml(
    root: _Element, transform: Transform, ns_map: Dict[str, str]
) -> None: ...
def _add_gate_to_gml(
    root: _Element, gate: Gate, ns_map: Dict[str, str]
) -> _Element: ...
def _add_gates_from_gate_dict(
    gating_strategy: GatingStrategy,
    gate_dict: Dict[str, Gate],
    ns_map: Dict[str, str],
    parent_ml: _Element,
    sample_id: Optional[str] = None,
) -> None: ...
def export_gatingml(
    gating_strategy: GatingStrategy,
    file_handle: Union[str, IO, Path],
    sample_id: Optional[str] = None,
) -> None: ...
