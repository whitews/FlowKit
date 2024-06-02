from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from anytree import Resolver

    from .dimension import RatioDimension
    from .gate_node import GateNode
    from .gates._base_gate import Gate
    from .gating_results import GatingResults
    from .sample import Sample
    from .transforms._base_transform import Transform
    from .transforms._matrix import Matrix

class GatingStrategy(object):
    """
    Represents a flow cytometry gating strategy, including instructions
    for compensation and transformation.
    """

    resolver: Resolver = ...

    def __init__(self): ...
    def __repr__(self) -> str: ...
    def add_gate(
        self, gate: Gate, gate_path: Tuple[str], sample_id: Optional[str] = None
    ) -> None: ...
    def is_custom_gate(
        self, sample_id: str, gate_name: str, gate_path: Optional[Tuple[str]] = None
    ) -> bool: ...
    def get_gate(
        self,
        gate_name: str,
        gate_path: Optional[Tuple[str]] = None,
        sample_id: str = None,
    ): ...
    def _rebuild_dag(self) -> None: ...
    def remove_gate(
        self,
        gate_name: str,
        gate_path: Optional[Tuple[str]] = None,
        keep_children: bool = False,
    ) -> None: ...
    def add_transform(self, transform: Transform) -> None: ...
    def get_transform(self, transform_id: str) -> Transform: ...
    def add_comp_matrix(self, matrix: Matrix) -> None: ...
    def get_comp_matrix(self, matrix_id: str) -> Matrix: ...
    def _get_gate_node(
        self, gate_name: str, gate_path: Optional[Tuple[str]] = None
    ) -> GateNode: ...
    def find_matching_gate_paths(self, gate_name: str) -> List[Tuple[str]]: ...
    def get_root_gates(self, sample_id: Optional[str] = None) -> List[Gate]: ...
    def get_parent_gate_id(
        self, gate_name: str, gate_path: Optional[Tuple[str]] = None
    ) -> str: ...
    def get_child_gate_ids(
        self, gate_name: str, gate_path: Optional[Tuple[str]] = None
    ) -> List[str]: ...
    def get_gate_ids(self) -> List[Tuple[str]]: ...
    def get_max_depth(self) -> int: ...
    def get_gate_hierarchy(
        self, output: Literal["ascii", "dict", "JSON"] = "ascii", **kwargs
    ) -> Union[str, Dict[str, Any]]: ...
    def export_gate_hierarchy_image(self, output_file_path: str) -> None: ...
    def _get_cached_preprocessed_events(
        self,
        sample_id: str,
        comp_ref: str,
        xform_ref: str,
        dim_idx: Optional[int] = None,
    ) -> Optional[np.ndarray]: ...
    def clear_cache(self) -> None: ...
    def _cache_preprocessed_events(
        self,
        preprocessed_events: np.ndarray,
        sample_id: str,
        comp_ref: str,
        xform_ref: str,
        dim_idx: Optional[int] = None,
    ) -> None: ...
    def _compensate_sample(
        self, dim_comp_refs: List[str], sample: Sample
    ) -> np.ndarray: ...
    def _preprocess_sample_events(
        self, sample: Sample, gate: Gate, cache_events: bool = False
    ) -> pd.DataFrame: ...
    def _process_new_dims(
        self, sample: Sample, new_dims: List[RatioDimension]
    ) -> np.ndarray: ...
    @staticmethod
    def _apply_parent_results(
        sample: Sample, gate: Gate, results: Dict, parent_results: Dict
    ) -> Dict[str, Any]: ...
    def gate_sample(
        self, sample: Sample, cache_events: bool = False, verbose: bool = False
    ) -> GatingResults: ...
