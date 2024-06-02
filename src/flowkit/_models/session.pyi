from typing import IO, TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

if TYPE_CHECKING:
    import io

    import numpy as np
    import pandas as pd
    from bokeh.plotting import figure

    from .dimension import Dimension
    from .gates._base_gate import Gate
    from .gating_results import GatingResults
    from .gating_strategy import GatingStrategy
    from .sample import Sample
    from .transforms._base_transform import Transform
    from .transforms._matrix import Matrix

class Session(object):
    def __init__(
        self,
        gating_strategy: Optional[Union[GatingStrategy, str, io.IOBase]] = None,
        fcs_samples: Optional[Union[str, List[str], List[Sample]]] = None,
    ): ...
    def __repr__(self) -> str: ...
    def add_samples(self, fcs_samples: Union[str, List[str], List[Sample]]) -> None: ...
    def get_sample_ids(self) -> List[str]: ...
    def get_gate_ids(self) -> List[Tuple[str, str]]: ...
    def add_gate(
        self, gate: Gate, gate_path: Tuple[str], sample_id: Optional[str] = None
    ) -> None: ...
    def remove_gate(
        self,
        gate_name: str,
        gate_path: Optional[Tuple[str]] = None,
        keep_children: bool = False,
    ) -> None: ...
    def add_transform(self, transform: Transform) -> None: ...
    def get_transforms(self) -> List[Transform]: ...
    def get_transform(self, transform_id: str) -> Transform: ...
    def add_comp_matrix(self, matrix: Matrix) -> None: ...
    def get_comp_matrices(self) -> List[Matrix]: ...
    def get_comp_matrix(self, matrix_id: str) -> Matrix: ...
    def find_matching_gate_paths(self, gate_name: str) -> List[Tuple[str, str]]: ...
    def get_child_gate_ids(
        self, gate_name: str, gate_path: Optional[Tuple[str]] = None
    ) -> List[Tuple[str, str]]: ...
    def get_gate(
        self,
        gate_name: str,
        gate_path: Optional[Tuple[str]] = None,
        sample_id: Optional[str] = None,
    ) -> Gate: ...
    def get_sample_gates(self, sample_id: str) -> List[Gate]: ...
    def get_gate_hierarchy(
        self, output: Literal["ascii", "dict", "JSON"] = "ascii", **kwargs
    ) -> Union[str, Dict]: ...
    def export_gml(self, file_handle: IO, sample_id: Optional[str] = None) -> None: ...
    def export_wsp(self, file_handle: IO, group_name: str) -> None: ...
    def get_sample(self, sample_id: str) -> Sample: ...
    def analyze_samples(
        self,
        sample_id: Optional[str] = None,
        cache_events: bool = False,
        use_mp: bool = True,
        verbose: bool = False,
    ) -> None: ...
    def get_gating_results(self, sample_id: str) -> GatingResults: ...
    def get_analysis_report(self) -> pd.DataFrame: ...
    def get_gate_membership(
        self, sample_id: str, gate_name: str, gate_path: Optional[Tuple[str]] = None
    ) -> np.ndarray[bool]: ...
    def get_gate_events(
        self,
        sample_id: str,
        gate_name: Optional[str] = None,
        gate_path: Optional[Tuple[str]] = None,
        matrix: Optional[Matrix] = None,
        transform: Optional[Transform] = None,
    ) -> pd.DataFrame: ...
    def plot_gate(
        self,
        sample_id: str,
        gate_name: str,
        gate_path: Optional[Tuple[str]] = None,
        subsample_count: int = 10000,
        random_seed: int = 1,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        color_density: bool = True,
        bin_width: int = 4,
    ) -> figure: ...
    def plot_scatter(
        self,
        sample_id: str,
        x_dim: Dimension,
        y_dim: Dimension,
        gate_name: str,
        gate_path: Optional[Tuple[str]] = None,
        subsample_count: int = 10000,
        random_seed: int = 1,
        color_density: bool = True,
        bin_width: int = 4,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
    ) -> figure: ...
