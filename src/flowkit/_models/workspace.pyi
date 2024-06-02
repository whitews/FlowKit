from typing import IO, TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

if TYPE_CHECKING:
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from bokeh.plotting import figure

    from .gates._base_gate import Gate
    from .gating_results import GatingResults
    from .gating_strategy import GatingStrategy
    from .sample import Sample
    from .transforms._base_transform import Transform
    from .transforms._matrix import Matrix

class Workspace(object):
    def __init__(
        self,
        wsp_file_path: Union[str, Path, IO],
        fcs_samples: Optional[Union[str, List[str], List[Sample]]] = None,
        ignore_missing_files: bool = False,
        find_fcs_files_from_wsp: bool = False,
    ): ...
    def __repr__(self) -> str: ...
    def summary(self) -> pd.DataFrame: ...
    def get_sample_ids(
        self, group_name: Optional[str] = None, loaded_only: bool = True
    ) -> List[str]: ...
    def get_sample(self, sample_id: str) -> Sample: ...
    def get_samples(self, group_name: Optional[str] = None) -> List[Sample]: ...
    def get_sample_groups(self) -> List[str]: ...
    def get_gate_ids(self, sample_id: str) -> List[str]: ...
    def find_matching_gate_paths(
        self, sample_id: str, gate_name: str
    ) -> List[Tuple[str]]: ...
    def get_child_gate_ids(
        self, sample_id: str, gate_name: str, gate_path: Optional[Tuple[str]] = None
    ) -> List[str]: ...
    def get_gate_hierarchy(
        self, sample_id, output: Literal["ascii", "dict", "JSON"] = "ascii", **kwargs
    ) -> Union[str, Dict[str, Any]]: ...
    def get_gating_strategy(self, sample_id: str) -> GatingStrategy: ...
    def get_comp_matrix(self, sample_id: str) -> Matrix: ...
    def get_transform(self, sample_id: str, transform_id: str) -> Transform: ...
    def get_transforms(self, sample_id: str) -> List[Transform]: ...
    def get_gate(
        self, sample_id: str, gate_name: str, gate_path: Tuple[str] = None
    ) -> Gate: ...
    def analyze_samples(
        self,
        group_name: str = None,
        sample_id: str = None,
        cache_events: bool = False,
        use_mp: bool = True,
        verbose: bool = False,
    ) -> None: ...
    def get_gating_results(self, sample_id: str) -> GatingResults: ...
    def get_analysis_report(self, group_name: Optional[str] = None) -> pd.DataFrame: ...
    def _get_processed_events(self, sample_id: str) -> pd.DataFrame: ...
    def get_gate_membership(
        self, sample_id: str, gate_name: str, gate_path: Optional[Tuple[str]] = None
    ) -> np.ndarray[bool]: ...
    def get_gate_events(
        self,
        sample_id: str,
        gate_name: Optional[str] = None,
        gate_path: Optional[Tuple[str]] = None,
    ) -> pd.DataFrame: ...
    def plot_gate(
        self,
        sample_id: str,
        gate_name: Optional[str] = None,
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
        x_label: str,
        y_label: str,
        gate_name: Optional[str] = None,
        gate_path: Optional[Tuple[str]] = None,
        subsample_count: int = 10000,
        random_seed: int = 1,
        color_density: bool = True,
        bin_width: int = 4,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
    ): ...
