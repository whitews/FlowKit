from functools import total_ordering
from typing import IO, TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

if TYPE_CHECKING:

    from pathlib import Path

    import numpy as np
    import pandas as pd
    from bokeh.plotting import figure
    from flowio import FlowData

    from .transforms._base_transform import Transform
    from .transforms._matrix import Matrix

@total_ordering
class Sample(object):
    def __init__(
        self,
        fcs_path_or_data: Union[str, IO, Path, FlowData, np.array, pd.DataFrame],
        sample_id: str = None,
        channel_labels: List[str] = None,
        compensation: Union[Matrix, np.array, str, Path] = None,
        null_channel_list: List[str] = None,
        ignore_offset_error: bool = False,
        ignore_offset_discrepancy: bool = False,
        use_header_offsets: bool = False,
        cache_original_events: bool = False,
        subsample: int = 10000,
    ): ...
    def __repr__(self): ...
    def __lt__(self, other) -> bool: ...
    def __eq__(self, other) -> bool: ...
    def filter_negative_scatter(self, reapply_subsample: bool = True) -> None: ...
    def set_flagged_events(self, event_indices: List): ...
    def get_index_sorted_locations(self) -> List[Tuple[str]]: ...
    def subsample_events(
        self, subsample_count: int = 10000, random_seed: int = 1
    ) -> None: ...
    def apply_compensation(
        self, compensation: Union[Matrix, np.array, str, Path], comp_id="custom_spill"
    ): ...
    def get_metadata(self) -> Dict[str, Any]: ...
    def _get_orig_events(self) -> np.array: ...
    def _get_raw_events(self) -> np.array: ...
    def _get_comp_events(self) -> np.ndarray: ...
    def _get_transformed_events(self) -> np.ndarray: ...
    def get_events(
        self,
        source: Literal["orig", "raw", "comp", "xform"] = "xform",
        subsample: bool = False,
    ) -> np.ndarray: ...
    def as_dataframe(
        self,
        source: Literal["orig", "raw", "comp", "xform"] = "xform",
        subsample: bool = False,
        col_order: List[str] = None,
        col_names: List[str] = None,
    ) -> pd.DataFrame: ...
    def get_channel_number_by_label(self, label: str) -> int: ...
    def get_channel_index(self, channel_label_or_number: Union[str, int]) -> int: ...
    def get_channel_events(
        self,
        channel_index: int,
        source: Literal["orig", "raw", "comp", "xform"] = "xform",
        subsample: bool = False,
    ) -> np.ndarray: ...
    def _transform(
        self,
        transform: Union[Transform, Dict[str, Transform]],
        include_scatter: bool = False,
    ) -> np.ndarray: ...
    def apply_transform(
        self,
        transform: Union[Transform, Dict[str, Transform]],
        include_scatter: bool = False,
    ): ...
    def plot_channel(
        self,
        channel_label_or_number: Union[str, int],
        source: Literal["raw", "comp", "xform"] = "xform",
        subsample: bool = True,
        color_density: bool = True,
        bin_width: int = 4,
        event_mask: Optional[List[bool]] = None,
        highlight_mask: Optional[List[bool]] = None,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
    ) -> figure: ...
    def plot_contour(
        self,
        x_label_or_number: Union[str, int],
        y_label_or_number: Union[str, int],
        source: Literal["raw", "comp", "xform"] = "xform",
        subsample: bool = True,
        plot_events: bool = False,
        fill: bool = False,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
    ): ...
    def plot_scatter(
        self,
        x_label_or_number: Union[str, int],
        y_label_or_number: Union[str, int],
        source: Literal["raw", "comp", "xform"] = "xform",
        subsample: bool = True,
        color_density: bool = True,
        bin_width: int = 4,
        event_mask: Optional[List[bool]] = None,
        highlight_mask: Optional[List[bool]] = None,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
    ): ...
    def plot_scatter_matrix(
        self,
        channel_labels_or_numbers: Optional[List[Union[str, int]]] = None,
        source: Literal["raw", "comp", "xform"] = "xform",
        subsample: bool = True,
        event_mask: Optional[List[bool]] = None,
        highlight_mask: Optional[List[bool]] = None,
        color_density: bool = True,
        plot_height: int = 256,
        plot_width: int = 256,
    ): ...
    def plot_histogram(
        self,
        channel_label_or_number: Union[str, int],
        source: Literal["raw", "comp", "xform"] = "xform",
        subsample: bool = False,
        bins: Optional[Union[int, str]] = None,
        data_min: Optional[float] = None,
        data_max: Optional[float] = None,
        x_range: Optional[Tuple[float, float]] = None,
    ): ...
    def _get_metadata_for_export(self, source, include_all=False) -> Dict[str, str]: ...
    def export(
        self,
        filename: str,
        source: Literal["orig", "raw", "comp", "xform"] = "xform",
        exclude_neg_scatter: bool = False,
        exclude_flagged: bool = False,
        exclude_normal: bool = False,
        subsample: bool = False,
        include_metadata: bool = False,
        directory: str = None,
    ) -> None: ...
