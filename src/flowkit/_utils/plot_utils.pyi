from typing import TYPE_CHECKING, Callable, List, Literal, Optional, Tuple, Union

if TYPE_CHECKING:
    import numpy as np
    from bokeh.models import ColumnDataSource, Ellipse, Patch, Rect, Renderer, Span
    from bokeh.plotting import figure
    from contourpy import ContourGenerator

    from .._models.gating_strategy import GatingStrategy

LINE_COLOR_DEFAULT: str = "#1F77B4"
LINE_COLOR_CONTRAST: str = "#73D587"
LINE_WIDTH_DEFAULT: int = 3
FILL_COLOR_DEFAULT: str = "lime"
FILL_ALPHA_DEFAULT: float = 0.08

custom_heat_palette: List[str] = ...

def _get_false_bounds(
    bool_array: np.ndarray[bool],
) -> Tuple[np.ndarray, np.ndarray]: ...
def _calculate_extent(
    data_1d: np.ndarray,
    d_min: Optional[float] = None,
    d_max: Optional[float] = None,
    pad: float = 0.0,
) -> Tuple[float, float]: ...
def _quantiles_to_levels(data: np.ndarray, quantiles: np.ndarray) -> np.ndarray: ...
def _calculate_2d_gaussian_kde(
    x: np.ndarray,
    y: np.ndarray,
    bw_method: Union[Literal["scott", "silverman"], float, Callable] = "scott",
    grid_size: int = 200,
    pad_factor: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def _build_contour_generator(
    mesh_x: np.ndarray, mesh_y: np.ndarray, estimated_pdf: np.ndarray
) -> ContourGenerator: ...
def render_polygon(
    vertices: List[Tuple[float, float]],
    line_color: str = LINE_COLOR_CONTRAST,
    line_width: int = LINE_WIDTH_DEFAULT,
    fill_color: str = FILL_COLOR_DEFAULT,
    fill_alpha: float = FILL_ALPHA_DEFAULT,
) -> Tuple[ColumnDataSource, Patch]: ...
def render_ranges(
    dim_minimums: Tuple[int], dim_maximums: Tuple[int]
) -> Tuple[Span]: ...
def render_rectangle(
    dim_minimums: Tuple[int, int], dim_maximums: Tuple[int, int]
) -> Rect: ...
def render_dividers(x_locs: List[float], y_locs: List[float]) -> List[Renderer]: ...
def render_ellipse(
    center_x: float,
    center_y: float,
    covariance_matrix: np.ndarray,
    distance_square: float,
) -> Ellipse: ...
def plot_histogram(
    x: np.ndarray, x_label: str = "x", bins: Optional[Union[int, str]] = None
) -> figure: ...
def plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    event_mask: Optional[np.ndarray[bool]] = None,
    highlight_mask: Optional[np.ndarray[bool]] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    color_density: bool = True,
    bin_width: int = 4,
) -> figure: ...
def plot_contours(
    x: np.ndarray,
    y: np.ndarray,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    plot_events: bool = False,
    fill: bool = False,
) -> figure: ...
def plot_gate(
    gate_id: Tuple[str],
    gating_strategy: GatingStrategy,
    sample: Sample,
    subsample_count: int = 10000,
    random_seed: int = 1,
    event_mask: Optional[np.ndarray[bool]] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    color_density: bool = True,
    bin_width: int = 4,
) -> figure: ...
