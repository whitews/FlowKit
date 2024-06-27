from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import pandas as pd
    from lxml.etree import _Element

    from .._models.gates._base_gate import Gate
    from .._models.gating_strategy import GatingStrategy
    from .._models.sample import Sample
    from .._models.transforms._base_transform import Transform
    from .._models.transforms._matrix import Matrix

wsp_gate_constructor_lut: Dict[str, Gate] = ...

def _uri_to_path(uri: str, wsp_file_path: str) -> str: ...
def _parse_wsp_compensation(
    sample_el: _Element, transform_ns: str, data_type_ns: str
) -> Dict[str, Any]: ...
def _parse_wsp_transforms(
    transforms_el: _Element, transform_ns: str, data_type_ns: str
) -> Dict[str, Any]: ...
def _parse_wsp_keywords(keywords_el: _Element) -> Dict[str, Any]: ...
def _convert_wsp_gate(
    wsp_gate: Gate, comp_matrix: Optional[pd.DataFrame], xform_lut: Dict
) -> Gate: ...
def _recurse_wsp_sub_populations(
    sub_pop_el: _Element,
    gate_path: Optional[Tuple[str]],
    gating_ns: str,
    data_type_ns: str,
) -> Dict[str, Gate]: ...
def _parse_wsp_groups(
    group_node_els: _Element, ns_map: Dict, gating_ns: str, data_type_ns: str
) -> Dict[str, Any]: ...
def _parse_wsp_samples(
    sample_els: _Element,
    ns_map: Dict,
    gating_ns: str,
    transform_ns: str,
    data_type_ns: str,
) -> Dict[str, Any]: ...
def parse_wsp(workspace_file_or_path: str) -> Dict: ...
def extract_wsp_sample_data(workspace_file_or_path: str) -> Dict: ...
def _add_matrix_to_wsp(
    parent_el: _Element, prefix: str, matrix: Matrix, ns_map: Dict
) -> None: ...
def _add_transform_to_wsp(
    parent_el: _Element, parameter_label: str, transform: Transform, ns_map: Dict
) -> None: ...
def _add_sample_keywords_to_wsp(parent_el: _Element, sample: Sample) -> None: ...
def _add_polygon_gate(
    parent_el: _Element,
    gate: Gate,
    fj_gate_id: str,
    fj_parent_gate_id: str,
    gating_strategy: GatingStrategy,
    comp_prefix: str,
    ns_map: Dict,
) -> None: ...
def _add_rectangle_gate(
    parent_el: _Element,
    gate: Gate,
    fj_gate_id: str,
    fj_parent_gate_id: str,
    gating_strategy: GatingStrategy,
    comp_prefix: str,
    ns_map: Dict,
) -> None: ...
def _add_group_node_to_wsp(
    parent_el: _Element, group_name: str, sample_id_list: List[str]
) -> None: ...
def _recurse_add_sub_populations(
    parent_el: _Element,
    gate_id: str,
    gate_path: Tuple[str],
    gating_strategy: GatingStrategy,
    gate_fj_id_lut: Dict,
    comp_prefix: str,
    ns_map: Dict,
    sample_id: Optional[str] = None,
) -> None: ...
def _add_sample_node_to_wsp(
    parent_el: _Element,
    sample_name: str,
    sample_id: str,
    gating_strategy: GatingStrategy,
    comp_prefix_lut: Dict,
    ns_map: Dict,
) -> None: ...
def export_flowjo_wsp(
    gating_strategy: GatingStrategy,
    group_name: str,
    samples: List[Sample],
    file_handle: str,
) -> None: ...
