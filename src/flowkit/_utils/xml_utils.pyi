from typing import IO, TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

if TYPE_CHECKING:

    from pathlib import Path

    from lxml.etree import _Element

    from .._models import transforms
    from .._models.dimension import Dimension, QuadrantDivider
    from .._models.gates._gml_gates import _GMLGate
    from .._models.gating_strategy import GatingStrategy
    from .._models.transforms._matrix import Matrix

# map GatingML gate keys to our GML gate classes
gate_constructor_lut: Dict[str, _GMLGate] = ...

def parse_gating_xml(xml_file_or_path: Union[str, Path, IO]) -> GatingStrategy: ...
def _construct_gates(
    root_gml: _Element, gating_ns: str, data_type_ns: Optional[str]
) -> Dict[str, _GMLGate]: ...
def _construct_transforms(
    root_gml: _Element, transform_ns: Optional[str], data_type_ns: Optional[str]
) -> Dict[str, transforms.Transform]: ...
def _construct_matrices(
    root_gml: _Element, transform_ns: Optional[str], data_type_ns: Optional[str]
) -> Dict[str, Matrix]: ...
def parse_gate_element(
    gate_element: _Element, gating_namespace: str, data_type_namespace: str
) -> Tuple[str, str, List[Dimension]]: ...
def _parse_dimension_element(
    dim_element: _Element, gating_namespace: str, data_type_namespace: str
) -> Dimension: ...
def _parse_divider_element(
    divider_element: _Element, gating_namespace: str, data_type_namespace: str
) -> QuadrantDivider: ...
def parse_vertex_element(
    vertex_element: _Element, gating_namespace: str, data_type_namespace: str
) -> List[float]: ...
def _parse_matrix_element(
    matrix_element: _Element, xform_namespace: str, data_type_namespace: str
) -> Matrix: ...
def _parse_fratio_element(
    transform_id: str,
    fratio_element: _Element,
    transform_namespace: str,
    data_type_namespace: str,
) -> transforms.RatioTransform: ...
def _parse_flog_element(
    transform_id: str, flog_element: _Element, transform_namespace: str
) -> transforms.LogTransform: ...
def _parse_fasinh_element(
    transform_id: str, fasinh_element: _Element, transform_namespace: str
) -> transforms.AsinhTransform: ...
def _parse_hyperlog_element(
    transform_id: str, hyperlog_element: _Element, transform_namespace: str
) -> transforms.HyperlogTransform: ...
def _parse_flin_element(
    transform_id: str, flin_element: _Element, transform_namespace: str
) -> transforms.LinearTransform: ...
def _parse_logicle_element(
    transform_id: str, logicle_element: _Element, transform_namespace: str
) -> transforms.LogicleTransform: ...
def _parse_transformation_element(
    transformation_element: _Element, transform_namespace: str, data_type_namespace: str
) -> transforms.Transform: ...
