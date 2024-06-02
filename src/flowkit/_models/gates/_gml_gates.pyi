from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .. import gates

if TYPE_CHECKING:
    from lxml.etree import _Element

    from ..gates._gml_gates import _GMLGate
    from ._base_gate import Gate

class _GMLGate(ABC):
    @abstractmethod
    def convert_to_parent_class(self) -> Gate: ...

class GMLRectangleGate(gates.RectangleGate, _GMLGate):
    def __init__(
        self,
        gate_element: _Element,
        gating_namespace: str,
        data_type_namespace: str,
        use_complement: bool = False,
    ): ...
    def convert_to_parent_class(self) -> gates.RectangleGate: ...

class GMLPolygonGate(gates.PolygonGate, _GMLGate):
    def __init__(
        self,
        gate_element: _Element,
        gating_namespace: str,
        data_type_namespace: str,
        use_complement: bool = False,
    ): ...
    def convert_to_parent_class(self) -> gates.PolygonGate: ...

class GMLEllipsoidGate(gates.EllipsoidGate):
    def __init__(
        self,
        gate_element: _Element,
        gating_namespace: str,
        data_type_namespace: str,
    ): ...
    def convert_to_parent_class(self) -> gates.EllipsoidGate: ...

class GMLQuadrantGate(gates.QuadrantGate):
    def __init__(
        self,
        gate_element: _Element,
        gating_namespace: str,
        data_type_namespace: str,
    ): ...
    def convert_to_parent_class(self) -> gates.QuadrantGate: ...

class GMLBooleanGate(gates.BooleanGate):
    def __init__(
        self,
        gate_element: _Element,
        gating_namespace: str,
        data_type_namespace: str,
    ): ...
    def convert_to_parent_class(self) -> gates.BooleanGate: ...
