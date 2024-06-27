from typing import List, Literal, Optional, Union

class Dimension(object):
    def __init__(
        self,
        dimension_id: str,
        compensation_ref: Union[str, Literal["uncompensated"]] = "uncompensated",
        transformation_ref: Optional[str] = None,
        range_min: Optional[float] = None,
        range_max: Optional[float] = None,
    ): ...
    def __repr__(self) -> str: ...

class RatioDimension(object):
    def __init__(
        self,
        ratio_ref: str,
        compensation_ref: str,
        transformation_ref: Optional[str] = None,
        range_min: Optional[float] = None,
        range_max: Optional[float] = None,
    ): ...
    def __repr__(self) -> str: ...

class QuadrantDivider(object):
    def __init__(
        self,
        divider_id: str,
        dimension_ref: str,
        compensation_ref: str,
        values: Union[float, List[float]],
        transformation_ref: Optional[str] = None,
    ): ...
    def __repr__(self) -> str: ...
