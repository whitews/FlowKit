from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    import numpy as np

    from ._base_transform import Transform

def _log_root(b: float, w: float) -> float: ...
def generate_biex_lut(
    channel_range: int = 4096,
    pos: float = 4.418540,
    neg: float = 0.0,
    width_basis: float = -10,
    max_value: float = 262144.000029,
) -> Tuple[np.ndarray, np.ndarray]: ...

class WSPLogTransform(Transform):
    def __init__(self, transform_id: str, offset: float, decades: float): ...
    def __repr__(self) -> str: ...
    def apply(self, events: np.ndarray) -> np.ndarray: ...

class WSPBiexTransform(Transform):
    def __init__(
        self,
        transform_id: str,
        negative: float = 0,
        width: float = -10,
        positive: float = 4.418540,
        max_value: float = 262144.000029,
    ): ...
    def __repr__(self) -> str: ...
    def apply(self, events: np.ndarray) -> np.ndarray: ...
    def inverse(self, events: np.ndarray) -> np.ndarray: ...
