from typing import IO, TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from ..sample import Sample

class Matrix(object):
    def __init__(
        self,
        matrix_id: str,
        spill_data_or_file: Union[str, Path, IO, np.ndarray],
        detectors: List[str],
        fluorochromes: Optional[List[str]] = None,
        null_channels: Optional[List[str]] = None,
    ): ...
    def __repr__(self) -> str: ...
    def apply(self, sample: Sample) -> np.ndarray: ...
    def inverse(self, sample: Sample) -> np.ndarray: ...
    def as_dataframe(self, fluoro_labels: bool = False) -> pd.DataFrame: ...
