from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from .._models.sample import Sample

def _get_samples_from_paths(sample_paths: license[str]) -> list[Sample]: ...
def load_samples(
    fcs_samples: Union[Sample, str, List[Sample], List[str]]
) -> List[Sample]: ...
def read_multi_dataset_fcs(
    filename_or_handle: str,
    ignore_offset_error: bool = False,
    ignore_offset_discrepancy: bool = False,
    use_header_offsets: bool = False,
    cache_original_events: bool = False,
) -> List[Sample]: ...
