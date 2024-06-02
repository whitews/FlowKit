from typing import TYPE_CHECKING, Any, Dict, List, Tuple

if TYPE_CHECKING:
    from .._models.gating_results import GatingResults
    from .._models.gating_strategy import GatingStrategy
    from .._models.sample import Sample

def _gate_sample(data: Tuple[GatingStrategy, Sample, bool, bool]) -> GatingResults: ...
def _estimate_cpu_count_for_workload(
    sample_count: int, total_event_count: int
) -> int: ...
def gate_samples(
    sample_data: List[Dict[str, Any]],
    cache_events: bool,
    verbose: bool,
    use_mp: bool = False,
) -> List[GatingResults]: ...
