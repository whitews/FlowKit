"""
Utility functions for processing gating strategies on samples.
"""
import gc
import math
import psutil
from .._conf import multi_proc, mp, mp_context


# _gate_sample & _gate_samples are multi-proc wrappers for GatingStrategy _gate_sample method
# These are functions external to GatingStrategy as mp doesn't work well for class methods
def _gate_sample(data):
    gating_strategy = data[0]
    sample = data[1]
    cache_events = data[2]
    verbose = data[3]

    gating_results = gating_strategy.gate_sample(sample, cache_events=cache_events, verbose=verbose)

    gc.collect()

    return gating_results


def _estimate_cpu_count_for_workload(sample_count, total_event_count):
    # gather system resource info
    vm = psutil.virtual_memory()
    mem_available = vm.available
    proc_count = mp.cpu_count() - 1  # always start by leaving 1 cpu free to be nice

    # But workload is just the total number of values in the array for all samples.
    # Each value is 64-bit double, so multiply by 8 bytes for total byte size
    # AND we'll multiply by 5 because processing will almost certainly require
    # at least an additional copy for:
    #   +1 for preprocessed comp events
    #   +1 for preprocessed xform events
    #   +1 for carrying over parent populations
    #   +1 for space for the boolean results and other machinery.
    workload_in_bytes = total_event_count * 8 * 5

    # Now, determine how many samples we can run while staying at or below 80% mem usage.
    # We'll assume equal distribution of events per sample...not the most accurate way,
    # but it's easy
    workload_per_sample = workload_in_bytes / sample_count

    max_concurrent_samples = math.floor((mem_available * .8) / workload_per_sample)

    if max_concurrent_samples <= 2:
        # best we can do is run each sample separate
        proc_count = 1
    elif proc_count > max_concurrent_samples:
        # and leave one cpu free just for good measure
        proc_count = max_concurrent_samples - 1

    return proc_count


def gate_samples(sample_data, cache_events, verbose, use_mp=False):
    """
    Process gating strategies for multiple Sample instances. Attempts to use
    multiprocessing to optimize analysis.

    :param sample_data: list of dictionaries where keys are 'gating_strategy' & 'sample'
    :param cache_events: whether to cache processed event data in each GatingStrategy
    :param verbose: enables printing of progress status
    :param use_mp: enables multiprocessing
    :return: List of GatingResults instances
    """
    # NOTE: Multiprocessing can fail for very large workloads (lots of gates) due
    #       to running out of memory. For those cases setting use_mp should be set
    #       to False in the Session.analyze_samples() method

    # sample_data is a list of dictionaries, where each contains keys: 'gating_strategy' & 'sample'
    sample_count = len(sample_data)

    # get total number of data values for all samples
    event_dim_info = [(sd['sample'].event_count, len(sd['sample'].pnn_labels)) for sd in sample_data]
    total_data_size = sum([event_count * dim_count for event_count, dim_count in event_dim_info])

    proc_count = _estimate_cpu_count_for_workload(sample_count, total_data_size)

    if multi_proc and sample_count > 1 and use_mp and proc_count > 1:
        if sample_count < proc_count:
            proc_count = sample_count

        with mp.get_context(mp_context).Pool(processes=proc_count, maxtasksperchild=1) as pool:
            if verbose:
                # flush the output here so the print statement doesn't get buried in the
                # individual gate print output
                print(
                    '#### Processing gates for %d samples (multiprocessing is enabled - %d cpus) ####'
                    % (sample_count, proc_count),
                    flush=True
                )
            data = [(sd['gating_strategy'], sd['sample'], cache_events, verbose) for sd in sample_data]

            async_results = [pool.apply_async(_gate_sample, args=(d,)) for d in data]
            # all_results = pool.map_async(_gate_sample, data).get()

            pool.close()
            pool.join()

            all_results = [result.get() for result in async_results]
    else:
        if verbose:
            # flush the output here so the print statement doesn't get buried in the
            # individual gate print output
            print(
                '#### Processing gates for %d samples (multiprocessing is disabled) ####' % sample_count,
                flush=True
            )

        all_results = []
        for sd in sample_data:
            results = sd['gating_strategy'].gate_sample(sd['sample'], verbose=verbose)
            all_results.append(results)

    return all_results
