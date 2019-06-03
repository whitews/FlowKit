import os
from glob import glob
import pandas as pd
from flowkit import Sample, GatingStrategy

try:
    import multiprocessing as mp
    multi_proc = True
except ImportError as e:
    mp = None
    multi_proc = False


def get_samples_from_paths(sample_paths):
    if multi_proc:
        if len(sample_paths) < mp.cpu_count():
            proc_count = len(sample_paths)
        else:
            proc_count = mp.cpu_count() - 1  # leave a CPU free just to be nice

        pool = mp.Pool(processes=proc_count)
        samples = pool.map(Sample, sample_paths)
    else:
        samples = []
        for path in sample_paths:
            samples.append(Sample(path))

    return samples


def gate_sample(data):
    gating_strategy = data[0]
    sample = data[1]
    verbose = data[2]
    return gating_strategy.gate_sample(sample, verbose=verbose)


def gate_samples(gating_strategy, samples, verbose):
    if multi_proc:
        if len(samples) < mp.cpu_count():
            proc_count = len(samples)
        else:
            proc_count = mp.cpu_count() - 1  # leave a CPU free just to be nice

        pool = mp.Pool(processes=proc_count)
        data = [(gating_strategy, sample, verbose) for sample in samples]
        all_results = pool.map(gate_sample, data)
    else:
        all_results = []
        for sample in samples:
            results = gating_strategy.gate_sample(sample, verbose=verbose)
            all_results.append(results)

    return all_results


class Session(object):
    def __init__(self, fcs_samples=None, gating_strategy=None):
        self.samples = []
        self.report = None
        self._results = None

        if isinstance(fcs_samples, list):
            # 'fcs_samples' is a list of either file paths or Sample instances
            sample_types = set()

            for sample in fcs_samples:
                sample_types.add(type(sample))

            if len(sample_types) > 1:
                raise ValueError(
                    "Each item in 'fcs_sample' list must be a FCS file path or Sample instance"
                )

            if Sample in sample_types:
                self.samples = fcs_samples
            elif str in sample_types:
                self.samples = get_samples_from_paths(fcs_samples)
        elif isinstance(fcs_samples, Sample):
            # 'fcs_samples' is a Sample instance
            self.samples = [fcs_samples]
        elif isinstance(fcs_samples, str):
            # 'fcs_samples' is a str to either a single FCS file or a directory
            # If directory, search non-recursively for files w/ .fcs extension
            if os.path.isdir(fcs_samples):
                fcs_paths = glob(os.path.join(fcs_samples, '*.fcs'))
                self.samples = get_samples_from_paths(fcs_paths)

        if isinstance(gating_strategy, GatingStrategy):
            self.gating_strategy = gating_strategy
        elif isinstance(gating_strategy, str):
            # assume a path to a GatingML XML file
            self.gating_strategy = GatingStrategy(gating_strategy)
        elif gating_strategy is None:
            self.gating_strategy = GatingStrategy()
        else:
            raise ValueError(
                "'gating_strategy' must be either a GatingStrategy instance or a path to a GatingML document"
            )

    @property
    def gates(self):
        return self.gating_strategy.gates

    def get_sample(self, sample_id):
        for s in self.samples:
            if s.original_filename == sample_id:
                return s

    def analyze_samples(self, verbose=False):
        # Don't save just the DataFrame report, save the entire
        # GatingResults objects for each sample, since we'll need the gate
        # indices for each sample. Add convenience functions inside
        # GatingResults class for conveniently extracting information from
        # the DataFrame
        results = gate_samples(self.gating_strategy, self.samples, verbose)

        all_reports = [res.report for res in results]
        self.report = pd.concat(all_reports)

        self._results = {}
        for r in results:
            self._results[r.sample_id] = r

    def get_gate_indices(self, sample_id, gate_id):
        gating_result = self._results[sample_id]
        return gating_result.get_gate_indices(gate_id)
