import os
from glob import glob
import pandas as pd
from flowkit import Sample, GatingStrategy


class Session(object):
    def __init__(self, fcs_samples=None, gating_strategy=None):
        self.samples = []
        self.report = None

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
                for path in fcs_samples:
                    self.samples.append(
                        Sample(path)
                    )
        elif isinstance(fcs_samples, Sample):
            # 'fcs_samples' is a Sample instance
            self.samples = [fcs_samples]
        elif isinstance(fcs_samples, str):
            # 'fcs_samples' is a str to either a single FCS file or a directory
            # If directory, search non-recursively for files w/ .fcs extension
            if os.path.isdir(fcs_samples):
                fcs_paths = glob(os.path.join(fcs_samples, '*.fcs'))
                for fcs_path in fcs_paths:
                    self.samples.append(Sample(fcs_path))

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

    def analyze_samples(self, verbose=False):
        reports = []

        for sample in self.samples:
            results = self.gating_strategy.gate_sample(sample, verbose=verbose)
            reports.append(results.report)

        self.report = pd.concat(reports)
