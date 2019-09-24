from abc import ABC, abstractmethod
import numpy as np
from flowkit import _utils, Matrix
from flowkit._models import gates, dimension


class Gate(ABC):
    """
    Represents a single flow cytometry gate
    """
    def __init__(
            self,
            gate_id,
            parent_id,
            dimensions,
            gating_strategy
    ):
        self.__parent__ = gating_strategy
        self.id = gate_id
        self.parent = parent_id
        self.dimensions = dimensions
        self.gate_type = None

    def apply_parent_gate(self, sample, results, parent_results):
        if self.parent is not None:
            parent_gate = self.__parent__.get_gate_by_reference(self.parent)
            parent_id = parent_gate.id

            if parent_results is not None:
                results_and_parent = np.logical_and(parent_results['events'], results)
                parent_count = parent_results['count']
            else:
                parent_result = parent_gate.apply(sample, parent_results)

                if isinstance(parent_gate, gates.QuadrantGate):
                    parent_result = parent_result[self.parent]

                parent_count = parent_result['count']
                results_and_parent = np.logical_and(parent_result['events'], results)
        else:
            # no parent, so results are unchanged & parent count is total count
            parent_id = None
            parent_count = sample.event_count
            results_and_parent = results

        event_count = results_and_parent.sum()

        # check parent_count to avoid div by zero
        if parent_count == 0:
            relative_percent = 0.0
        else:
            relative_percent = (event_count / float(parent_count)) * 100.0
        final_results = {
            'sample': sample.original_filename,
            'events': results_and_parent,
            'count': event_count,
            'absolute_percent': (event_count / float(sample.event_count)) * 100.0,
            'relative_percent': relative_percent,
            'parent': parent_id,
            'gate_type': self.gate_type
        }

        return final_results

    @abstractmethod
    def apply(self, sample, parent_results):
        pass

    def compensate_sample(self, dim_comp_refs, sample):
        dim_comp_ref_count = len(dim_comp_refs)

        if dim_comp_ref_count == 0:
            events = sample.get_raw_events()
            return events.copy()
        elif dim_comp_ref_count > 1:
            raise NotImplementedError(
                "Mixed compensation between individual channels is not "
                "implemented. Never seen it, but if you are reading this "
                "message, submit an issue to have it implemented."
            )
        else:
            comp_ref = list(dim_comp_refs)[0]

        events = self.__parent__.get_cached_compensation(
            sample,
            comp_ref
        )

        if events is not None:
            return events

        if comp_ref == 'FCS':
            meta = sample.get_metadata()

            if 'spill' not in meta or 'spillover' not in meta:
                # GML 2.0 spec states if 'FCS' is specified but no spill is present, treat as uncompensated
                events = sample.get_raw_events()
                return events.copy()

            try:
                spill = meta['spillover']  # preferred, per FCS standard
            except KeyError:
                spill = meta['spill']

            spill = _utils.parse_compensation_matrix(
                spill,
                sample.pnn_labels,
                null_channels=sample.null_channels
            )
            indices = spill[0, :]  # headers are channel #'s
            indices = [int(i - 1) for i in indices]
            detectors = [sample.pnn_labels[i] for i in indices]
            fluorochromes = [sample.pns_labels[i] for i in indices]
            matrix = Matrix('fcs', fluorochromes, detectors, spill[1:, :])
        else:
            # lookup specified comp-ref in parent strategy
            matrix = self.__parent__.comp_matrices[comp_ref]

        if matrix is not None:
            events = matrix.apply(sample)
            # cache the comp events
            self.__parent__.cache_compensated_events(
                sample,
                comp_ref,
                events
            )

        return events

    def preprocess_sample_events(self, sample):
        pnn_labels = sample.pnn_labels
        pns_labels = sample.pns_labels

        dim_idx = []
        dim_min = []
        dim_max = []
        dim_comp_refs = set()
        new_dims = []
        dim_xform = []

        for dim in self.dimensions:
            dim_comp = False
            if dim.compensation_ref not in [None, 'uncompensated']:
                dim_comp_refs.add(dim.compensation_ref)
                dim_comp = True

            if isinstance(dim, dimension.RatioDimension):
                # dimension is a transform of other dimensions
                new_dims.append(dim)
                continue
            elif isinstance(dim, dimension.QuadrantDivider):
                dim_label = dim.dimension_ref
                dim_min.append(None)
                dim_max.append(None)
            else:
                dim_label = dim.label
                dim_min.append(dim.min)
                dim_max.append(dim.max)

            if dim_label in pnn_labels:
                dim_idx.append(pnn_labels.index(dim_label))
            elif dim.label in pns_labels:
                dim_idx.append(pns_labels.index(dim_label))
            else:
                # for a referenced comp, the label may have been the
                # fluorochrome instead of the channel's PnN label. If so,
                # the referenced matrix object will also have the detector
                # names that will match
                if not dim_comp:
                    raise LookupError(
                        "%s is not found as a channel label or channel reference in %s" % (dim_label, sample)
                    )
                matrix = self.__parent__.comp_matrices[dim.compensation_ref]
                try:
                    matrix_dim_idx = matrix.fluorochomes.index(dim_label)
                except ValueError:
                    raise ValueError("%s not found in list of matrix fluorochromes" % dim_label)
                detector = matrix.detectors[matrix_dim_idx]
                dim_idx.append(pnn_labels.index(detector))

            dim_xform.append(dim.transformation_ref)

        events = self.compensate_sample(dim_comp_refs, sample)

        for i, dim in enumerate(dim_idx):
            if dim_xform[i] is not None:
                xform = self.__parent__.transformations[dim_xform[i]]
                events[:, [dim]] = xform.apply(events[:, [dim]])

        return events, dim_idx, dim_min, dim_max, new_dims
