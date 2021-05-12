"""
GatingResults class
"""
import pandas as pd


class GatingResults(object):
    """
    A GatingResults instance is returned from the GatingStrategy `gate_samples` method
    as well as the Session `get_gating_results` method. End users will never create an
    instance of GatingResults directly, only via these GatingStrategy and Session
    methods. However, there are several GatingResults methods to retrieve the results.
    """
    def __init__(self, results_dict, sample_id):
        self._raw_results = results_dict
        self._gate_lut = {}
        self.report = None
        self.sample_id = sample_id
        self._process_results()

    @staticmethod
    def _get_pd_result_dict(res_dict, gate_id):
        return {
            'sample': res_dict['sample'],
            'gate_path': res_dict['gate_path'],
            'parent': res_dict['parent'],
            'gate_id': gate_id,
            'gate_type': res_dict['gate_type'],
            'count': res_dict['count'],
            'absolute_percent': res_dict['absolute_percent'],
            'relative_percent': res_dict['relative_percent'],
            'quadrant_parent': None
        }

    def _process_results(self):
        pd_list = []

        for (g_id, g_path), res in self._raw_results.items():
            if 'events' not in res:
                # it's a quad gate with sub-gates
                for sub_g_id, sub_res in res.items():
                    pd_dict = self._get_pd_result_dict(sub_res, sub_g_id)
                    pd_dict['quadrant_parent'] = g_id
                    pd_list.append(pd_dict)
                    if sub_g_id not in self._gate_lut:
                        self._gate_lut[sub_g_id] = {
                            'paths': [g_path]
                        }
                    else:
                        self._gate_lut[sub_g_id]['paths'].append(g_path)
            else:
                pd_list.append(self._get_pd_result_dict(res, g_id))
                if g_id not in self._gate_lut:
                    self._gate_lut[g_id] = {
                        'paths': [g_path]
                    }
                else:
                    self._gate_lut[g_id]['paths'].append(g_path)

        df = pd.DataFrame(
            pd_list,
            columns=[
                'sample',
                'gate_path',
                'gate_id',
                'gate_type',
                'quadrant_parent',
                'parent',
                'count',
                'absolute_percent',
                'relative_percent'
            ]
        )
        df['level'] = df.gate_path.map(len)

        # ???: sorting by non-index column will result in Pandas PerformanceWarning
        #   when looking up rows using .loc. The hit in this case is minimal and we
        #   really want the DataFrame sorted by 'level' for better readability.
        #   Maybe consider not setting a MultiIndex for this?
        self.report = df.set_index(['sample', 'gate_id']).sort_index().sort_values('level')

    def get_gate_indices(self, gate_id, gate_path=None):
        """
        Retrieve a boolean array indicating gate membership for the events in the GatingResults sample.
        Note, the same gate ID may be found in multiple gate paths, i.e. the gate ID can be ambiguous.
        In this case, specify the full gate path to retrieve gate indices.

        :param gate_id: text string of a gate ID
        :param gate_path: A list of ancestor gate IDs for the given gate ID. Alternatively, a string path delimited
            by forward slashes can also be given, e.g. ('/root/singlets/lymph/live')
        :return: NumPy boolean array (length of sample event count)
        """
        gate_paths = self._gate_lut[gate_id]['paths']
        if len(gate_paths) > 1:
            if gate_path is None:
                raise ValueError("Gate ID %s is ambiguous, specify the full gate path")
            elif isinstance(gate_path, list):
                gate_path = "/".join(gate_path)
        else:
            gate_path = gate_paths[0]

        gate_series = self.report.loc[(self.sample_id, gate_id)]
        if isinstance(gate_series, pd.DataFrame):
            gate_series = gate_series.iloc[0]

        quad_parent = gate_series['quadrant_parent']

        if quad_parent is not None:
            return self._raw_results[quad_parent, gate_path][gate_id]['events']
        else:
            return self._raw_results[gate_id, gate_path]['events']

    def get_gate_count(self, gate_id):
        """
        Retrieve event count for the specified gate ID for the gating results sample

        :param gate_id: text string of a gate ID
        :return: integer count of events in gate ID
        """
        return self.report.loc[(self.sample_id, gate_id), 'count']

    def get_gate_absolute_percent(self, gate_id):
        """
        Retrieve percent of events, relative to the total sample events, of the specified gate ID for the
        gating results sample

        :param gate_id: text string of a gate ID
        :return: floating point number of the absolute percent
        """
        return self.report.loc[(self.sample_id, gate_id), 'absolute_percent']

    def get_gate_relative_percent(self, gate_id):
        """
        Retrieve percent of events, relative to parent gate, of the specified gate ID for the gating results sample

        :param gate_id: text string of a gate ID
        :return: floating point number of the relative percent
        """
        return self.report.loc[(self.sample_id, gate_id), 'relative_percent']
