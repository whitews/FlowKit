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
    def _get_pd_result_dict(res_dict, gate_name):
        return {
            'sample_id': res_dict['sample'],
            'gate_path': res_dict['gate_path'],
            'parent': res_dict['parent'],
            'gate_name': gate_name,
            'gate_type': res_dict['gate_type'],
            'count': res_dict['count'],
            'absolute_percent': res_dict['absolute_percent'],
            'relative_percent': res_dict['relative_percent'],
            'quadrant_parent': None
        }

    def _process_results(self):
        pd_list = []

        for (g_name, g_path), res in self._raw_results.items():
            if 'events' not in res:
                # it's a quad gate with sub-gates
                for sub_g_id, sub_res in res.items():
                    pd_dict = self._get_pd_result_dict(sub_res, sub_g_id)
                    pd_dict['quadrant_parent'] = g_name
                    pd_list.append(pd_dict)
                    if sub_g_id not in self._gate_lut:
                        self._gate_lut[sub_g_id] = {
                            'paths': [g_path]
                        }
                    else:
                        self._gate_lut[sub_g_id]['paths'].append(g_path)
            else:
                pd_list.append(self._get_pd_result_dict(res, g_name))
                if g_name not in self._gate_lut:
                    self._gate_lut[g_name] = {
                        'paths': [g_path]
                    }
                else:
                    self._gate_lut[g_name]['paths'].append(g_path)

        df = pd.DataFrame(
            pd_list,
            columns=[
                'sample_id',
                'gate_path',
                'gate_name',
                'gate_type',
                'quadrant_parent',
                'parent',
                'count',
                'absolute_percent',
                'relative_percent'
            ]
        )
        df['level'] = df.gate_path.map(len)

        self.report = df.sort_values(['sample_id', 'level', 'gate_name'])

    def _filter_gate_report(self, gate_name, gate_path=None):
        results = self.report[(self.report['sample_id'] == self.sample_id) & (self.report['gate_name'] == gate_name)]

        if gate_path is not None:
            results = results[results.gate_path == gate_path]
        elif len(results) > 1:
            raise ValueError("Gate name %s is ambiguous, specify the full gate path")

        return results

    def get_gate_membership(self, gate_name, gate_path=None):
        """
        Retrieve a boolean array indicating gate membership for the Sample events in the GatingResults.
        Note, the same gate ID may be found in multiple gate paths, i.e. the gate ID can be ambiguous.
        In this case, specify the full gate path to retrieve gate indices.

        :param gate_name: text string of a gate name
        :param gate_path: A tuple of ancestor gate IDs for the given gate ID. Alternatively, a string path delimited
            by forward slashes can also be given, e.g. ('/root/singlets/lymph/live')
        :return: NumPy boolean array (length of sample event count)
        """
        gate_paths = self._gate_lut[gate_name]['paths']
        if len(gate_paths) > 1:
            if gate_path is None:
                raise ValueError("Gate name %s is ambiguous, specify the full gate path")
            elif isinstance(gate_path, tuple):
                gate_path = "/".join(gate_path)
        else:
            gate_path = gate_paths[0]

        # need to check for quadrant gates, as they need to be handled differently
        gate_data = self.report[(self.report['sample_id'] == self.sample_id) & (self.report.gate_name == gate_name)]
        quad_parent_values = set(gate_data.quadrant_parent.to_list())

        if len(quad_parent_values) == 1 and None in quad_parent_values:
            # it's not a quadrant gate
            return self._raw_results[gate_name, gate_path]['events']
        elif len(quad_parent_values) == 1:
            quad_parent = quad_parent_values.pop()
            return self._raw_results[quad_parent, gate_path][gate_name]['events']
        else:
            raise ValueError("Report as bug: The gate %s appears to have multiple quadrant parents." % gate_name)

    def get_gate_count(self, gate_name, gate_path=None):
        """
        Retrieve event count for the specified gate ID for the gating results sample

        :param gate_name: text string of a gate name
        :param gate_path: tuple of ancestor gate names
        :return: integer count of events in gate ID
        """
        results = self._filter_gate_report(gate_name, gate_path=gate_path)

        return results['count'].values[0]

    def get_gate_absolute_percent(self, gate_name, gate_path=None):
        """
        Retrieve percent of events, relative to the total sample events, of the specified gate ID for the
        gating results sample

        :param gate_name: text string of a gate name
        :param gate_path: tuple of ancestor gate names
        :return: floating point number of the absolute percent
        """
        results = self._filter_gate_report(gate_name, gate_path=gate_path)

        return results['absolute_percent'].values[0]

    def get_gate_relative_percent(self, gate_name, gate_path=None):
        """
        Retrieve percent of events, relative to parent gate, of the specified gate ID for the gating results sample

        :param gate_name: text string of a gate name
        :param gate_path: tuple of ancestor gate names
        :return: floating point number of the relative percent
        """
        results = self._filter_gate_report(gate_name, gate_path=gate_path)

        return results['relative_percent'].values[0]
