"""
FlowKit Test Suites
"""
import unittest

from tests.sample_tests import SampleTestCase
from tests.sample_export_tests import SampleExportTestCase
from tests.gatingml_tests import GatingMLTestCase
from tests.gating_strategy_prog_gate_tests import GatingTestCase
from tests.export_gml_tests import ExportGMLTestCase
from tests.gating_strategy_tests import GatingStrategyTestCase
from tests.gating_strategy_reused_gates_tests import GatingStrategyReusedGatesTestCase
from tests.gating_strategy_modify_gate_tree_tests import GatingStrategyRemoveGatesTestCase
from tests.gating_strategy_custom_gates_tests import GatingStrategyCustomGatesTestCase
from tests.session_tests import SessionTestCase
from tests.session_export_tests import SessionExportTestCase
from tests.workspace_tests import WorkspaceTestCase
from tests.gating_results_tests import GatingResultsTestCase
from tests.matrix_tests import MatrixTestCase
from tests.transform_tests import TransformsTestCase
from tests.gate_tests import GateTestCase
from tests.string_repr_tests import StringReprTestCase
from tests.plot_tests import PlotTestCase

if __name__ == "__main__":
    unittest.main()
