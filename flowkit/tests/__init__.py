import unittest

from .sample_tests import SampleTestCase
from .gatingml_tests import GatingMLTestCase
from .prog_gating_tests import GatingTestCase
from .export_gml_tests import ExportGMLTestCase
from .matrix_tests import MatrixTestCase
from .transform_tests import TransformsTestCase
from .string_repr_tests import StringReprTestCase
from .session_tests import SessionTestCase
from .flowjo_wsp_tests import FlowJoWSPTestCase
from .plot_tests import PlotTestCase

if __name__ == "__main__":
    unittest.main()
