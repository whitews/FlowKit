import unittest

from .sample_unit_tests import LoadSampleTestCase
from .gating_unit_tests import GatingMLTestCase
from .prog_gating_unit_tests import GatingTestCase
from .export_gml_unit_tests import ExportGMLTestCase
from .matrix_unit_tests import MatrixTestCase
from .transform_unit_tests import TransformsTestCase
from .string_repr_tests import StringReprTestCase
from .session_unit_tests import SessionTestCase
from .plot_unit_tests import PlotTestCase

if __name__ == "__main__":
    unittest.main()
