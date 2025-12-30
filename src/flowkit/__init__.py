"""
Defines the public API for FlowKit
"""
from ._models.sample import Sample
# noinspection PyProtectedMember
from ._models.transforms._matrix import Matrix, SpectralMatrix
from ._models import transforms
from ._models import gates
from ._models.gating_strategy import GatingStrategy
from ._models.session import Session
from ._models.workspace import Workspace
from ._models.dimension import Dimension, RatioDimension, QuadrantDivider
from ._utils.xml_utils import parse_gating_xml
from ._utils.gml_write import export_gatingml
from ._utils.wsp_utils import parse_wsp, extract_wsp_sample_data
from ._utils.sample_utils import load_samples, read_multi_dataset_fcs, extract_fcs_metadata
from ._utils.transform_utils import generate_transforms
from . import exceptions

from ._version import __version__

__all__ = [
    'Sample',
    'Session',
    'Workspace',
    'GatingStrategy',
    'Matrix',
    'Dimension',
    'RatioDimension',
    'QuadrantDivider',
    'gates',
    'transforms',
    'parse_gating_xml',
    'export_gatingml',
    'parse_wsp',
    'extract_wsp_sample_data',
    'extract_fcs_metadata',
    'load_samples',
    'read_multi_dataset_fcs',
    'generate_transforms',
    'exceptions'
]
