"""
Defines the public API for FlowKit
"""
from ._models.sample import Sample
# noinspection PyProtectedMember
from ._models.transforms._matrix import Matrix
from ._models import transforms
from ._models import gates
from ._models.gating_strategy import GatingStrategy
from ._models.session import Session
from ._models.dimension import Dimension, RatioDimension, QuadrantDivider
from ._models.vertex import Vertex
from ._utils.xml_utils import parse_gating_xml, export_gatingml
from ._utils.wsp_utils import parse_wsp
from ._utils.sample_utils import load_samples, calculate_compensation_from_beads

from ._version import __version__

__all__ = [
    'Sample',
    'Session',
    'GatingStrategy',
    'Matrix',
    'Dimension',
    'RatioDimension',
    'QuadrantDivider',
    'Vertex',
    'gates',
    'transforms',
    'parse_gating_xml',
    'export_gatingml',
    'parse_wsp',
    'load_samples',
    'calculate_compensation_from_beads'
]
