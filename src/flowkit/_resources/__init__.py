"""
Contains non-code resources used in FlowKit
"""

from importlib import resources
from lxml import etree

resource_dir = resources.files("flowkit")
xsd_file = resource_dir / "_resources" / "Gating-ML.v2.0.xsd"

with xsd_file.open("rb") as file:
    gml_tree = etree.parse(file)
gml_schema = etree.XMLSchema(gml_tree)
