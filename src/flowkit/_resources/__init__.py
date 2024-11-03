"""
Contains non-code resources used in FlowKit
"""
from pkg_resources import resource_filename
from lxml import etree
import os
import pathlib

resource_path = resource_filename('flowkit', '_resources')
# force POSIX-style path, even on Windows
resource_path = pathlib.Path(resource_path).as_posix()

# We used to edit the import paths for Transformations & DataTypes,
# but it still caused an XMLSchemaParseError when FlowKit was
# installed in a location where the path contained "special"
# characters (i.e. accented letters). Instead, we now change
# directories temporarily to the XSD location, read in the files,
# and then change back to the original CWD.
orig_dir = os.getcwd()
os.chdir(resource_path)
gml_tree = etree.parse('Gating-ML.v2.0.xsd')
gml_schema = etree.XMLSchema(gml_tree)
os.chdir(orig_dir)
