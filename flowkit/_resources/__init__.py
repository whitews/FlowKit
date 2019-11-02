from pkg_resources import resource_stream, resource_filename
from lxml import etree
import pathlib

resource_path = resource_filename('flowkit', '_resources')
# force POSIX-style path, even on Windows
resource_path = pathlib.Path(resource_path).as_posix()
gml_tree = etree.parse(
    resource_stream('flowkit._resources', 'Gating-ML.v2.0.xsd')
)

gml_root = gml_tree.getroot()
gml_imports = gml_root.findall('import', gml_root.nsmap)

for i in gml_imports:
    schema_loc = i.get('schemaLocation')
    i.set('schemaLocation', "/".join([resource_path, schema_loc]))

gml_schema = etree.XMLSchema(
    gml_tree
)
