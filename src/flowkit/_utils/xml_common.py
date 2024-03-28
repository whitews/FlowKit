"""
Common XML utility functions used by other modules
"""

from lxml import etree
from .._resources import gml_schema


def _get_xml_type(xml_file_or_path):
    xml_document = etree.parse(xml_file_or_path)

    val = gml_schema.validate(xml_document)
    root = xml_document.getroot()

    if val:
        doc_type = 'gatingml'
    else:
        # Try parsing as a FlowJo workspace
        if 'flowJoVersion' in root.attrib:
            if int(root.attrib['flowJoVersion'].split('.')[0]) >= 10:
                doc_type = 'flowjo'
            else:
                raise ValueError("FlowKit only supports FlowJo workspaces for version 10 or higher.")
        else:
            raise ValueError("File is neither GatingML 2.0 compliant nor a FlowJo workspace.")

    gating_ns = None
    data_type_ns = None
    transform_ns = None

    # find GatingML target namespace in the map
    for ns, url in root.nsmap.items():
        if url == 'http://www.isac-net.org/std/Gating-ML/v2.0/gating':
            gating_ns = ns
        elif url == 'http://www.isac-net.org/std/Gating-ML/v2.0/datatypes':
            data_type_ns = ns
        elif url == 'http://www.isac-net.org/std/Gating-ML/v2.0/transformations':
            transform_ns = ns

    if gating_ns is None:
        raise ValueError("GatingML namespace reference is missing from GatingML file")

    return doc_type, root, gating_ns, data_type_ns, transform_ns


def find_attribute_value(xml_el, namespace, attribute_name):
    """
    Extract the value from an XML element attribute.

    :param xml_el: lxml etree Element
    :param namespace: string for the XML element's namespace prefix
    :param attribute_name: attribute string to retrieve the value from
    :return: value string for the given attribute_name
    """
    attribs = xml_el.xpath(
        '@%s:%s' % (namespace, attribute_name),
        namespaces=xml_el.nsmap,
        smart_strings=False
    )
    attribs_cnt = len(attribs)

    if attribs_cnt > 1:
        raise ValueError(
            "Multiple %s attributes found (line %d)" % (
                attribute_name, xml_el.sourceline
            )
        )
    elif attribs_cnt == 0:
        return None

    # return as pure str to save memory (otherwise it's an _ElementUnicodeResult from lxml)
    return str(attribs[0])
