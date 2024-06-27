from typing import IO, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from lxml import etree
    from pathlib import Path


def _get_xml_type(
    xml_file_or_path: Union[str, Path, IO]
) -> Tuple[str, etree._Element, str, Optional[str], Optional[str]]: ...


def find_attribute_value(
    xml_el: etree._Element, namespace: str, attribute_name: str
) -> str: ...
