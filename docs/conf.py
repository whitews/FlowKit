"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import sys
from unittest.mock import MagicMock

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

on_rtd = os.environ.get('READTHEDOCS') == 'True'

if on_rtd:
    sys.path.insert(0, os.path.abspath('..'))
else:
    sys.path.insert(0, os.path.abspath('../..'))


class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


# mock the C extension
MOCK_MODULES = ['flowkit._utils_c']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# -- Project information -----------------------------------------------------

project = 'FlowKit'
copyright = '2020, Scott White'
author = 'Scott White'


# -- General configuration ---------------------------------------------------

master_doc = 'index'
autodoc_member_order = 'bysource'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'autoclasstoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary'
]

autodoc_default_options = {
    'members': True,
    'private-members': False,
    'inherited-members': True,
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

autoclasstoc_sections = [
        'public-methods'
]

# use 'both' to show both Class and __init__ docstrings
autoclass_content = 'both'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

html_theme_options = {
    'logo': 'flowkit.png',
    'github_user': 'whitews',
    'github_repo': 'flowkit',
    'github_banner': True
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'css/custom.css',
]
