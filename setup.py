"""
Setup script for the FlowKit package
"""
from setuptools import setup, find_packages

# read in version string
VERSION_FILE = 'src/flowkit/_version.py'
__version__ = None  # to avoid inspection warning and check if __version__ was loaded
exec(open(VERSION_FILE).read())

if __version__ is None:
    raise RuntimeError("__version__ string not found in file %s" % VERSION_FILE)

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

reqs = [
    'anytree>=2.12',
    'bokeh>=3.4',
    'contourpy>=1.2.0',
    'flowio>=1.4.0b0,<1.5',
    'flowutils>=1.2.0,<1.3',
    'lxml>=5.3',
    'networkx>=3.2.1',
    'numpy>2',
    'pandas>=2.1',
    'psutil>=6',
    'scipy>=1.13'
]

setup(
    name='FlowKit',
    version=__version__,  # noqa PyTypeChecker
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    package_data={'': ['_resources/*.xsd']},
    include_package_data=True,
    description='Flow Cytometry Toolkit',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Scott White",
    license='BSD',
    url="https://github.com/whitews/flowkit",
    ext_modules=[],
    install_requires=reqs,
    classifiers=[
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.9'
    ]
)
