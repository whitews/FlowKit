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
    'anytree>=2.9',
    'bokeh>=3.1',
    'contourpy>=1.1.0',
    'flowio>=1.3.0,<1.4',
    'flowutils>=1.1.0,<1.2',
    'lxml>=4.9',
    'networkx>=3.1',
    'numpy>=1.22,<2',
    'pandas>=2.0',
    'psutil>=5.8',
    'scipy>=1.8'
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
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.8'
    ]
)
