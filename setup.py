"""
Setup script for the FlowKit package
"""
from setuptools import dist, setup, find_packages, Extension

# read in version string
VERSION_FILE = 'flowkit/_version.py'
__version__ = None  # to avoid inspection warning and check if __version__ was loaded
exec(open(VERSION_FILE).read())

if __version__ is None:
    raise RuntimeError("__version__ string not found in file %s" % VERSION_FILE)

# NumPy is needed to build
# This retrieves a version at build time compatible with run time version
dist.Distribution().fetch_build_eggs(['numpy>=1.19'])

# override inspection for import not at top of file
# this has to be imported here, after fetching the NumPy egg
import numpy as np  # noqa: E402

with open("README.md", "r") as fh:
    long_description = fh.read()

utils_extension = Extension(
    'flowkit._utils_c',
    sources=[
        'flowkit/utils_c_ext/_utils.c',
        'flowkit/utils_c_ext/utils.c'
    ],
    include_dirs=[np.get_include(), 'flowkit/utils_c_ext'],
    extra_compile_args=['-std=c99']
)

reqs = [
    'flowio',
    'flowutils',
    'matplotlib',
    'scipy',
    'statsmodels',
    'seaborn',
    'pandas',
    'numpy',
    'lxml',
    'bokeh',
    'anytree',
    'networkx',
    'psutil'
]

setup(
    name='FlowKit',
    version=__version__,
    packages=find_packages(exclude=["flowkit/tests/"]),
    package_data={'': ['_resources/*.xsd']},
    include_package_data=True,
    description='Flow Cytometry Toolkit',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Scott White",
    license='BSD',
    url="https://github.com/whitews/flowkit",
    ext_modules=[utils_extension],
    install_requires=reqs,
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.6'
    ]
)
