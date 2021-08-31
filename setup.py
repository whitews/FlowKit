from setuptools import dist, setup, find_packages, Extension

dist.Distribution().fetch_build_eggs(['numpy>=1.19'])

import numpy as np

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
    'anytree'
]

setup(
    name='FlowKit',
    version='0.7.0',
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
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.6'
    ]
)
