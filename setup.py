from setuptools import setup, find_packages, Extension
import numpy as np

with open("README.md", "r") as fh:
    long_description = fh.read()

utils_extension = Extension(
    'flowkit.utils_c',
    sources=[
        'flowkit/utils_c_ext/_utils.c',
        'flowkit/utils_c_ext/utils.c'
    ],
    include_dirs=[np.get_include(), 'flowkit/utils_c_ext'],
    extra_compile_args=['-std=c99']
)

setup(
    name='FlowKit',
    version='0.3.1',
    packages=find_packages(),
    package_data={'': ['*.xsd']},
    include_package_data=True,
    description='Flow Cytometry Toolkit',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Scott White",
    license='BSD',
    url="https://github.com/whitews/flowkit",
    ext_modules=[utils_extension],
    install_requires=[
        'flowio',
        'flowutils',
        'matplotlib',
        'multicoretsne',
        'scipy',
        'sklearn',
        'statsmodels',
        'seaborn',
        'pandas',
        'numpy',
        'lxml',
        'bokeh',
        'anytree'
    ]
)
