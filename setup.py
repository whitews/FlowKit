from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='FlowKit',
    version='0.0.9',
    packages=find_packages(),
    package_data={'': ['*.xsd']},
    include_package_data=True,
    description='Flow Cytometry Toolkit',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Scott White",
    license='BSD',
    url="https://github.com/whitews/flowkit",
    ext_modules=[],
    install_requires=[
        'flowio',
        'flowutils',
        'matplotlib',
        'scipy',
        'seaborn',
        'pandas',
        'numpy',
        'lxml',
        'bokeh',
        'anytree'
    ]
)
