from setuptools import setup, find_packages

setup(
    name='FlowKit',
    version='0.0.9b',
    packages=find_packages(),
    package_data={'': ['*.xsd']},
    include_package_data=True,
    description='Flow Cytometry Toolkit',
    license='BSD',
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
        'bokeh'
    ]
)
