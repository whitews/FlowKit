from setuptools import setup

setup(
    name='FlowKit',
    version='0.0.6',
    packages=['flowkit', 'flowkit.models'],
    package_data={'flowkit': ['resources/*.xsd']},
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
