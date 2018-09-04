from setuptools import setup

setup(
    name='FlowKit',
    version='0.0.4',
    packages=['flowkit', 'flowkit.models'],
    package_data={'': []},
    description='Flow Cytometry Toolkit',
    license='BSD',
    ext_modules=[],
    install_requires=[
        'flowio',
        'flowutils',
        'matplotlib',
        'scipy',
        'seaborn',
        'pandas'
    ]
)
