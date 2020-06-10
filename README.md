# <img alt="FlowKit" src="docs/_static/flowkit.png" />

[![PyPI license](https://img.shields.io/pypi/l/flowkit.svg?colorB=dodgerblue)](https://pypi.python.org/pypi/flowkit/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/flowkit.svg)](https://pypi.python.org/pypi/flowkit/)
[![PyPI version](https://img.shields.io/pypi/v/flowkit.svg?colorB=blue)](https://pypi.python.org/pypi/flowkit/)

[![Build Status](https://travis-ci.com/whitews/FlowKit.svg?branch=master)](https://travis-ci.com/whitews/FlowKit)
[![Coverage](https://codecov.io/gh/whitews/FlowKit/branch/develop/graph/badge.svg)](https://codecov.io/gh/whitews/flowkit)
[![Documentation Status](https://readthedocs.org/projects/flowkit/badge/?version=latest)](https://flowkit.readthedocs.io/en/latest/?badge=latest)

* [Overview](#overview)
* [Requirements](#requirements)
* [Installation](#installation)
* [Usage](#usage)
* [Contributing](#contributing)

## Overview

FlowKit is an intuitive Python toolkit for flow cytometry analysis and visualization, with full support for the [GatingML 2.0 standard](http://flowcyt.sourceforge.net/gating/latest.pdf) and limited support for FlowJo 10 workspace files. Features include:

* Reading Flow Cytometry Standard data (FCS files), including FCS versions:
  * 2.0
  * 3.0
  * 3.1
* Exporting FCS data in any of the following formats:
  * A new FCS 3.1 file, with modified metadata and/or filtered events
  * NumPy array
  * Pandas DataFrame
  * CSV text file
* Compensation of FCS events
  * Apply compensation from spillover matrices in multiple formats:
    * As the $SPILL or $SPILLOVER keyword value format
    * FlowJo tab-delimited text format
    * NumPy array
    * GatingML 2.0 spectrumMatrix XML element
  * Automatically create a compensation matrix from a set of compensation bead files
* Transformation of FCS events in a variety of transforms used in the flow community:
  * Logicle
  * Inverse hyperbolic sine (arcsinh)
  * Hyperlog
  * Logarithmic
  * Channel ratios
  * Linear
* Gating
  * Full support for the GatingML 2.0 specification
    * Import GatingML XML documents as gating strategies
    * Export gating strategies as a valid GatingML XML document
  * Limited support for importing FlowJo 10 workspace files. Workspace files are currently limited to the following features:
    * Linear, logarithmic, and logicle transforms
    * Polygon, rectangle, and quadrant gates
  * Programmatically create gating strategies including polygon, rectangle, range, ellipsoid, quadrant, and boolean gates
  * Easily retrieve gating results from a gating strategy as a Pandas DataFrame. Results include:
    * FCS sample ID
    * Gate name
    * Absolute event count
    * Relative percentage
    * Absolute percentage
* Optional, automatic filtering of negative scatter events and/or anomalous events
* Visualizing FCS event data:
  * Histogram of single channel data
  * Contour density plot of two channels
  * Interactive scatter plot of two channels
  * Interactive scatter plot matrix of any combination of channels
  * Interactive scatter plots of gates with sample events

<img alt="Screenshot of scatterplot" src="examples/fk_scatterplot.png" style="width:200px;" />

## Requirements

FlowKit supports Python version 3.6 or above. All dependencies are installable 
via pip, and are listed below.

***Note: FlowKit and FlowUtils use C extensions for significant performance 
improvements relating to various transformations. If using `gcc`, version 5 or 
above is required for correct Logicle and Hyperlog transformations.***

Required Python dependencies:

* [flowio](https://github.com/whitews/flowio) >= 0.9.5
* [flowutils](https://github.com/whitews/flowutils) >= 0.7.1
* numpy >= 1.13
* scipy >= 1.3
* scikit-learn >= 0.21
* statsmodels >= 0.8
* pandas >= 0.22
* matplotlib >= 3.0
* seaborn >= 0.9
* bokeh >= 1.4
* lxml >= 4.2
* anytree >= 2.6
* MulticoreTSNE >= 0.1

## Installation

### From PyPI

FlowKit is available via the `pip` command. However, NumPy must be installed prior in order to
compile the C extensions.

```
pip install numpy
pip install flowkit
```

### From source

```
git clone https://github.com/whitews/flowkit
cd flowkit
python setup.py install
```

## Usage

Click on the links below to a few Jupyter notebooks that demonstrate basic usage of the library. Note, the interactive scatterplots do not render on Github. Clone the repo (or download the example notebooks), and run them locally to see the fully interactive plots.

* [General Overview](https://github.com/whitews/FlowKit/blob/master/examples/flowkit-tutorial.ipynb)
* [Applying Transforms to a Sample](https://github.com/whitews/FlowKit/blob/master/examples/sample_transforms.ipynb)
* [Compensating Spillover in a Sample](https://github.com/whitews/FlowKit/blob/master/examples/sample_compensation.ipynb)
* [Importing a FlowJo 10 WSP file & replicating analysis in FlowKit](https://github.com/whitews/FlowKit/blob/master/examples/flowkit-session-replicate-flowjo-wsp-example.ipynb)

## Contributing

Want to get involved in the development of FlowKit? 

[Read our CONTRIBUTING guidelines](https://github.com/whitews/FlowKit/blob/master/CONTRIBUTING.md)
