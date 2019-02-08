# <img alt="FlowKit" src="flowkit/resources/flowkit.png" />

[![PyPI license](https://img.shields.io/pypi/l/flowkit.svg?colorB=dodgerblue)](https://pypi.python.org/pypi/flowkit/)
[![PyPI version](https://img.shields.io/pypi/v/flowkit.svg?colorB=blue)](https://pypi.python.org/pypi/flowkit/)
[![Build Status](https://travis-ci.com/whitews/FlowKit.svg?branch=master)](https://travis-ci.com/whitews/FlowKit)

## Overview

FlowKit is an intuitive Python toolkit for flow cytometry analysis and visualization, including GatingML 2.0 support. Features include:

* Reading Flow Cytometry Standard data (FCS files), including FCS versions:
  * 2.0
  * 3.0
  * 3.1
* Exporting FCS data in any of the following formats:
  * A new FCS 3.1 file, with modified metadata and/or filtered events
  * NumPy array
  * Pandas DataFrame
  * CSV text file
* Compensating FCS events using spillover matrices in multiple formats:
  * As the $SPILL or $SPILLOVER keyword value format
  * FlowJo tab-delimited text format
  * NumPy array
  * GatingML 2.0 spectrumMatrix XML element
* Tranformation of original or compensated events in a variety of transforms used in the flow community:
  * Logicle
  * Inverse hyperbolic sine (arcsinh)
  * Hyperlog
  * Logarithmic
  * Channel ratios
  * Linear
* Optional, automatic filtering of negative scatter events and/or anomalous events
* Visualizing FCS event data:
  * Histogram of single channel data with a Gaussian kernel density estimate curve
  * Contour density plot of two channels
  * Interactive scatter plot of two channels
  * Interactive scatter plot matrix of any combination of channels

<img alt="Screenshot of scatterplot" src="examples/fk_scatterplot.png" style="width:200px;" />

## Requirements

FlowKit supports Python version 3.6 or above. All dependencies are installable 
via pip, and include:

* [flowio](https://github.com/whitews/flowio) >= 0.9.3
* [flowutils](https://github.com/whitews/flowutils) >= 0.6.8
* numpy >= 0.15
* scipy >= 1.0
* pandas >= 0.19
* matplotlib >= 3.0
* seaborn >= 0.9
* bokeh >= 1.0
* lxml >= 4.2
* anytree >= 2.4

## Installation

### From PyPI

`pip install flowkit`

### From source

```
git clone https://github.com/whitews/flowkit
cd flowkit
python setup.py install
```

## Usage

Check out the example notebooks:

* [General Overview](https://github.com/whitews/FlowKit/blob/master/examples/flowkit-tutorial.ipynb)
* [Applying Transforms to a Sample](https://github.com/whitews/FlowKit/blob/master/examples/sample_transforms.ipynb)