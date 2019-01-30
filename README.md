[![Build Status](https://travis-ci.com/whitews/FlowKit.svg?branch=master)](https://travis-ci.com/whitews/FlowKit)

# FlowKit
Intuitive Python framework for flow cytometry analysis and visualization, including GatingML support.

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

See the [notebook tutorial](https://github.com/whitews/FlowKit/blob/master/examples/flowkit-tutorial.ipynb).
