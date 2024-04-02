# FlowKit

[!["FlowKit"](https://raw.githubusercontent.com/whitews/FlowKit/master/docs/_static/flowkit.png)](https://github.com/whitews/flowkit)

[![PyPI license](https://img.shields.io/pypi/l/flowkit.svg?colorB=dodgerblue)](https://pypi.python.org/pypi/flowkit/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/flowkit.svg)](https://pypi.python.org/pypi/flowkit/)
[![PyPI version](https://img.shields.io/pypi/v/flowkit.svg?colorB=blue)](https://pypi.python.org/pypi/flowkit/)
[![DOI](https://zenodo.org/badge/138655889.svg)](https://zenodo.org/badge/latestdoi/138655889)


[![Test (master)](https://github.com/whitews/FlowKit/actions/workflows/tests_master.yml/badge.svg)](https://github.com/whitews/FlowKit/actions/workflows/tests_master.yml)
[![Test (develop)](https://github.com/whitews/FlowKit/actions/workflows/tests_develop.yml/badge.svg)](https://github.com/whitews/FlowKit/actions/workflows/tests_develop.yml)
[![codecov](https://codecov.io/gh/whitews/flowkit/branch/master/graph/badge.svg?token=joeOIVWmHi)](https://codecov.io/gh/whitews/flowkit)
[![Documentation Status](https://readthedocs.org/projects/flowkit/badge/?version=latest)](https://flowkit.readthedocs.io/en/latest/?badge=latest)
![PyPI - Downloads](https://img.shields.io/pypi/dm/flowkit)


* [Overview](#overview)
* [Features](#features)
* [Requirements](#requirements)
* [Installation](#installation)
* [Documentation](#documentation)
  * [Tutorials](#tutorials)
  * [Advanced Examples](#advanced-examples)
* [Contributing](#contributing)
* [Cite FlowKit](#cite-flowkit)
* [Projects & Research Using FlowKit](#projects--research-using-flowkit)

## Overview

FlowKit is a Python toolkit for flow cytometry analysis and visualization, with full support for the [GatingML 2.0 standard](http://flowcyt.sourceforge.net/gating/latest.pdf) and limited support for FlowJo 10 workspace files.

<img alt="FlowKit Overview" src="https://raw.githubusercontent.com/whitews/FlowKit/master/docs/_static/flowkit_overview.jpg" style="width:400px;" />

## Features

* Read / Write FCS Files
  * Read FCS files, supporting FCS versions 2.0, 3.0, and 3.1
  * Export FCS data as:
    * A new FCS 3.1 file
    * NumPy array
    * Pandas DataFrame
    * CSV text file
* Compensation
  * Compensate events using spillover matrices from:
    * $SPILL or $SPILLOVER keyword value
    * FlowJo tab-delimited text
    * NumPy array
    * GatingML 2.0 spectrumMatrix XML element
* Transformation
  * Logicle
  * Inverse hyperbolic sine (ArcSinh)
  * FlowJo Bi-exponential
  * Hyperlog
  * Logarithmic
  * Channel ratios
  * Linear
* Gating
  * Full support for the GatingML 2.0 specification
    * Import GatingML XML documents as gating strategies
    * Export gating strategies as a valid GatingML XML document
  * Import FlowJo 10 workspace files, supporting the following features:
    * Linear, logarithmic, bi-exponential, and logicle transforms
    * Polygon, rectangle, ellipse, and quadrant gates
  * Programmatically create gating strategies including polygon, rectangle, range, ellipsoid, quadrant, and boolean gates
  * Retrieve gating results as a Pandas DataFrame 
* Visualization
  * Histogram of single channel data
  * Contour density plot of two channels
  * Interactive scatter plot of two channels
  * Interactive scatter plot matrix of any combination of channels
  * Interactive scatter plots of gates with sample events

## Requirements

FlowKit supports Python version 3.8 - 3.12. All dependencies are installable 
via pip, and are listed below.

> **NOTE**: FlowUtils uses C extensions for significant performance 
> improvements. For most platforms and Python versions, pre-built 
> binaries of FlowUtils are available in PyPI (i.e. installable via 
> `pip`). If a pre-built binary is not available for your environment, 
> the C extensions must be compiled using the source package.

Required Python dependencies:

* [flowio](https://github.com/whitews/flowio) >= 1.3.0
* [flowutils](https://github.com/whitews/flowutils) >= 1.1.0
* anytree >= 2.9
* bokeh >= 3.1
* contourpy >= 1.1.0
* lxml >= 4.9
* networkx >= 3.1
* numpy >= 1.22, <2
* pandas >= 2.0
* psutils >= 5.8
* scipy >= 1.8

## Installation

> **NOTE**: For macOS users running on Apple Silicon, the version of `pip` may need to be upgraded prior to installing FlowKit in order to install the required dependencies.

### From PyPI

```
pip install flowkit
```

### From source

Clone the repository and ensure `pip` is up-to-date. It is recommended to use `pip` to install requirements as relying on `setuptools` may result in incompatible dependency versions.

```
git clone https://github.com/whitews/flowkit
cd flowkit
pip install --upgrade pip
pip install -r requirements.txt
python setup.py install
```

## Documentation

The FlowKit API documentation is available [on ReadTheDocs here](https://flowkit.readthedocs.io/en/latest/?badge=latest). The tutorial notebooks in the `docs/notebooks` directory are a great place to get started with FlowKit, and are linked below.
If you have any questions about FlowKit, find any bugs, or feel something is missing from the tutorials below [please submit an issue to the GitHub repository here](https://github.com/whitews/FlowKit/issues/new/).

### Changelogs

[Changelogs for versions are available here](https://github.com/whitews/FlowKit/releases)

### Tutorials

The series of Jupyter notebook tutorials can be found in the `docs/notebooks` directory of this repository. Note, the interactive scatterplots do not render on GitHub. The rendered versions are available on our [documentation page here](https://flowkit.readthedocs.io/en/latest/?badge=latest). Or, clone the repo and run them locally to see the fully interactive plots.

* [Part 1 - Sample Class](https://github.com/whitews/FlowKit/blob/master/docs/notebooks/flowkit-tutorial-part01-sample-class.ipynb)
* [Part 2 - transforms Module & Matrix Class](https://github.com/whitews/FlowKit/blob/master/docs/notebooks/flowkit-tutorial-part02-transforms-module-matrix-class.ipynb)
* [Part 3 - GatingStrategy & GatingResults Classes](https://github.com/whitews/FlowKit/blob/master/docs/notebooks/flowkit-tutorial-part03-gating-strategy-and-gating-results-classes.ipynb)
* [Part 4 - gates Module](https://github.com/whitews/FlowKit/blob/master/docs/notebooks/flowkit-tutorial-part04-gates-module.ipynb)
* [Part 5 - Session Class](https://github.com/whitews/FlowKit/blob/master/docs/notebooks/flowkit-tutorial-part05-session-class.ipynb)
* [Part 6 - Workspace Class](https://github.com/whitews/FlowKit/blob/master/docs/notebooks/flowkit-tutorial-part06-workspace-class.ipynb)

### Advanced Examples

Below are more advanced and practical examples for using FlowKit. If you have an example you would like to submit for consideration in this list (preferably with data), please [submit an issue](https://github.com/whitews/FlowKit/issues/new/).

* [Compare mean fluorescence intensity (MFI) in gated populations](https://github.com/whitews/FlowKit/blob/master/docs/notebooks/advanced/compare-mfi-of-gated-events.ipynb)
* [Importing a FlowJo 10 WSP file & replicating analysis in FlowKit](https://github.com/whitews/FlowKit/blob/master/docs/notebooks/advanced/replicate-flowjo-wsp.ipynb)
* [Dimension reduction on gated populations](https://github.com/whitews/FlowKit/blob/master/docs/notebooks/advanced/dimension_reduction_on_gated_populations.ipynb)
* [Comparison between Leiden & Louvain clustering](https://github.com/whitews/FlowKit/blob/master/docs/notebooks/advanced/clustering_comparison_leiden_vs_louvain.ipynb)
* [Saving Flow Analysis Data as AnnData objects for ScanPy](https://github.com/whitews/FlowKit/blob/master/docs/notebooks/advanced/scanpy_creating_and_using_AnnData_objects.ipynb)

## Contributing

Want to get involved in the development of FlowKit? 

[Read our CONTRIBUTING guidelines](https://github.com/whitews/FlowKit/blob/master/CONTRIBUTING.md)

## Cite FlowKit

[White, S., Quinn, J., Enzor, J., Staats, J., Mosier, S. M., Almarode, J., Denny, T. N., Weinhold, K. J., Ferrari, G., & Chan, C. (2021). FlowKit: A Python toolkit for integrated manual and automated cytometry analysis workflows. Frontiers in Immunology, 12. https://doi.org/10.3389/fimmu.2021.768541](https://www.frontiersin.org/articles/10.3389/fimmu.2021.768541/full)

## Projects & Research Using FlowKit 

The following projects and publications have utilized FlowKit. If you have a package or publication where FlowKit was used, and you want it listed here, feel free to [submit an issue](https://github.com/whitews/FlowKit/issues/new/) letting me know.

* Mortelecque, Justine et al. "A selection and optimization strategy for single-domain antibodies targeting the PHF6 linear peptide within the Tau intrinsically disordered protein." Journal of Biological Chemistry (2024), [doi:10.1016/j.jbc.2024.107163](https://www.jbc.org/article/S0021-9258(24)01658-2/fulltext)
* Schmidt, Florian et al. "In-depth analysis of human virus-specific CD8+ T cells delineates unique phenotypic signatures for T cell specificity prediction." Cell Reports vol. 42,10 (2023), [doi:10.1016/j.celrep.2023.113250](https://www.cell.com/cell-reports/fulltext/S2211-1247(23)01262-7)
* Schuster, Jonas et al. “Combining real-time fluorescence spectroscopy and flow cytometry to reveal new insights in DOC and cell characterization of drinking water.” Frontiers in Environmental Chemistry (2022), [doi:10.3389/fenvc.2022.931067](https://www.frontiersin.org/articles/10.3389/fenvc.2022.931067/full)
* Rendeiro, André F et al. "Profiling of immune dysfunction in COVID-19 patients allows early prediction of disease progression." Life science alliance vol. 4,2 e202000955. 24 Dec. 2020, [doi:10.26508/lsa.202000955](https://www.life-science-alliance.org/content/4/2/e202000955.full)
