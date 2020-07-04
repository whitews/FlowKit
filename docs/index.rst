.. FlowKit documentation master file, created by
   sphinx-quickstart on Fri Mar 20 17:54:48 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FlowKit Documentation
=====================

.. image:: https://img.shields.io/pypi/v/flowkit.svg?colorB=dodgerblue
    :target: https://pypi.org/project/flowkit/

.. image:: https://img.shields.io/pypi/l/flowkit.svg?colorB=green
    :target: https://pypi.python.org/pypi/flowkit/

.. image:: https://img.shields.io/pypi/pyversions/flowkit.svg
    :target: https://pypi.org/project/flowkit/

FlowKit is an intuitive Python toolkit for flow cytometry analysis and visualization, with full support for the
GatingML 2.0 standard and limited support for FlowJo 10 workspace files.

----

* :ref:`genindex`

.. toctree::
   :maxdepth: 2

   api

Features
--------

* Read FCS files, including versions 2.0, 3.0, and 3.1
* Export FCS data as a new FCS 3.1 file, NumPy array, Pandas DataFrame, or CSV
* Compensation of FCS events
* Automatically create compensation matrix from compensation bead files
* Tranform FCS events in a variety of transforms used in the flow community (including logicle)
* Full support for the GatingML 2.0 specification
* Limited support for importing FlowJo 10 workspace files. Workspace files are currently limited to the following features:
  * Linear, logarithmic, and logicle transforms
  * Polygon and rectangle gates

* Programmatically create gating strategies including polygon, rectangle, range, ellipsoid, quadrant, and boolean gates
* Easily retrieve gating results from a gating strategy as a Pandas DataFrame.
* Optional, automatic filtering of negative scatter events and/or anomalous events
* Visualize FCS data as histograms, contour plots, and interactive scatter plots

