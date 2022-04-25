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

FlowKit is an intuitive Python toolkit for flow cytometry analysis and visualization, with full support for the GatingML 2.0 standard and limited support for FlowJo 10 workspace files.

`FlowKit source <https://github.com/whitews/FlowKit>`_

----

* :ref:`genindex`

.. toctree::
   :maxdepth: 2

   api

Features
--------

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
    * Create a compensation matrix from a set of compensation bead files
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
    * Limited support for importing FlowJo 10 workspace files. Workspace files are currently limited to the following features:
        * Linear, logarithmic, bi-exponential, and logicle transforms
        * Polygon, rectangle, ellipse, and quadrant gates
        * Export a Session's sample group analysis as a FlowJo 10 workspace file
    * Programmatically create gating strategies including polygon, rectangle, range, ellipsoid, quadrant, and boolean gates
    * Retrieve gating results as a Pandas DataFrame
* Visualization
    * Histogram of single channel data
    * Contour density plot of two channels
    * Interactive scatter plot of two channels
    * Interactive scatter plot matrix of any combination of channels
    * Interactive scatter plots of gates with sample events
