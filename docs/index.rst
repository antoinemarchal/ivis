deconv
======

**deconv** is a fast, GPU-accelerated non-linear deconvolution tool for radio interferometry, scalable to large mosaics and hyperspectral cubes.

Useful resources:
- `Github repository <https://github.com/antoinemarchal/DECONV>`_
- `Documentation <https://deconv.readthedocs.io>`_

.. note::
   
   Parts of this code were inspired by the `MPol <https://github.com/MPoL-dev/MPoL>`_ package,
   which implements a Regularized Maximum Likelihood (RML) framework for radio interferometric imaging without mosaicking.

Installation
------------

To get started, we recommend using `mamba` to create a clean environment with `casacore`:

.. code-block:: bash

   mamba create -n casacore python=3.10 casacore python-casacore
   mamba activate casacore

Then install the latest development version of **DECONV** directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/antoinemarchal/DECONV.git


Tutorials
---------

.. toctree::
   :maxdepth: 1
   :caption: Basics:

   tutorials/getting_started

API Reference
-------------

.. toctree::
   :maxdepth: 1
   :caption: API:

   deconv
