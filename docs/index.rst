IViS
======

**IViS** — Interferometric Visibility-space Inversion Software

IViS is a fast, GPU-accelerated non-linear deconvolution tool for radio interferometry, scalable to large mosaics and hyperspectral cubes.

Because of its intrinsic framework based on a regularized criterion, IViS — like MEM — is better suited for diffuse emission rather than point sources, whose flux tends to be spread by the regularizer. IViS was developed in the context of imaging H I spectral line data from the ASKAP and MeerKAT instruments.

IViS is designed as a modular framework upon which new deconvolution methods can be built.

.. note::

   In its current implementation, IViS is a generic MEM-like mosaicking framework, but it aims to incorporate arbitrary spectral models in the near future to fully exploit the spectral dimension of the data.
   IViS is very new, and this documentation is in its early stages — it will continue to evolve as the project develops.

Useful resources:
- `Github repository <https://github.com/antoinemarchal/ivis>`_

.. note::
   
   Parts of this code were inspired by the `MPol <https://github.com/MPoL-dev/MPoL>`_ package,
   which implements a Regularized Maximum Likelihood (RML) framework for radio interferometric imaging.
   In contrast to `MPol <https://github.com/MPoL-dev/MPoL>`_, IViS includes native support for image-plane mosaicking with DDEs of the Primary Beam.

Installation
------------

To get started, we recommend using `mamba` to create a clean environment with `casacore`:

.. code-block:: bash

   mamba create -n casacore python=3.10 casacore python-casacore
   mamba activate casacore

Then install the latest development version of **IViS** directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/antoinemarchal/ivis.git


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

   api
