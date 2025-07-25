# IViS - Interferometric Visibility-space Inversion Software

[![Documentation Status](https://readthedocs.org/projects/ivis-dev/badge/?version=latest)](https://ivis-dev.readthedocs.io/en/latest/)

**IViS** is a fast, GPU-accelerated non-linear deconvolution tool for radio interferometry, scalable to large mosaics and hyperspectral cubes.

> **Note**  
> In its current implementation, IViS is a generic MEM-like mosaicking framework, but it aims to incorporate arbitrary spectral models in the near future to fully exploit the spectral dimension of the data.

## Useful Resources

[Documentation](https://HIJE.readthedocs.io)

> **Acknowledgment**  
> Parts of this code were inspired by the [MPol](https://github.com/MPoL-dev/MPoL) package, which implements a Regularized Maximum Likelihood (RML) framework for radio interferometric imaging without mosaicking.

## Installation

We recommend using [`mamba`](https://mamba.readthedocs.io) to create a clean environment with `casacore`:

```bash
mamba create -n casacore python=3.10 casacore python-casacore
mamba activate casacore
```
```bash
pip install git+https://github.com/antoinemarchal/ivis.git
```
