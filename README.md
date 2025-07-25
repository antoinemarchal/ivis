# HIJE — Hyperspectral Imaging using Joint deconvolution for low-brightness diffuse emission lines

[![Documentation Status](https://readthedocs.org/projects/hije/badge/?version=latest)](https://hije.readthedocs.io/en/latest/)

**HIJE** is a fast, GPU-accelerated non-linear deconvolution tool for radio interferometry, scalable to large mosaics and hyperspectral cubes.

> **Note**  
> In its current implementation, HIJE is a generic MEM-like mosaicking framework, but it aims to incorporate arbitrary spectral models in the near future to fully exploit the spectral dimension of the data.

## Useful Resources

[Documentation]: https://hije.readthedocs.io/en/latest/

> **Acknowledgment**  
> Parts of this code were inspired by the [MPol](https://github.com/MPoL-dev/MPoL) package, which implements a Regularized Maximum Likelihood (RML) framework for radio interferometric imaging without mosaicking.

## Installation

We recommend using [`mamba`](https://mamba.readthedocs.io) to create a clean environment with `casacore`:

```bash
mamba create -n casacore python=3.10 casacore python-casacore
mamba activate casacore
```
```bash
pip install git+https://github.com/antoinemarchal/HIJE.git
```
