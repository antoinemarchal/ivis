# IViS - Interferometric Visibility-space Inversion Software

[![Documentation Status](https://readthedocs.org/projects/ivis-dev/badge/?version=latest)](https://ivis-dev.readthedocs.io/en/latest/)

**IViS** is a fast, GPU-accelerated non-linear deconvolution tool for radio interferometry, scalable to large mosaics and hyperspectral cubes.

> **Note**  
> In its current implementation, IViS is a generic MEM-like mosaicking framework, but it aims to incorporate arbitrary spectral models in the near future to fully exploit the spectral dimension of the data.

## Useful Resources

[Documentation](https://ivis-dev.readthedocs.io)

> **Acknowledgment**  
> Parts of this code were inspired by the [MPol](https://github.com/MPoL-dev/MPoL) package, which implements a Regularized Maximum Likelihood (RML) framework for radio interferometric. In contrast to MPol, IViS includes native support for image-plane mosaicking with DDEs of the Primary Beam. 

# Installation

To get started, we recommend using [`uv`](https://github.com/astral-sh/uv) to manage packages and `mamba` to create a clean environment with `casacore`:

```bash
# Install mamba (if not already available)
conda install mamba -n base -c conda-forge

# Create the IViS environment
mamba create -n ivis \
  python=3.10 casacore=3.4.0 python-casacore=3.4.0 gsl=2.6 pip \
  -c conda-forge -c pkgs/main

mamba activate ivis

# Install uv (if not already available)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"  # if needed

# Install IViS and dependencies using uv
uv pip install git+https://github.com/antoinemarchal/ivis.git
```

