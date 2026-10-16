# IViS - Interferometric Visibility-domain Inversion Software

<pre>
        _ _| \ \     / _)   ___| 
          |   \ \   /   | \___ \ 
          |    \ \ /    |       |
        ___|    \_/    _| _____/ 
</pre>

[![Documentation Status](https://readthedocs.org/projects/ivis-dev/badge/?version=latest)](https://ivis-dev.readthedocs.io/en/latest/)
[![Tests](https://github.com/antoinemarchal/ivis/actions/workflows/tests.yml/badge.svg)](https://github.com/antoinemarchal/ivis/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![License](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)

**IViS** is a fast, GPU-accelerated non-linear deconvolution tool for radio interferometry, scalable to large mosaics and hyperspectral cubes.

> **Note**  
> In its current implementation, IViS is a generic MEM-like mosaicking framework, but it aims to incorporate arbitrary spectral models in the near future to fully exploit the spectral dimension of the data.

## Useful Resources

[Documentation](https://ivis-dev.readthedocs.io)

> **Acknowledgment**  
> Parts of this code were inspired by the [MPol](https://github.com/MPoL-dev/MPoL) package, which implements a Regularized Maximum Likelihood (RML) framework for radio interferometry. In contrast to MPol, IViS includes native support for image-plane mosaicking with DDEs of the Primary Beam.

# Installation

To get started, we recommend using [`uv`](https://github.com/astral-sh/uv) to manage packages and `mamba` to create a clean environment with `casacore`:

```bash
# Install mamba (if not already available)
conda install mamba -n base -c conda-forge

# Create the IViS environment
mamba create -n ivis -y -c conda-forge \
  python=3.10 casacore python-casacore gsl h5py \
  numpy scipy astropy matplotlib \
  reproject radio-beam psutil tqdm joblib \
  pip

conda activate ivis

# Torch CUDA wheel
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch

# Install uv (if not already available)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"  # if needed

# Install IViS and dependencies using uv
uv pip install "ivis[gpu,hdf5] @ git+https://github.com/antoinemarchal/ivis.git"
```

## Reading HDF5 visibility caches

HDF5 inputs follow the MS directory tree: one `.h5` or `.hdf5` file per beam,
inside each observation-block directory. Conversion is handled outside IViS.

```python
from ivis.readers import HDF5Reader

with HDF5Reader(prefer_weight_spectrum=False, n_workers=12) as reader:
    vis = reader.read_blocks_I(
        "/path/to/hdf5_root", chan_sel=slice(0, 20), mode="merge",
        uvmin=0, uvmax=3000,
    )
```

The weight and autocorrelation settings must match those used for conversion.
The reader preserves beam order, block merging, channel selections, flags, and
UV cuts. Small channel batches work with contiguous HDF5 storage: each complete
channel is already contiguous on disk.

Workers share the final output arrays and read directly into each beam/block's
assigned region. This avoids process transfers and intermediate block copies.
The reader reuses metadata and worker threads between batches; returned arrays
remain valid after `close()`. Use one caller per reader instance.

For compatible contiguous datasets, `io_backend="auto"` uses ordinary buffered
positional reads at offsets obtained from HDF5. Other layouts, including chunked
or compressed datasets, use h5py. `io_backend="h5py"` forces that fallback.
Caches must remain unchanged while being read. Both paths return the same
materialized `VisIData` arrays, so choose a channel batch that fits available RAM.
