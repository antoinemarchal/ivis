import os

import numpy as np
from astropy.io import fits

from ivis.io import DataProcessor


def processor_for(path):
    processor = object.__new__(DataProcessor)
    processor.pathout = str(path)
    return processor


def write_products(path, pb, grid):
    fits.writeto(path / "pb.fits", pb, overwrite=True)
    fits.writeto(path / "grid.fits", grid, overwrite=True)


def test_read_uses_native_endian_mapped_sidecars(tmp_path):
    pb = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    grid = np.arange(48, dtype=np.float32).reshape(2, 1, 3, 4, 2)
    write_products(tmp_path, pb, grid)
    DataProcessor._write_native_float32_cache(str(tmp_path / "pb.fits"), pb)
    DataProcessor._write_native_float32_cache(str(tmp_path / "grid.fits"), grid)

    loaded_pb, loaded_grid = processor_for(tmp_path).read_pb_and_grid(
        "pb.fits", "grid.fits")

    assert isinstance(loaded_pb, np.memmap)
    assert isinstance(loaded_grid, np.memmap)
    assert loaded_pb.dtype == loaded_grid.dtype == np.dtype("float32")
    assert loaded_pb.dtype.isnative and loaded_grid.dtype.isnative
    np.testing.assert_array_equal(loaded_pb, pb)
    np.testing.assert_array_equal(loaded_grid, grid)


def test_stale_sidecar_falls_back_to_fits(tmp_path):
    pb = np.ones((1, 2, 3), dtype=np.float32)
    grid = np.zeros((1, 1, 2, 3, 2), dtype=np.float32)
    write_products(tmp_path, pb, grid)
    DataProcessor._write_native_float32_cache(str(tmp_path / "pb.fits"), pb)
    DataProcessor._write_native_float32_cache(str(tmp_path / "grid.fits"), grid)

    # Make one FITS file newer than its sidecar to model external replacement.
    pb_path = tmp_path / "pb.fits"
    future_ns = max(pb_path.stat().st_mtime_ns,
                    (tmp_path / "pb.npy").stat().st_mtime_ns) + 1_000_000_000
    os.utime(pb_path, ns=(future_ns, future_ns))
    updated = np.full_like(pb, 2)
    fits.writeto(pb_path, updated, overwrite=True)

    loaded_pb, _ = processor_for(tmp_path).read_pb_and_grid("pb.fits", "grid.fits")
    assert not isinstance(loaded_pb, np.memmap)
    np.testing.assert_array_equal(loaded_pb, updated)
