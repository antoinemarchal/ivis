"""uvmin / uvmax are cuts on the projected (u, v) baseline length in meters, not on the 3D length sqrt(u^2 + v^2 + w^2)."""
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from ivis.readers.ms_hdf5 import HDF5Reader

# (u, v, w) in meters: uv length 50, 50, 500, 0; 3D length 50, 502.2, 500, 50
UVW = np.array([[30.0, 40.0, 0.0], [30.0, 40.0, 500.0], [300.0, 400.0, 0.0], [0.0, 0.0, 50.0]])


def write_cache(path):
    nchan, nvis = 2, len(UVW)
    with h5py.File(path, "w") as out:
        out.attrs.update(complete=True, keep_autocorr=False, prefer_weight_spectrum=False)
        out.create_dataset("data_I", data=np.ones((nchan, 1, nvis), np.complex64))
        out.create_dataset("flag_I", data=np.zeros((nchan, 1, nvis), bool))
        out.create_dataset("sigma_rows", data=np.full((1, nvis), 0.7071, np.float32))
        for k, name in enumerate(("uu", "vv", "ww")):
            out.create_dataset(name, data=UVW[:, k][None, :].astype(np.float32))
        out.create_dataset("frequency", data=np.linspace(1.4e9, 1.4001e9, nchan))
        out.create_dataset("velocity", data=np.linspace(-1.0, 1.0, nchan))
        out.create_dataset("nvis", data=np.asarray([nvis], np.int32))
        out.create_dataset("centers_radec_rad", data=np.asarray([[1.0, -1.0]]))


class UVCutTests(unittest.TestCase):
    def read(self, backend, **cut):
        with tempfile.TemporaryDirectory() as root:
            write_cache(Path(root) / "beam00.hdf5")
            with HDF5Reader(prefer_weight_spectrum=False, keep_autocorr=False, n_workers=1, io_backend=backend) as reader:
                return reader.read_blocks_I(ms_root=root, **cut)

    def test_the_cut_ignores_w(self):
        for backend in ("auto", "h5py"):
            vis = self.read(backend, uvmin=10.0, uvmax=100.0)
            cut = vis.flag_I[:, 0, :]
            # kept: rows 0 and 1 (uv length 50, although row 1 has a 3D length of 502 m); cut: row 2 (uv 500 m) and row 3 (uv length 0 < uvmin)
            np.testing.assert_array_equal(cut, np.broadcast_to([False, False, True, True], cut.shape), err_msg=backend)
            self.assertTrue(np.isinf(vis.sigma_I[:, 0, 2:]).all())
            self.assertTrue(np.isfinite(vis.sigma_I[:, 0, :2]).all())

    def test_no_cut_keeps_everything(self):
        vis = self.read("auto")
        self.assertFalse(vis.flag_I[:, 0, :].any())


if __name__ == "__main__":
    unittest.main()
