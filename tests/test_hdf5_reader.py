import numpy as np
import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u

h5py = pytest.importorskip("h5py")

from ivis.readers import HDF5Reader


def _write_cache(path, value, row_offset=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    nchan, nvis = 4, 5
    uvw = np.array([
        [1, 0, 0], [2, 0, 0], [6, 0, 0], [3, 0, 0], [4, 0, 0],
    ], dtype=np.float32)
    data = (np.arange(nchan*nvis).reshape(nchan, nvis) + value).astype(np.complex64)
    flags = np.zeros((nchan, nvis), dtype=bool)
    flags[2, 1] = True
    with h5py.File(path, "w") as cache:
        cache.attrs.update(complete=True, keep_autocorr=False, prefer_weight_spectrum=False)
        cache["frequency"] = np.array([1.4e9, 1.401e9, 1.402e9, 1.403e9])
        cache["velocity"] = np.array([0., 1., 2., 3.])
        cache["uvw"] = uvw + np.array([0, 0, row_offset], dtype=np.float32)
        cache["nvis"] = np.array([nvis], dtype=np.int32)
        cache["sigma_rows"] = np.full(nvis, 2., dtype=np.float32)
        cache["center_radec_rad"] = np.array([1., -0.5], dtype=np.float64)
        cache["data_I"] = data
        cache["flag_I"] = flags
    return data


def test_hdf5_reader_loads_channel_range_and_applies_uv_cut(tmp_path):
    source = tmp_path / "82863" / "beam00.hdf5"
    full_data = _write_cache(source, 0)

    vis = HDF5Reader(prefer_weight_spectrum=False).read_blocks_I(
        str(tmp_path), chan_sel=slice(1, 4), uvmin=2, uvmax=4, mode="merge"
    )

    np.testing.assert_array_equal(vis.data_I[:, 0], full_data[1:4])
    np.testing.assert_array_equal(vis.sigma_I[:, 0, :], np.array([
        [np.inf, 2, np.inf, 2, 2], [np.inf, 2, np.inf, 2, 2], [np.inf, 2, np.inf, 2, 2]
    ], dtype=np.float32))
    assert vis.flag_I[:, 0, 2].all()
    assert vis.flag_I[1, 0, 1]
    assert vis.nvis.tolist() == [5]
    assert vis.centers[0].separation(SkyCoord(ra=1*u.rad, dec=-0.5*u.rad)).to_value(u.rad) < 1e-12


def test_hdf5_reader_parallelizes_beams_and_merges_blocks(tmp_path):
    first0 = _write_cache(tmp_path / "block0" / "beam00.hdf5", 0)
    first1 = _write_cache(tmp_path / "block0" / "beam01.hdf5", 100)
    second0 = _write_cache(tmp_path / "block1" / "beam00.hdf5", 200)
    second1 = _write_cache(tmp_path / "block1" / "beam01.hdf5", 300)

    vis = HDF5Reader(prefer_weight_spectrum=False, n_workers=2).read_blocks_I(
        str(tmp_path), chan_sel=slice(0, 2), mode="merge"
    )

    assert vis.nvis.tolist() == [10, 10]
    np.testing.assert_array_equal(vis.data_I[:, 0, :5], first0[:2])
    np.testing.assert_array_equal(vis.data_I[:, 0, 5:], second0[:2])
    np.testing.assert_array_equal(vis.data_I[:, 1, :5], first1[:2])
    np.testing.assert_array_equal(vis.data_I[:, 1, 5:], second1[:2])

    reader = HDF5Reader(prefer_weight_spectrum=False)
    stacked = reader.read_blocks_I(str(tmp_path), chan_sel=slice(0, 1), mode="stack")
    assert stacked.nvis.tolist() == [5, 5, 5, 5]
    separate = reader.read_blocks_I(str(tmp_path), chan_sel=slice(0, 1), mode="separate")
    assert len(separate) == 2


def test_hdf5_reader_accepts_benchmark_single_file_layout(tmp_path):
    path = tmp_path / "single.h5"
    values = _write_cache(path, 17)
    # The benchmark wrote a beam axis into the channel-major arrays.
    with h5py.File(path, "a") as cache:
        del cache["data_I"]
        del cache["flag_I"]
        cache["data_I"] = values[:, None, :]
        cache["flag_I"] = np.zeros((4, 1, 5), dtype=bool)
        del cache["center_radec_rad"]
        cache["centers_radec_rad"] = np.array([[1., -0.5]])
        del cache["uvw"]
        for name, col in zip(("uu", "vv", "ww"), range(3)):
            cache[name] = np.array([[1, 2, 6, 3, 4]], dtype=np.float32) if col == 0 else np.zeros((1, 5), dtype=np.float32)
        del cache["sigma_rows"]
        cache["sigma_rows"] = np.full((1, 5), 2., dtype=np.float32)

    vis = HDF5Reader(prefer_weight_spectrum=False).read_block_I(str(path), chan_sel=slice(0, 2))
    np.testing.assert_array_equal(vis.data_I[:, 0], values[:2])


@pytest.mark.parametrize('backend', ['auto', 'h5py'])
@pytest.mark.parametrize('cube,chunked', [(False, False), (True, False), (False, True), (True, True)])
def test_direct_reader_irregular_channels_padding_and_spectral_sigma(tmp_path, backend, cube, chunked):
    """Exercise strided merged destinations and the HDF5 fallback on real files."""
    values = {}
    sigmas = {}
    for block in range(2):
        for beam in range(2):
            path = tmp_path / f'block{block}' / f'beam{beam}.h5'
            full = _write_cache(path, 100*block+10*beam)
            nv = 5-block-beam
            values[block, beam] = full[:, :nv]
            sigmas[block, beam] = np.arange(4*nv, dtype=np.float32).reshape(4, nv)+1
            with h5py.File(path, 'a') as cache:
                for name in ('uvw', 'sigma_rows'):
                    value = cache[name][:][:nv]
                    del cache[name]
                    cache[name] = value
                cache['nvis'][...] = nv
                for name in ('data_I', 'flag_I'):
                    value = cache[name][:, :nv]
                    del cache[name]
                    if cube:
                        value = value[:, None, :]
                    cache.create_dataset(name, data=value, chunks=True if chunked else None,
                                         compression='gzip' if chunked else None)
                sigma = sigmas[block, beam]
                cache['sigma_I'] = sigma[:, None, :] if cube else sigma
                del cache['sigma_rows']
    selected = [3, 0, 3, 2]
    with HDF5Reader(prefer_weight_spectrum=False, n_workers=2, io_backend=backend) as reader:
        vis = reader.read_blocks_I(str(tmp_path), chan_sel=selected)
        assert vis.nvis.tolist() == [9, 7]
        for beam in range(2):
            expected = np.concatenate([values[b, beam][selected] for b in range(2)], axis=1)
            sigma = np.concatenate([sigmas[b, beam][selected] for b in range(2)], axis=1)
            nv = int(vis.nvis[beam])
            np.testing.assert_array_equal(vis.data_I[:, beam, :nv], expected)
            np.testing.assert_array_equal(vis.sigma_I[:, beam, :nv], sigma)
            assert vis.flag_I[:, beam, nv:].all()
            assert np.all(vis.data_I[:, beam, nv:] == 0)
            assert np.all(vis.sigma_I[:, beam, nv:] == 0)
        saved = vis.data_I.copy()
        reader.read_blocks_I(str(tmp_path), chan_sel=[1], n_workers=1, beam_sel=[1])
        np.testing.assert_array_equal(vis.data_I, saved)
        skipped = reader.read_blocks_I(str(tmp_path), target_center=SkyCoord(0*u.deg, 0*u.deg),
                                       target_radius=0.01*u.deg)
        assert skipped.nvis.tolist() == [0, 0]
        assert skipped.data_I.shape == (4, 2, 0)
    # Returned arrays must remain valid after closing the reusable reader.
    np.testing.assert_array_equal(vis.data_I, saved)


def test_direct_reader_refreshes_replaced_cache(tmp_path):
    path = tmp_path / 'beam.h5'
    _write_cache(path, 0)
    with HDF5Reader(prefer_weight_spectrum=False) as reader:
        first = reader.read_blocks_I(str(tmp_path), chan_sel=[0])
        replacement = tmp_path / 'replacement.tmp'
        _write_cache(replacement, 100)
        replacement.replace(path)
        second = reader.read_blocks_I(str(tmp_path), chan_sel=[0])
        np.testing.assert_array_equal(second.data_I, first.data_I+100)
        with h5py.File(path, 'a') as cache:
            cache.attrs['complete'] = False
        with pytest.raises(ValueError, match='Incomplete'):
            reader.read_blocks_I(str(tmp_path))


def test_direct_reader_handles_partial_reads_and_eof(tmp_path, monkeypatch):
    import os
    from ivis.readers.ms_hdf5 import _pread_into
    if not hasattr(os, 'preadv'):
        pytest.skip('Positional vector reads unavailable on this platform')
    path = tmp_path / 'bytes'
    path.write_bytes(b'0123456789')
    original = os.preadv
    def partial(fd, buffers, offset):
        return original(fd, [buffers[0][:3]], offset)
    monkeypatch.setattr(os, 'preadv', partial)
    target = np.empty(10, dtype=np.uint8)
    with path.open('rb') as stream:
        _pread_into(stream.fileno(), target, 0)
        assert target.tobytes() == b'0123456789'
        with pytest.raises(EOFError):
            _pread_into(stream.fileno(), target, 1)


def test_direct_reader_converts_non_native_dataset_dtype(tmp_path):
    path = tmp_path / 'beam.h5'
    values = _write_cache(path, 0)
    with h5py.File(path, 'a') as cache:
        del cache['data_I']
        cache['data_I'] = values.astype('>c8')
    with HDF5Reader(prefer_weight_spectrum=False, n_workers=2) as reader:
        vis = reader.read_blocks_I(str(tmp_path), chan_sel=[2, 0])
    np.testing.assert_array_equal(vis.data_I[:, 0], values[[2, 0]])
    assert vis.data_I.dtype == np.complex64
