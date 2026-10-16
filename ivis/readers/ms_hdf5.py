"""HDF5 visibility reader with parallel reads into the final VisIData arrays.

One HDF5 cache replaces each MS in the original block/beam directory tree.
Contiguous datasets use buffered positional reads at offsets obtained from HDF5.
Chunked, compressed, external, and virtual datasets use the HDF5 library instead.
No storage conversion is performed here. Source caches must remain immutable
while a read is in progress.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
import os
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from ivis.logger import logger
from ivis.types import VisIData
from .ms_casacore import _centers_equal, _normalize_beam_sel, _within_radius, _norm_radius


def _h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("HDF5 visibility caches require h5py; install ivis[hdf5].") from exc
    return h5py


def _list_h5_sorted(directory):
    path = Path(directory)
    if path.is_file() and path.suffix.lower() in ('.h5', '.hdf5'):
        return [str(path)]
    files = sorted(str(p) for p in path.iterdir() if p.is_file() and p.suffix.lower() in ('.h5', '.hdf5'))
    if not files:
        raise FileNotFoundError(f'No HDF5 caches in {directory}')
    return files


def _list_h5_block_dirs(root):
    path = Path(root)
    if path.is_file() and path.suffix.lower() in ('.h5', '.hdf5'):
        return [str(path)]
    blocks = []
    def contains(directory):
        return any(p.is_file() and p.suffix.lower() in ('.h5', '.hdf5') for p in directory.iterdir())
    if contains(path):
        blocks.append(str(path))
    blocks.extend(str(p) for p in sorted(path.iterdir()) if p.is_dir() and contains(p))
    if not blocks:
        raise FileNotFoundError(f'No HDF5 caches in {root} or its immediate subdirectories')
    return blocks


def _signature(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _metadata(path, cache):
    signature = _signature(os.stat(path))
    cached = cache.get(path)
    if cached is not None and cached['signature'] == signature:
        return cached
    with _h5py().File(path, 'r') as source:
        if not source.attrs.get('complete', False):
            raise ValueError(f'Incomplete HDF5 cache: {path}')
        names = ['data_I', 'flag_I']
        names += ['sigma_I'] if 'sigma_I' in source else ['sigma_rows']
        names += ['uvw'] if 'uvw' in source else ['uu', 'vv', 'ww']
        datasets = {}
        for name in names:
            dataset = source[name]
            offset = dataset.id.get_offset()
            raw = (source.driver == 'sec2' and dataset.chunks is None and
                   not dataset.is_virtual and not dataset.external and
                   offset is not None and not dataset.dtype.hasobject)
            datasets[name] = dict(shape=dataset.shape, dtype=dataset.dtype,
                                  offset=int(offset) if raw else None)
        shape = datasets['data_I']['shape']
        if len(shape) not in (2, 3) or (len(shape) == 3 and shape[1] != 1):
            raise ValueError(f'Expected one beam per HDF5 cache: {path}')
        if datasets['flag_I']['shape'] != shape:
            raise ValueError(f'Flag and visibility shapes differ: {path}')
        frequency = np.asarray(source['frequency'][:], dtype=np.float64)
        velocity = np.asarray(source['velocity'][:], dtype=np.float64)
        if frequency.shape != (shape[0],) or velocity.shape != frequency.shape:
            raise ValueError(f'Invalid spectral axes: {path}')
        nvis = shape[-1]
        if 'nvis' in source and int(np.asarray(source['nvis'][...]).reshape(-1)[0]) != nvis:
            raise ValueError(f'Visibility count differs from dataset shape: {path}')
        center_key = 'center_radec_rad' if 'center_radec_rad' in source else 'centers_radec_rad'
        center = np.asarray(source[center_key][...]).reshape(-1,2)[0]
        value = dict(path=path, signature=signature, datasets=datasets, nvis=nvis,
                     frequency=frequency, velocity=velocity,
                     center=SkyCoord(ra=center[0]*u.rad, dec=center[1]*u.rad, frame='icrs'),
                     keep_autocorr=bool(source.attrs['keep_autocorr']),
                     prefer_weight_spectrum=bool(source.attrs['prefer_weight_spectrum']))
    if _signature(os.stat(path)) != signature:
        raise RuntimeError(f'HDF5 cache changed while reading metadata: {path}')
    cache[path] = value
    return value


def _channel_indices(selection, size):
    if selection is None or isinstance(selection, slice):
        indices = np.arange(size, dtype=np.intp)
        if selection is not None:
            indices = indices[selection]
    else:
        indices = np.asarray(selection, dtype=np.intp)
    if indices.ndim != 1 or not indices.size:
        raise ValueError('chan_sel must select a nonempty one-dimensional channel sequence')
    if np.any(indices < 0) or np.any(indices >= size):
        raise IndexError('chan_sel contains a channel outside the cache frequency grid')
    return indices


def _runs(indices):
    begin = 0
    for end in range(1, len(indices)+1):
        if end == len(indices) or indices[end] != indices[end-1]+1:
            yield begin, end, int(indices[begin]), int(indices[end-1])+1
            begin = end


@contextmanager
def _source_array(handle, h5, name, info, backend):
    """Read a small row-dependent array; never map shared-filesystem pages."""
    if backend == 'auto' and hasattr(os, 'preadv') and info['offset'] is not None:
        array = np.empty(info['shape'], dtype=info['dtype'])
        _pread_into(handle.fileno(), array, info['offset'])
        yield array
    else:
        yield h5()[name][...]


def _pread_into(fd, destination, offset):
    """Fill a contiguous NumPy view, handling short reads without extra arrays."""
    view = memoryview(destination).cast('B')
    done = 0
    while done < len(view):
        count = os.preadv(fd, [view[done:]], offset+done)
        if count == 0:
            raise EOFError('Unexpected end of contiguous HDF5 dataset')
        done += count


def _fill_file(meta, arrays, beam, offset, indices, uvmin, uvmax, backend):
    path = meta['path']
    logger.info(f'    [MS] Opening: {path}')
    nv = meta['nvis']
    if not nv:
        logger.info(f'    [MS] Done: {Path(path).name}  rows=0')
        return
    region = slice(offset, offset+nv)
    source_h5 = None
    def h5():
        nonlocal source_h5
        if source_h5 is None:
            source_h5 = _h5py().File(path, 'r')
        return source_h5
    try:
        with open(path, 'rb', buffering=0) as handle:
            if _signature(os.fstat(handle.fileno())) != meta['signature']:
                raise RuntimeError(f'HDF5 cache changed after planning the read: {path}')
            descriptors = meta['datasets']
            if 'uvw' in descriptors:
                with _source_array(handle, h5, 'uvw', descriptors['uvw'], backend) as source:
                    for column,name in enumerate(('uu','vv','ww')):
                        arrays[name][beam,region] = source[:,column]
            else:
                for name in ('uu','vv','ww'):
                    with _source_array(handle, h5, name, descriptors[name], backend) as source:
                        arrays[name][beam,region] = source.reshape(-1)
            uu, vv, ww = (arrays[name][beam,region] for name in ('uu','vv','ww'))
            baseline = np.sqrt(uu**2 + vv**2 + ww**2)
            bad = ~((baseline >= uvmin) & (baseline <= uvmax))
            spectral_names = ['data_I','flag_I'] + (['sigma_I'] if 'sigma_I' in descriptors else [])
            for name in spectral_names:
                info = descriptors[name]
                destination = arrays[name]
                if backend == 'auto' and hasattr(os, 'preadv') and info['offset'] is not None and info['dtype'] == destination.dtype:
                    for output_channel, input_channel in enumerate(indices):
                        source_offset = info['offset'] + int(input_channel)*nv*info['dtype'].itemsize
                        _pread_into(handle.fileno(), destination[output_channel,beam,region], source_offset)
                else:
                    dataset = h5()[name]
                    for begin,end,start,stop in _runs(indices):
                        source_sel = (slice(start,stop),slice(None)) if dataset.ndim == 2 else (slice(start,stop),slice(None),slice(None))
                        dataset.read_direct(destination, source_sel=source_sel,
                                            dest_sel=(slice(begin,end),beam if dataset.ndim == 2 else slice(beam,beam+1),region))
            if 'sigma_rows' in descriptors:
                with _source_array(handle, h5, 'sigma_rows', descriptors['sigma_rows'], backend) as source:
                    arrays['sigma_I'][:,beam,region] = source.reshape(-1)[None,:]
            arrays['sigma_I'][:,beam,region][:,bad] = np.inf
            arrays['flag_I'][:,beam,region][:,bad] = True
    finally:
        if source_h5 is not None:
            source_h5.close()
    logger.info(f'    [MS] Done: {Path(path).name}  rows={nv}')


class HDF5Reader:
    """Read caches directly into their final merged or stacked positions.

    ``n_workers > 1`` uses I/O threads sharing the destination arrays. Metadata
    and threads are reused across calls; ``close()`` releases them. The returned
    arrays own their memory and remain valid after the reader is closed.

    ``io_backend="auto"`` uses buffered positional reads for compatible
    contiguous datasets and HDF5 for everything else. ``"h5py"`` forces the
    HDF5 library, useful when diagnosing storage compatibility. HDF5 library
    calls may serialize across threads, so contiguous caches are preferred.

    A reader instance should be used by one caller at a time. Different
    instances can be used concurrently. Channel batches remain caller-selected.
    """
    def __init__(self, *, prefer_weight_spectrum=True, keep_autocorr=False, n_workers=0, io_backend='auto'):
        if io_backend not in ('auto', 'h5py'):
            raise ValueError('io_backend must be auto or h5py')
        self.prefer_weight_spectrum = prefer_weight_spectrum
        self.keep_autocorr = keep_autocorr
        self.n_workers = n_workers
        self.io_backend = io_backend
        self._metadata_cache = {}
        self._pool = None
        self._pool_size = 0

    def close(self):
        """Release idle workers and metadata; previously returned data stay valid."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
            self._pool_size = 0
        self._metadata_cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def list_ms(self, ms_dir):
        return _list_h5_sorted(ms_dir)

    def freq_grid(self, ms_dir, rest_freq=1.42040575177e9):
        return _metadata(self.list_ms(ms_dir)[0], self._metadata_cache)['frequency'].copy()

    def read_block_I(self, cache_dir, **kwargs):
        return self._read([cache_dir], mode='merge', **kwargs)

    def read_blocks_I(self, ms_root=None, **kwargs):
        if ms_root is None:
            ms_root = kwargs.pop('cache_root', None)
        if ms_root is None:
            raise TypeError("read_blocks_I() missing required argument: 'ms_root'")
        return self._read(_list_h5_block_dirs(ms_root), **kwargs)

    def _read(self, block_dirs, *, uvmin=0., uvmax=float('inf'), chan_sel=None,
              rest_freq=1.42040575177e9, keep_autocorr=None, prefer_weight_spectrum=None,
              mode='merge', n_workers=None, target_center=None, target_radius=None,
              center_tol_deg=1e-12, beam_sel=None):
        if mode not in ('merge','stack','separate'):
            raise ValueError('mode must be merge, stack, or separate')
        keep_autocorr = self.keep_autocorr if keep_autocorr is None else keep_autocorr
        prefer_weight_spectrum = self.prefer_weight_spectrum if prefer_weight_spectrum is None else prefer_weight_spectrum
        if mode == 'separate':
            return [self._read([block], uvmin=uvmin, uvmax=uvmax, chan_sel=chan_sel,
                    rest_freq=rest_freq, keep_autocorr=keep_autocorr, prefer_weight_spectrum=prefer_weight_spectrum,
                    n_workers=n_workers, target_center=target_center, target_radius=target_radius,
                    center_tol_deg=center_tol_deg, beam_sel=beam_sel) for block in block_dirs]
        workers = self.n_workers if n_workers is None else n_workers
        radius = _norm_radius(target_radius) if target_center is not None and target_radius is not None else None
        # Plan every beam/block offset before allocating the final arrays.
        plans = []
        frequency = velocity = None
        for directory in block_dirs:
            files = _list_h5_sorted(directory)
            first = _metadata(files[0],self._metadata_cache)
            local_indices = _channel_indices(chan_sel,len(first['frequency']))
            local_frequency = first['frequency'][local_indices]
            if frequency is None:
                frequency, velocity = local_frequency.copy(), first['velocity'][local_indices].copy()
            elif local_frequency.shape != frequency.shape or not np.allclose(local_frequency,frequency,rtol=0,atol=1e-6):
                raise ValueError(f'Frequency grids differ between blocks: {directory}')
            selected = _normalize_beam_sel(beam_sel,len(files))
            metadata = [_metadata(files[i],self._metadata_cache) for i in selected]
            counts=[]
            for meta in metadata:
                keep = radius is None or _within_radius(target_center,radius,meta['center'])
                if keep:
                    if meta['keep_autocorr'] != keep_autocorr:
                        raise ValueError(f"Cache autocorrelation setting does not match reader: {meta['path']}")
                    if meta['prefer_weight_spectrum'] != prefer_weight_spectrum:
                        raise ValueError(f"Cache weight setting does not match reader: {meta['path']}")
                    if meta['frequency'].shape != first['frequency'].shape or not np.allclose(meta['frequency'],first['frequency'],rtol=0,atol=1e-6):
                        raise ValueError(f"Frequency grid differs in HDF5 cache {meta['path']}")
                counts.append(meta['nvis'] if keep else 0)
            plans.append(dict(directory=directory, total=len(files), metadata=metadata,
                              counts=np.asarray(counts,dtype=np.int64), indices=local_indices))
        first = plans[0]
        if mode == 'merge':
            centers = [m['center'] for m in first['metadata']]
            for plan in plans[1:]:
                if len(plan['metadata']) != len(centers):
                    raise ValueError('Beam counts differ between blocks')
                if any(not _centers_equal(c,m['center'],center_tol_deg) for c,m in zip(centers,plan['metadata'])):
                    raise ValueError('Beam center mismatch between blocks')
            counts = np.sum([p['counts'] for p in plans],axis=0,dtype=np.int64)
        else:
            centers = [m['center'] for p in plans for m in p['metadata']]
            counts = np.concatenate([p['counts'] for p in plans])
        if np.any(counts > np.iinfo(np.int32).max):
            raise OverflowError('Visibility counts exceed the VisIData int32 limit')
        nvis = counts.astype(np.int32)
        nbeam, maxvis = len(nvis), int(nvis.max(initial=0))
        shape = (len(frequency),nbeam,maxvis)
        # Workers write disjoint regions; no large arrays cross process pipes.
        arrays = {name:np.zeros((nbeam,maxvis),np.float32) for name in ('uu','vv','ww')}
        arrays.update(data_I=np.zeros(shape,np.complex64),sigma_I=np.zeros(shape,np.float32),flag_I=np.zeros(shape,bool))
        for beam,nv in enumerate(nvis):
            arrays['flag_I'][:,beam,int(nv):] = True
        offsets = np.zeros(nbeam,dtype=np.int64)
        beam_offset = 0
        pool_size = min(max(1,int(workers)),max(len(p['metadata']) for p in plans))
        if pool_size > 1 and (self._pool is None or self._pool_size != pool_size):
            if self._pool is not None:
                self._pool.shutdown(wait=True)
            self._pool = ThreadPoolExecutor(max_workers=pool_size,thread_name_prefix='ivis-hdf5')
            self._pool_size = pool_size
        for plan in plans:
            metadata = plan['metadata']
            logger.info(f"[BLOCK] Loading {len(metadata)}/{plan['total']} beam(s) from: {plan['directory']}")
            logger.info(f'[BLOCK] Parallel read with {pool_size} workers (order-preserving; selective)' if pool_size>1 else '[BLOCK] Serial read (order-preserving; selective)')
            futures=[]
            try:
                for beam,(meta,nv) in enumerate(zip(metadata,plan['counts'])):
                    target = beam if mode == 'merge' else beam_offset+beam
                    if nv:
                        args = (meta,arrays,target,int(offsets[target]),plan['indices'],uvmin,uvmax,self.io_backend)
                        if pool_size>1:
                            futures.append(self._pool.submit(_fill_file,*args))
                        else:
                            _fill_file(*args)
                    offsets[target] += nv
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                for future in futures:
                    if not future.cancelled():
                        try:
                            future.result()
                        except BaseException:
                            pass
                raise
            logger.info(f"[BLOCK] Done: nchan={len(frequency)}, nbeam={len(metadata)}, nvis_max={int(plan['counts'].max(initial=0))} (read {int(np.count_nonzero(plan['counts']))}/{len(metadata)} beams)")
            beam_offset += len(metadata)
        return VisIData(frequency=frequency,velocity=velocity,centers=np.asarray(centers,dtype=object),nvis=nvis,**arrays)

    def iter_channel_slabs(self, cache_dir, **kwargs):
        indices = _channel_indices(kwargs.get('chan_sel'),len(self.freq_grid(cache_dir)))
        slab = int(kwargs.pop('slab',64))
        if slab < 1:
            raise ValueError('slab must be positive')
        for offset in range(0,len(indices),slab):
            selected = indices[offset:offset+slab]
            yield int(selected[0]),int(selected[-1])+1,self.read_block_I(cache_dir,**{**kwargs,'chan_sel':selected})


def read_hdf5_block_I(cache_dir, uvmin=0., uvmax=float('inf'), chan_sel=None,
                      rest_freq=1.42040575177e9, keep_autocorr=False,
                      prefer_weight_spectrum=False, n_workers=0, target_center=None,
                      target_radius=None, beam_sel=None):
    with HDF5Reader(prefer_weight_spectrum=prefer_weight_spectrum,keep_autocorr=keep_autocorr,n_workers=n_workers) as reader:
        return reader.read_block_I(cache_dir,uvmin=uvmin,uvmax=uvmax,chan_sel=chan_sel,
                                  rest_freq=rest_freq,target_center=target_center,target_radius=target_radius,beam_sel=beam_sel)


def read_hdf5_blocks_I(cache_root, uvmin=0., uvmax=float('inf'), chan_sel=None,
                       rest_freq=1.42040575177e9, keep_autocorr=False,
                       prefer_weight_spectrum=False, mode='merge', n_workers=0,
                       target_center=None,target_radius=None,center_tol_deg=1e-12,beam_sel=None):
    with HDF5Reader(prefer_weight_spectrum=prefer_weight_spectrum,keep_autocorr=keep_autocorr,n_workers=n_workers) as reader:
        return reader.read_blocks_I(cache_root,uvmin=uvmin,uvmax=uvmax,chan_sel=chan_sel,
            rest_freq=rest_freq,mode=mode,target_center=target_center,target_radius=target_radius,
            center_tol_deg=center_tol_deg,beam_sel=beam_sel)
