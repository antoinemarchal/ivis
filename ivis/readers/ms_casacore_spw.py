# -*- coding: utf-8 -*-
import contextlib
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Iterator, List, Optional, Tuple, Union

import astropy.units as u
import numpy as np
from astropy.constants import c as c_light
from casacore.tables import table, taql

from ivis.logger import logger
from ivis.readers import ms_casacore as base
from ivis.types import VisIData


def _spw_summaries(ms_path: str) -> list[tuple[int, int, float, float]]:
    out: list[tuple[int, int, float, float]] = []
    with base._quiet_tables():
        with table(f"{ms_path}/SPECTRAL_WINDOW", readonly=True) as t:
            for spw_id in range(t.nrows()):
                freqs = np.atleast_1d(np.squeeze(t.getcell("CHAN_FREQ", spw_id))).astype(np.float64)
                out.append((spw_id, int(freqs.size), float(freqs.min()), float(freqs.max())))
    return out


def _ddids_for_spw(ms_path: str, spw_sel: int) -> np.ndarray:
    with base._quiet_tables():
        with table(f"{ms_path}/DATA_DESCRIPTION", readonly=True) as t_dd:
            spw_per_ddid = np.asarray(t_dd.getcol("SPECTRAL_WINDOW_ID"), dtype=np.int64)

    ddids = np.flatnonzero(spw_per_ddid == int(spw_sel)).astype(np.int64)
    if ddids.size == 0:
        summary = ", ".join(f"SPW {spw_id}" for spw_id, *_ in _spw_summaries(ms_path))
        raise ValueError(f"{ms_path} does not contain SPECTRAL_WINDOW_ID={spw_sel}. Available: {summary}")
    return ddids


def _normalize_spw_sel(spw_sel, nms_total: int, beam_idx: np.ndarray) -> list[Optional[int]]:
    if spw_sel is None:
        return [None] * int(beam_idx.size)

    if np.isscalar(spw_sel):
        return [int(spw_sel)] * int(beam_idx.size)

    if isinstance(spw_sel, Sequence) and not isinstance(spw_sel, (str, bytes)):
        spw_list = list(spw_sel)
        if len(spw_list) == int(beam_idx.size):
            return [int(x) for x in spw_list]
        if len(spw_list) == nms_total:
            return [int(spw_list[i]) for i in beam_idx.tolist()]
        raise ValueError(
            f"spw_sel must have length {int(beam_idx.size)} (selected beams) or {nms_total} "
            f"(all beams before beam_sel); got {len(spw_list)}."
        )

    raise TypeError("spw_sel must be None, an int, or a sequence of ints.")


def _freqs(ms_path: str,
           rest_freq: float | u.Quantity = 1.42040575177e9 * u.Hz,
           spw_sel: Optional[int] = None):
    if spw_sel is None:
        return base._freqs(ms_path, rest_freq)

    with base._quiet_tables():
        with table(f"{ms_path}/SPECTRAL_WINDOW", readonly=True) as t:
            if spw_sel < 0 or spw_sel >= t.nrows():
                summary = ", ".join(f"SPW {spw_id}" for spw_id, *_ in _spw_summaries(ms_path))
                raise ValueError(f"{ms_path} has no SPW {spw_sel}. Available: {summary}")
            freqs = np.atleast_1d(np.squeeze(t.getcell("CHAN_FREQ", int(spw_sel)))).astype(np.float64)

    rest_freq = rest_freq * u.Hz if np.isscalar(rest_freq) else rest_freq.to(u.Hz)
    vel_q = ((rest_freq - freqs * u.Hz) / rest_freq * c_light)
    vel = vel_q.to_value(u.km / u.s).astype(np.float64)
    return freqs, vel


def _read_one_ms(ms_path: str,
                 chan_idx: np.ndarray,
                 keep_autocorr: bool,
                 prefer_weight_spectrum: bool,
                 spw_sel: Optional[int] = None):
    logger.info(f"    [MS] Opening: {ms_path}")

    with base._quiet_tables():
        with contextlib.ExitStack() as stack:
            t0 = stack.enter_context(table(ms_path, readonly=True))
            t = t0

            if spw_sel is not None:
                ddids = _ddids_for_spw(ms_path, spw_sel)
                if ddids.size == 1:
                    where = f"DATA_DESC_ID = {int(ddids[0])}"
                else:
                    where = f"DATA_DESC_ID IN [{', '.join(str(int(x)) for x in ddids)}]"
                t = stack.enter_context(taql(f"SELECT * FROM $t0 WHERE {where}"))

            uvw = t.getcol("UVW")
            a1 = t.getcol("ANTENNA1")
            a2 = t.getcol("ANTENNA2")
            has_ws = ("WEIGHT_SPECTRUM" in t.colnames()) and prefer_weight_spectrum

            if chan_idx.size == 1 or np.all(np.diff(chan_idx) == 1):
                ch0 = int(chan_idx[0])
                ch1 = int(chan_idx[-1])
                data = t.getcolslice("DATA", blc=[ch0, 0], trc=[ch1, -1])
                flag = t.getcolslice("FLAG", blc=[ch0, 0], trc=[ch1, -1])
                if has_ws:
                    w = t.getcolslice("WEIGHT_SPECTRUM", blc=[ch0, 0], trc=[ch1, -1])
                else:
                    sigma = t.getcol("SIGMA")
            else:
                d_blocks = []
                f_blocks = []
                w_blocks = []
                for ch in chan_idx.tolist():
                    d_blocks.append(t.getcolslice("DATA", blc=[ch, 0], trc=[ch, -1])[:, None, :])
                    f_blocks.append(t.getcolslice("FLAG", blc=[ch, 0], trc=[ch, -1])[:, None, :])
                    if has_ws:
                        w_blocks.append(t.getcolslice("WEIGHT_SPECTRUM", blc=[ch, 0], trc=[ch, -1])[:, None, :])
                data = np.concatenate(d_blocks, axis=1)
                flag = np.concatenate(f_blocks, axis=1)
                if has_ws:
                    w = np.concatenate(w_blocks, axis=1)
                else:
                    sigma = t.getcol("SIGMA")

    row_mask = np.ones(uvw.shape[0], dtype=bool)
    if not keep_autocorr:
        row_mask &= (a1 != a2)

    uvw = uvw[row_mask]
    data = data[row_mask]
    flag = flag[row_mask]
    if has_ws:
        w = w[row_mask]
    else:
        sigma = sigma[row_mask]

    _nrow2, nchan_chk, npol = data.shape
    if npol == 1:
        stokes_i = data[..., 0]
        flag_i = flag[..., 0]
        if has_ws:
            eps = 1e-12
            sigma_i = 1.0 / np.sqrt(np.maximum(w[..., 0], eps))
        else:
            row_sig = sigma[:, 0]
            sigma_i = np.repeat(row_sig[:, None], nchan_chk, 1)
    else:
        p0, p1 = 0, -1
        stokes_i = 0.5 * (data[..., p0] + data[..., p1])
        flag_i = (flag[..., p0] | flag[..., p1])
        if has_ws:
            eps = 1e-12
            sig0 = 1.0 / np.sqrt(np.maximum(w[..., p0], eps))
            sig1 = 1.0 / np.sqrt(np.maximum(w[..., p1], eps))
            sigma_i = 0.5 * np.sqrt(sig0**2 + sig1**2)
        else:
            row_sig0 = sigma[:, p0]
            row_sig1 = sigma[:, p1]
            row_sigma_i = 0.5 * np.sqrt(row_sig0**2 + row_sig1**2)
            sigma_i = np.repeat(row_sigma_i[:, None], nchan_chk, 1)

    center = base._phasecenter(ms_path)
    out = {
        "uu": uvw[:, 0].astype(np.float32),
        "vv": uvw[:, 1].astype(np.float32),
        "ww": uvw[:, 2].astype(np.float32),
        "I": stokes_i.astype(np.complex64),
        "sI": sigma_i.astype(np.float32),
        "fI": flag_i.astype(bool),
        "center": center,
        "nrow": uvw.shape[0],
    }
    logger.info(f"    [MS] Done: {os.path.basename(ms_path)}  rows={out['nrow']}")
    return out


def read_ms_block_I(
    ms_dir: str,
    uvmin: float = 0.0,
    uvmax: float = float("inf"),
    chan_sel=None,
    rest_freq: float = 1.42040575177e9,
    keep_autocorr: bool = False,
    prefer_weight_spectrum: bool = True,
    n_workers: int = 0,
    target_center=None,
    target_radius=None,
    beam_sel=None,
    spw_sel: Optional[int] = None,
) -> VisIData:
    ms_list = base._list_ms_sorted(ms_dir)
    nbeam_total = len(ms_list)

    sel = base._normalize_beam_sel(beam_sel, nbeam_total)
    ms_list = [ms_list[i] for i in sel]
    spw_per_ms = _normalize_spw_sel(spw_sel, nbeam_total, sel)

    logger.info(f"[BLOCK] Loading {len(ms_list)}/{nbeam_total} beam(s) from: {ms_dir}")

    all_freq, all_vel = _freqs(ms_list[0], rest_freq, spw_sel=spw_per_ms[0])
    nchan_total = all_freq.size
    if chan_sel is None:
        chan_idx = np.arange(nchan_total, dtype=int)
    elif isinstance(chan_sel, slice):
        chan_idx = np.arange(nchan_total, dtype=int)[chan_sel]
    else:
        chan_idx = np.asarray(chan_sel, dtype=int)
    if chan_idx.size == 0:
        raise ValueError(f"chan_sel selects 0 channels; available nchan={nchan_total}")

    frequency = all_freq[chan_idx]
    velocity = all_vel[chan_idx]
    nchan = int(frequency.size)

    nbeam = len(ms_list)
    centers = [None] * nbeam
    keep_mask = [True] * nbeam
    radius = base._norm_radius(target_radius) if (target_center is not None and target_radius is not None) else None

    for i, ms in enumerate(ms_list):
        center = base._phasecenter(ms)
        centers[i] = center
        if radius is not None and not base._within_radius(target_center, radius, center):
            keep_mask[i] = False
            logger.info(f"[SKIP-READ] {os.path.basename(ms)} outside {radius.to_string()} of "
                        f"{target_center.to_string('hmsdms')}")

    uu_list = [None] * nbeam
    vv_list = [None] * nbeam
    ww_list = [None] * nbeam
    i_list = [None] * nbeam
    si_list = [None] * nbeam
    fi_list = [None] * nbeam

    if n_workers and n_workers > 1:
        logger.info(f"[BLOCK] Parallel read with {n_workers} workers (order-preserving; selective)")
        ctx = get_context("fork")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            futs = {
                ex.submit(_read_one_ms, ms_list[i], chan_idx, keep_autocorr, prefer_weight_spectrum, spw_per_ms[i]): i
                for i in range(nbeam) if keep_mask[i]
            }
            for fut in as_completed(futs):
                i = futs[fut]
                out = fut.result()
                uu_list[i] = out["uu"]
                vv_list[i] = out["vv"]
                ww_list[i] = out["ww"]
                i_list[i] = out["I"]
                si_list[i] = out["sI"]
                fi_list[i] = out["fI"]
    else:
        logger.info("[BLOCK] Serial read (order-preserving; selective)")
        for i in range(nbeam):
            if not keep_mask[i]:
                continue
            out = _read_one_ms(ms_list[i], chan_idx, keep_autocorr, prefer_weight_spectrum, spw_per_ms[i])
            uu_list[i] = out["uu"]
            vv_list[i] = out["vv"]
            ww_list[i] = out["ww"]
            i_list[i] = out["I"]
            si_list[i] = out["sI"]
            fi_list[i] = out["fI"]

    nvis = np.zeros((nbeam,), dtype=np.int32)
    for i in range(nbeam):
        nvis[i] = 0 if uu_list[i] is None else int(uu_list[i].shape[0])
    nvis_max = int(nvis.max())

    data_i = np.zeros((nchan, nbeam, nvis_max), dtype=np.complex64)
    sigma_i = np.zeros((nchan, nbeam, nvis_max), dtype=np.float32)
    flag_i = np.ones((nchan, nbeam, nvis_max), dtype=bool)

    uu = np.zeros((nbeam, nvis_max), dtype=np.float32)
    vv = np.zeros_like(uu)
    ww = np.zeros_like(uu)

    for b in range(nbeam):
        if nvis[b] == 0:
            continue
        uvw0 = uu_list[b]
        uvw1 = vv_list[b]
        uvw2 = ww_list[b]
        stokes_i = i_list[b]
        sigma_chan = si_list[b]
        flag_chan = fi_list[b]

        bl_m = np.sqrt(uvw0**2 + uvw1**2 + uvw2**2)
        in_row = (bl_m >= uvmin) & (bl_m <= uvmax)
        flag_chan[~in_row, :] = True
        sigma_chan[~in_row, :] = np.inf

        i_cb = stokes_i.transpose(1, 0).astype(np.complex64)
        si_cb = sigma_chan.transpose(1, 0).astype(np.float32)
        fi_cb = flag_chan.transpose(1, 0).astype(bool)

        nv = nvis[b]
        uu[b, :nv] = uvw0
        vv[b, :nv] = uvw1
        ww[b, :nv] = uvw2
        data_i[:, b, :nv] = i_cb
        sigma_i[:, b, :nv] = si_cb
        flag_i[:, b, :nv] = fi_cb

    centers = np.asarray(centers, dtype=object)
    kept_count = int(np.count_nonzero(keep_mask)) if radius is not None else nbeam
    logger.info(f"[BLOCK] Done: nchan={nchan}, nbeam={nbeam}, nvis_max={nvis_max} "
                f"(read {kept_count}/{nbeam} beams)")

    return VisIData(
        frequency=frequency,
        velocity=velocity,
        centers=centers,
        nvis=nvis,
        uu=uu, vv=vv, ww=ww,
        data_I=data_i,
        sigma_I=sigma_i,
        flag_I=flag_i,
    )


def read_ms_blocks_I(
    ms_root: str,
    uvmin: float = 0.0,
    uvmax: float = float("inf"),
    chan_sel=None,
    rest_freq: float = 1.42040575177e9,
    keep_autocorr: bool = False,
    prefer_weight_spectrum: bool = True,
    mode: str = "merge",
    n_workers: int = 0,
    target_center=None,
    target_radius=None,
    center_tol_deg: float = 1e-12,
    beam_sel=None,
    spw_sel: Optional[int] = None,
) -> Union[VisIData, List[VisIData]]:
    block_dirs = base._list_block_dirs(ms_root)
    if not block_dirs:
        raise FileNotFoundError(f"No blocks found under {ms_root}")

    blocks: List[VisIData] = []
    for bdir in block_dirs:
        logger.info(f"[BLOCK] Loading block from: {bdir}")
        vi = read_ms_block_I(
            bdir,
            uvmin=uvmin,
            uvmax=uvmax,
            chan_sel=chan_sel,
            rest_freq=rest_freq,
            keep_autocorr=keep_autocorr,
            prefer_weight_spectrum=prefer_weight_spectrum,
            n_workers=n_workers,
            target_center=target_center,
            target_radius=target_radius,
            beam_sel=beam_sel,
            spw_sel=spw_sel,
        )
        blocks.append(vi)

    if mode == "separate":
        return blocks
    if mode not in ("merge", "stack"):
        raise ValueError('mode must be one of: "merge", "stack", "separate"')

    base._check_same_freq_grid(blocks)
    nchan = blocks[0].frequency.size

    if mode == "stack":
        total_beams = sum(b.uu.shape[0] for b in blocks)
        global_nvis_max = max(int(b.nvis.max()) for b in blocks)

        data_i = np.zeros((nchan, total_beams, global_nvis_max), dtype=np.complex64)
        sigma_i = np.zeros((nchan, total_beams, global_nvis_max), dtype=np.float32)
        flag_i = np.ones((nchan, total_beams, global_nvis_max), dtype=bool)
        uu = np.zeros((total_beams, global_nvis_max), dtype=np.float32)
        vv = np.zeros_like(uu)
        ww = np.zeros_like(uu)
        nvis = np.zeros((total_beams,), dtype=np.int32)
        centers = np.empty((total_beams,), dtype=object)

        b_off = 0
        for blk in blocks:
            nb = blk.uu.shape[0]
            for j in range(nb):
                nv = int(blk.nvis[j])
                dst = b_off + j
                nvis[dst] = nv
                centers[dst] = blk.centers[j]
                uu[dst, :nv] = blk.uu[j, :nv]
                vv[dst, :nv] = blk.vv[j, :nv]
                ww[dst, :nv] = blk.ww[j, :nv]
                data_i[:, dst, :nv] = blk.data_I[:, j, :nv]
                sigma_i[:, dst, :nv] = blk.sigma_I[:, j, :nv]
                flag_i[:, dst, :nv] = blk.flag_I[:, j, :nv]
            b_off += nb

        return VisIData(
            frequency=blocks[0].frequency,
            velocity=blocks[0].velocity,
            centers=centers,
            nvis=nvis,
            uu=uu, vv=vv, ww=ww,
            data_I=data_i,
            sigma_I=sigma_i,
            flag_I=flag_i,
        )

    nbeams = blocks[0].uu.shape[0]
    for bi, blk in enumerate(blocks[1:], start=1):
        if blk.uu.shape[0] != nbeams:
            raise ValueError(f"Block {bi} has {blk.uu.shape[0]} beams; expected {nbeams}.")
        for j in range(nbeams):
            if not base._centers_equal(blocks[0].centers[j], blk.centers[j], tol_deg=center_tol_deg):
                raise ValueError(
                    f"Beam center mismatch at beam {j}: block0={blocks[0].centers[j]} vs block{bi}={blk.centers[j]}"
                )

    merged_nvis = np.zeros((nbeams,), dtype=np.int32)
    for j in range(nbeams):
        merged_nvis[j] = sum(int(blk.nvis[j]) for blk in blocks)
    global_nvis_max = int(merged_nvis.max())

    data_i = np.zeros((nchan, nbeams, global_nvis_max), dtype=np.complex64)
    sigma_i = np.zeros((nchan, nbeams, global_nvis_max), dtype=np.float32)
    flag_i = np.ones((nchan, nbeams, global_nvis_max), dtype=bool)
    uu = np.zeros((nbeams, global_nvis_max), dtype=np.float32)
    vv = np.zeros_like(uu)
    ww = np.zeros_like(uu)
    centers = np.array(blocks[0].centers, dtype=object)

    for j in range(nbeams):
        w = 0
        for blk in blocks:
            nv = int(blk.nvis[j])
            if nv <= 0:
                continue
            uu[j, w:w+nv] = blk.uu[j, :nv]
            vv[j, w:w+nv] = blk.vv[j, :nv]
            ww[j, w:w+nv] = blk.ww[j, :nv]
            data_i[:, j, w:w+nv] = blk.data_I[:, j, :nv]
            sigma_i[:, j, w:w+nv] = blk.sigma_I[:, j, :nv]
            flag_i[:, j, w:w+nv] = blk.flag_I[:, j, :nv]
            w += nv

    return VisIData(
        frequency=blocks[0].frequency,
        velocity=blocks[0].velocity,
        centers=centers,
        nvis=merged_nvis,
        uu=uu, vv=vv, ww=ww,
        data_I=data_i,
        sigma_I=sigma_i,
        flag_I=flag_i,
    )


def iter_channel_slabs(
    ms_dir: str,
    uvmin: float = 0.0,
    uvmax: float = float("inf"),
    chan_sel=None,
    rest_freq: float = 1.42040575177e9,
    slab: int = 64,
    keep_autocorr: bool = False,
    prefer_weight_spectrum: bool = True,
    n_workers: int = 0,
    beam_sel=None,
    spw_sel: Optional[int] = None,
) -> Iterator[Tuple[int, int, VisIData]]:
    ms_list = base._list_ms_sorted(ms_dir)
    spw_per_ms = _normalize_spw_sel(spw_sel, len(ms_list), np.arange(len(ms_list), dtype=int))
    all_freq, _ = _freqs(ms_list[0], rest_freq, spw_sel=spw_per_ms[0])
    all_idx = np.arange(all_freq.size, dtype=int)

    if chan_sel is None:
        sel_idx = all_idx
    elif isinstance(chan_sel, slice):
        sel_idx = all_idx[chan_sel]
    else:
        sel_idx = np.asarray(chan_sel, dtype=int)

    if sel_idx.size == 0:
        return

    i = 0
    n = sel_idx.size
    while i < n:
        j = i + 1
        while j < n and sel_idx[j] == sel_idx[j - 1] + 1 and (j - i) < slab:
            j += 1

        start = int(sel_idx[i])
        stop = int(sel_idx[j - 1] + 1)
        vis_i = read_ms_block_I(
            ms_dir,
            uvmin=uvmin,
            uvmax=uvmax,
            chan_sel=slice(start, stop),
            rest_freq=rest_freq,
            keep_autocorr=keep_autocorr,
            prefer_weight_spectrum=prefer_weight_spectrum,
            n_workers=n_workers,
            beam_sel=beam_sel,
            spw_sel=spw_sel,
        )
        yield start, stop, vis_i
        i = j


class CasacoreReaderSPW:
    def __init__(self, *, prefer_weight_spectrum: bool = True,
                 keep_autocorr: bool = False, n_workers: int = 0):
        self.prefer_weight_spectrum = prefer_weight_spectrum
        self.keep_autocorr = keep_autocorr
        self.n_workers = n_workers

    def list_ms(self, ms_dir: str) -> List[str]:
        return base._list_ms_sorted(ms_dir)

    def freq_grid(self, ms_dir: str, rest_freq: float, spw_sel: Optional[int] = None):
        msl = self.list_ms(ms_dir)
        if not msl:
            raise FileNotFoundError(f"No .ms found in {ms_dir}")
        freqs, _vel = _freqs(msl[0], rest_freq, spw_sel=spw_sel)
        return freqs

    def read_blocks_I(self, ms_root: str, **kwargs) -> Union[VisIData, List[VisIData]]:
        return read_ms_blocks_I(
            ms_root,
            uvmin=kwargs.get("uvmin", 0.0),
            uvmax=kwargs.get("uvmax", float("inf")),
            chan_sel=kwargs.get("chan_sel"),
            rest_freq=kwargs.get("rest_freq", 1.42040575177e9),
            keep_autocorr=kwargs.get("keep_autocorr", self.keep_autocorr),
            prefer_weight_spectrum=kwargs.get("prefer_weight_spectrum", self.prefer_weight_spectrum),
            mode=kwargs.get("mode", "merge"),
            n_workers=kwargs.get("n_workers", self.n_workers),
            center_tol_deg=kwargs.get("center_tol_deg", 1e-12),
            target_center=kwargs.get("target_center"),
            target_radius=kwargs.get("target_radius"),
            beam_sel=kwargs.get("beam_sel"),
            spw_sel=kwargs.get("spw_sel"),
        )

    def iter_channel_slabs(self, ms_dir: str, **kwargs) -> Iterator[Tuple[int, int, VisIData]]:
        return iter_channel_slabs(
            ms_dir,
            uvmin=kwargs.get("uvmin", 0.0),
            uvmax=kwargs.get("uvmax", float("inf")),
            chan_sel=kwargs.get("chan_sel"),
            rest_freq=kwargs.get("rest_freq", 1.42040575177e9),
            slab=kwargs.get("slab", 64),
            keep_autocorr=kwargs.get("keep_autocorr", self.keep_autocorr),
            prefer_weight_spectrum=kwargs.get("prefer_weight_spectrum", self.prefer_weight_spectrum),
            n_workers=kwargs.get("n_workers", self.n_workers),
            beam_sel=kwargs.get("beam_sel"),
            spw_sel=kwargs.get("spw_sel"),
        )
