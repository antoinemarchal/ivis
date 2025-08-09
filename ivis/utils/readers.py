# -*- coding: utf-8 -*-
import os, glob, contextlib, numpy as np
from dataclasses import dataclass
from typing import Iterator, Tuple, Sequence, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from casacore.tables import table
from astropy.constants import c as c_light
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u

from ivis.logger import logger

# -------------------- utils --------------------

@contextlib.contextmanager
def _quiet_tables():
    """Silence casacore 'Successful readonly open...' spam while opening tables."""
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield

def _list_block_dirs(ms_root: str) -> list[str]:
    """
    Return a list of block directories under ms_root. A block directory is any
    directory (ms_root itself or its immediate subdirs) that contains at least one *.ms.
    """
    blocks = []
    # root itself as a block?
    if glob.glob(os.path.join(ms_root, "*.ms")):
        blocks.append(ms_root)

    # immediate subdirs
    for name in sorted(os.listdir(ms_root)):
        sub = os.path.join(ms_root, name)
        if os.path.isdir(sub) and glob.glob(os.path.join(sub, "*.ms")):
            blocks.append(sub)

    if not blocks:
        raise FileNotFoundError(f"No *.ms found in {ms_root} or its immediate subdirectories.")
    return blocks

def _check_same_freq_grid(blocks: list["VisIData"]) -> None:
    """Ensure all blocks have identical frequency arrays (required for concat)."""
    ref = blocks[0].frequency
    for i, b in enumerate(blocks[1:], start=1):
        if b.frequency.shape != ref.shape or not np.allclose(b.frequency, ref, rtol=0, atol=1e-6):
            raise ValueError(
                f"Frequency grids differ between blocks (block 0 vs {i}). "
                "Use the same SPW/chan_sel or resample first."
            )

def _list_ms(ms_dir: str) -> Sequence[str]:
    ms_list = sorted(glob.glob(os.path.join(ms_dir, "*.ms")))
    if not ms_list:
        raise FileNotFoundError(f"No .ms found in: {ms_dir}")
    return ms_list

def _phasecenter(ms_path: str) -> SkyCoord:
    with _quiet_tables():
        with table(f"{ms_path}/FIELD", readonly=True) as t:
            ra_rad, dec_rad = t.getcol("PHASE_DIR")[0, 0, :]
    ra_hms  = Angle(ra_rad, unit=u.rad).to_string(unit=u.hourangle, sep=":")
    dec_dms = Angle(dec_rad, unit=u.rad).to_string(unit=u.deg,       sep=":")
    return SkyCoord(ra_hms, dec_dms, unit=(u.hourangle, u.deg), frame="icrs")

def _freqs(ms_path: str):
    with _quiet_tables():
        with table(f"{ms_path}/SPECTRAL_WINDOW", readonly=True) as t:
            freqs = np.atleast_1d(np.squeeze(t.getcol("CHAN_FREQ"))).astype(np.float64)  # Hz
    rest = 1.42040575177e9 * u.Hz
    vel_q = ((rest - freqs * u.Hz) / rest * c_light)        # Quantity [m/s]
    vel   = vel_q.to_value(u.km/u.s).astype(np.float64)     # km/s
    return freqs, vel

# -------------------- I-only container (channel-major) --------------------

@dataclass
class VisIData:
    # Axes / coords
    frequency: np.ndarray     # (nchan,) float64 [Hz]
    velocity:  np.ndarray     # (nchan,) float64 [km/s]
    centers:   np.ndarray     # (nbeam,), dtype=object (SkyCoord)
    nvis:      np.ndarray     # (nbeam,) int32

    # Coordinates (stored in METERS; we scale to wavelengths per channel on-demand)
    uu: np.ndarray            # (nbeam, nvis_max) float32  [meters]
    vv: np.ndarray            # (nbeam, nvis_max) float32  [meters]
    ww: np.ndarray            # (nbeam, nvis_max) float32  [meters]

    # I-only measurements (channel-major)
    data_I:  np.ndarray       # (nchan, nbeam, nvis_max) complex64
    sigma_I: np.ndarray       # (nchan, nbeam, nvis_max) float32
    flag_I:  np.ndarray       # (nchan, nbeam, nvis_max) bool

    def __post_init__(self):
        self.frequency = np.asarray(self.frequency, dtype=np.float64)
        self.velocity  = np.asarray(self.velocity,  dtype=np.float64)
        self.centers   = np.asarray(self.centers,   dtype=object)
        self.nvis      = np.asarray(self.nvis,      dtype=np.int32)
        self.uu        = np.asarray(self.uu,        dtype=np.float32, order="C")
        self.vv        = np.asarray(self.vv,        dtype=np.float32, order="C")
        self.ww        = np.asarray(self.ww,        dtype=np.float32, order="C")
        self.data_I    = np.asarray(self.data_I,    dtype=np.complex64, order="C")
        self.sigma_I   = np.asarray(self.sigma_I,   dtype=np.float32,   order="C")
        self.flag_I    = np.asarray(self.flag_I,    dtype=bool,         order="C")

        nchan = self.frequency.shape[0]
        nbeam, nvis_max = self.uu.shape
        assert self.vv.shape == (nbeam, nvis_max)
        assert self.ww.shape == (nbeam, nvis_max)
        assert self.centers.shape == (nbeam,)
        assert self.nvis.shape == (nbeam,)
        assert self.data_I.shape  == (nchan, nbeam, nvis_max)
        assert self.sigma_I.shape == (nchan, nbeam, nvis_max)
        assert self.flag_I.shape  == (nchan, nbeam, nvis_max)

        # Safety: flag padded tails
        for b in range(nbeam):
            nv = int(self.nvis[b])
            if nv < nvis_max:
                self.flag_I[:, b, nv:] = True

    def slice_chan_beam_I(self, c: int, b: int):
        """
        Return I, σ, and (u,v,w) in WAVELENGTHS for the given channel & beam,
        with flagged visibilities removed.
        """
        nv = int(self.nvis[b])
        flg = self.flag_I[c, b, :nv]
        good = ~flg

        # exact per-channel scaling: meters → wavelengths
        scale = self.frequency[c] / c_light.value  # scalar
        uu_l = (self.uu[b, :nv] * scale)[good]
        vv_l = (self.vv[b, :nv] * scale)[good]
        ww_l = (self.ww[b, :nv] * scale)[good]

        return (
            self.data_I [c, b, :nv][good],
            self.sigma_I[c, b, :nv][good],
            uu_l, vv_l, ww_l,
        )

    def iter_chan_beam_I(self) -> Iterator[Tuple[int,int,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]]:
        nchan, nbeam, _ = self.data_I.shape
        for c in range(nchan):
            for b in range(nbeam):
                I, sI, uu, vv, ww = self.slice_chan_beam_I(c, b)
                if I.size:
                    yield c, b, I, sI, uu, vv, ww

# ----------------------------- Public loader ------------------------------

def _read_one_ms(ms_path: str,
                 chan_idx: np.ndarray,
                 keep_autocorr: bool,
                 prefer_weight_spectrum: bool):
    """
    Worker: read a single .ms, return per-beam arrays (no UV filtering here).
    """
    logger.info(f"    [MS] Opening: {ms_path}")

    with _quiet_tables():
        with table(ms_path, readonly=True) as T:
            UVW = T.getcol("UVW")                   # (nrow, 3) meters
            A1  = T.getcol("ANTENNA1"); A2 = T.getcol("ANTENNA2")
            has_ws = ("WEIGHT_SPECTRUM" in T.colnames()) and prefer_weight_spectrum

            # Read only selected channels
            if chan_idx.size == 1 or np.all(np.diff(chan_idx) == 1):
                ch0 = int(chan_idx[0]); ch1 = int(chan_idx[-1])
                DATA = T.getcolslice("DATA",  blc=[ch0, 0], trc=[ch1, -1])   # (nrow, nchan, npol)
                FLAG = T.getcolslice("FLAG",  blc=[ch0, 0], trc=[ch1, -1])
                if has_ws:
                    W = T.getcolslice("WEIGHT_SPECTRUM", blc=[ch0, 0], trc=[ch1, -1])
                else:
                    SIGMA = T.getcol("SIGMA")                                 # (nrow, npol)
            else:
                d_blocks=[]; f_blocks=[]; w_blocks=[]
                for ch in chan_idx.tolist():
                    d_blocks.append(T.getcolslice("DATA", blc=[ch,0], trc=[ch,-1])[:,None,:])
                    f_blocks.append(T.getcolslice("FLAG", blc=[ch,0], trc=[ch,-1])[:,None,:])
                    if has_ws:
                        w_blocks.append(T.getcolslice("WEIGHT_SPECTRUM", blc=[ch,0], trc=[ch,-1])[:,None,:])
                DATA = np.concatenate(d_blocks, axis=1)
                FLAG = np.concatenate(f_blocks, axis=1)
                if has_ws:
                    W = np.concatenate(w_blocks, axis=1)
                else:
                    SIGMA = T.getcol("SIGMA")

    # Row mask: remove autocorr only (per-channel uv mask happens in parent)
    row_mask = np.ones(UVW.shape[0], dtype=bool)
    if not keep_autocorr:
        row_mask &= (A1 != A2)

    UVW  = UVW[row_mask]
    DATA = DATA[row_mask]
    FLAG = FLAG[row_mask]
    if has_ws:
        W = W[row_mask]
    else:
        SIGMA = SIGMA[row_mask]

    nrow2, nchan_chk, npol = DATA.shape
    # Stokes I + noise
    if npol == 1:
        I  = DATA[..., 0]                # (nrow2, nchan)
        fI = FLAG[..., 0]
        if has_ws:
            eps = 1e-12
            sI = 1.0 / np.sqrt(np.maximum(W[..., 0], eps))
        else:
            row_sig = SIGMA[:, 0]
            sI = np.repeat(row_sig[:, None], nchan_chk, 1)
    else:
        p0, p1 = 0, -1
        I  = 0.5 * (DATA[..., p0] + DATA[..., p1])
        fI = (FLAG[..., p0] | FLAG[..., p1])
        if has_ws:
            eps  = 1e-12
            sig0 = 1.0 / np.sqrt(np.maximum(W[..., p0], eps))
            sig1 = 1.0 / np.sqrt(np.maximum(W[..., p1], eps))
            sI = 0.5 * np.sqrt(sig0**2 + sig1**2)
        else:
            row_sig0 = SIGMA[:, p0]; row_sig1 = SIGMA[:, p1]
            row_sI   = 0.5 * np.sqrt(row_sig0**2 + row_sig1**2)
            sI = np.repeat(row_sI[:, None], nchan_chk, 1)

    center = _phasecenter(ms_path)

    # Return (channel-major is done in parent after UV mask)
    out = {
        "uu": UVW[:, 0].astype(np.float32),
        "vv": UVW[:, 1].astype(np.float32),
        "ww": UVW[:, 2].astype(np.float32),
        "I":  I.astype(np.complex64),        # (nrow2, nchan)
        "sI": sI.astype(np.float32),         # (nrow2, nchan)
        "fI": fI.astype(bool),               # (nrow2, nchan)
        "center": center,
        "nrow": UVW.shape[0],
    }
    logger.info(f"    [MS] Done: {os.path.basename(ms_path)}  rows={out['nrow']}")
    return out


def read_ms_block_I(
    ms_dir: str,
    uvmin: float = 0.0,               # in wavelengths (desired)
    uvmax: float = float("inf"),      # in wavelengths (desired)
    chan_sel=None,                    # None | slice | list[int] | np.ndarray[int]
    keep_autocorr: bool = False,
    prefer_weight_spectrum: bool = True,
    n_workers: int = 0,               # 0/1 = serial; >1 = parallel per-MS
) -> "VisIData":
    """
    Load a directory of .ms files (one per beam) into an I-only, channel-major VisIData.
    Uses per-MS parallelism (processes) if n_workers > 1. Per-channel UV mask
    is applied to match the original behavior.
    """
    ms_list = _list_ms(ms_dir)
    logger.info(f"[BLOCK] Loading {len(ms_list)} beam(s) from: {ms_dir}")

    # Normalize channel selection
    all_freq, all_vel = _freqs(ms_list[0])
    nchan_total = all_freq.size
    if chan_sel is None:
        chan_idx = np.arange(nchan_total, dtype=int)
    elif isinstance(chan_sel, slice):
        chan_idx = np.arange(nchan_total, dtype=int)[chan_sel]
    else:
        chan_idx = np.asarray(chan_sel, dtype=int)
    if chan_idx.size == 0:
        raise ValueError(f"chan_sel selects 0 channels; available nchan={nchan_total}")

    frequency = all_freq[chan_idx]     # (nchan,)
    velocity  = all_vel [chan_idx]     # (nchan,)
    nchan = int(frequency.size)

    # Collect per-MS results
    centers = []
    uu_list=[]; vv_list=[]; ww_list=[]
    I_list=[]; sI_list=[]; fI_list=[]

    if n_workers and n_workers > 1:
        logger.info(f"[BLOCK] Parallel read with {n_workers} workers")
        ctx = get_context("fork")   # works on Linux/macOS; not on Windows
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            futs = {ex.submit(_read_one_ms, ms, chan_idx, keep_autocorr, prefer_weight_spectrum): ms
                    for ms in ms_list}
            for fut in as_completed(futs):
                ms = futs[fut]
                out = fut.result()
                centers.append(out["center"])
                uu_list.append(out["uu"]); vv_list.append(out["vv"]); ww_list.append(out["ww"])
                I_list.append(out["I"]);   sI_list.append(out["sI"]);   fI_list.append(out["fI"])
    else:
        logger.info("[BLOCK] Serial read")
        for ms in ms_list:
            out = _read_one_ms(ms, chan_idx, keep_autocorr, prefer_weight_spectrum)
            centers.append(out["center"])
            uu_list.append(out["uu"]); vv_list.append(out["vv"]); ww_list.append(out["ww"])
            I_list.append(out["I"]);   sI_list.append(out["sI"]);   fI_list.append(out["fI"])

    # Pack to dense (nchan, nbeam, nvis_max)
    nbeam = len(ms_list)
    nvis = np.array([u.shape[0] for u in uu_list], dtype=np.int32)
    nvis_max = int(nvis.max())

    data_I  = np.zeros((nchan, nbeam, nvis_max), dtype=np.complex64)
    sigma_I = np.zeros((nchan, nbeam, nvis_max), dtype=np.float32)
    flag_I  = np.ones( (nchan, nbeam, nvis_max), dtype=bool)

    uu = np.zeros((nbeam, nvis_max), dtype=np.float32)
    vv = np.zeros_like(uu); ww = np.zeros_like(uu)

    # Per‑channel UV mask (old behavior) and channel‑major transpose
    for b in range(nbeam):
        UVW0 = uu_list[b]; UVW1 = vv_list[b]; UVW2 = ww_list[b]
        I    = I_list[b];  sI    = sI_list[b];  fI = fI_list[b]     # (nrow_b, nchan)

        # baseline length in meters for each row
        bl_m   = np.sqrt((UVW0**2 + UVW1**2 + UVW2**2))[:, None]   # (nrow_b, 1)
        # convert to wavelengths per channel
        bl_lam = bl_m * (frequency[None, :] / c_light.value)       # (nrow_b, nchan)
        in_rng = (bl_lam >= uvmin) & (bl_lam <= uvmax)

        fI |= ~in_rng
        sI[~in_rng] = np.inf

        # to channel‑major for this beam
        I_cb  = I.transpose(1, 0).astype(np.complex64)    # (nchan, nvis_b)
        sI_cb = sI.transpose(1, 0).astype(np.float32)     # (nchan, nvis_b)
        fI_cb = fI.transpose(1, 0).astype(bool)           # (nchan, nvis_b)

        nv = int(UVW0.shape[0])
        uu[b, :nv] = UVW0
        vv[b, :nv] = UVW1
        ww[b, :nv] = UVW2
        data_I [:, b, :nv] = I_cb
        sigma_I[:, b, :nv] = sI_cb
        flag_I [:, b, :nv] = fI_cb

        nvis_good = np.count_nonzero(~fI_cb)
        # logger.info(f"    [MS] Packed beam {b}: nvis raw={nchan*nv}, nvis after flagging={nvis_good}")

    centers = np.asarray(centers, dtype=object)
    logger.info(f"[BLOCK] Done: nchan={nchan}, nbeam={nbeam}, nvis_max={nvis_max}")

    return VisIData(
        frequency=frequency,
        velocity=velocity,
        centers=centers,
        nvis=nvis,
        uu=uu, vv=vv, ww=ww,            # meters (scaled to λ on demand)
        data_I=data_I,
        sigma_I=sigma_I,
        flag_I=flag_I,
    )


# def read_ms_block_I(
#     ms_dir: str,
#     uvmin: float = 0.0,               # in wavelengths (desired)
#     uvmax: float = float("inf"),      # in wavelengths (desired)
#     chan_sel=None,                    # None | slice | list[int] | np.ndarray[int]
#     keep_autocorr: bool = False,
#     prefer_weight_spectrum: bool = True,
# ) -> "VisIData":
#     """
#     Load a directory of .ms files (one per beam) into an I-only, channel-major VisIData.
#     Only the requested channels are read from disk (getcolslice).

#     UV filtering is done in METERS using bounds derived from the selected channel range,
#     so we don't throw away visibilities that are in-range for some channel.
#     """
#     ms_list = _list_ms(ms_dir)

#     # ---- normalize channel selection -> explicit indices ----
#     all_freq, all_vel = _freqs(ms_list[0])
#     nchan_total = all_freq.size
#     if chan_sel is None:
#         chan_idx = np.arange(nchan_total, dtype=int)
#     elif isinstance(chan_sel, slice):
#         chan_idx = np.arange(nchan_total, dtype=int)[chan_sel]
#     else:
#         chan_idx = np.asarray(chan_sel, dtype=int)
#     if chan_idx.size == 0:
#         raise ValueError(f"chan_sel selects 0 channels; available nchan={nchan_total}")

#     frequency = all_freq[chan_idx]
#     velocity  = all_vel [chan_idx]
#     nchan = int(frequency.size)

#     # --- derive conservative METER bounds from desired wavelength bounds across selected chans
#     f_min = float(frequency.min())
#     f_max = float(frequency.max())
#     uvmin_m = 0.0 if not np.isfinite(uvmin) else (uvmin * c_light.value / f_max)  # keep anything that could be >= uvmin
#     uvmax_m = np.inf if not np.isfinite(uvmax) else (uvmax * c_light.value / f_min)  # keep anything that could be <= uvmax

#     centers = []
#     uu_list=[]; vv_list=[]; ww_list=[]
#     I_list=[]; sI_list=[]; fI_list=[]

#     for ms in ms_list:
#         logger.info(f"    [MS] Reading: {ms}") 
#         centers.append(_phasecenter(ms))

#         with _quiet_tables():
#             with table(ms, readonly=True) as T:
#                 UVW = T.getcol("UVW")                   # (nrow, 3) [meters]
#                 A1  = T.getcol("ANTENNA1"); A2 = T.getcol("ANTENNA2")
#                 has_ws = ("WEIGHT_SPECTRUM" in T.colnames()) and prefer_weight_spectrum

#                 # ---- Read only selected channels ----
#                 if chan_idx.size == 1 or np.all(np.diff(chan_idx) == 1):
#                     # contiguous block
#                     ch0 = int(chan_idx[0]); ch1 = int(chan_idx[-1])
#                     DATA = T.getcolslice("DATA",  blc=[ch0, 0], trc=[ch1, -1])   # (nrow, nchan, npol)
#                     FLAG = T.getcolslice("FLAG",  blc=[ch0, 0], trc=[ch1, -1])
#                     if has_ws:
#                         W = T.getcolslice("WEIGHT_SPECTRUM", blc=[ch0, 0], trc=[ch1, -1])
#                     else:
#                         SIGMA = T.getcol("SIGMA")                                 # (nrow, npol)
#                 else:
#                     # non-contiguous: gather per-channel
#                     d_blocks=[]; f_blocks=[]; w_blocks=[]
#                     for ch in chan_idx.tolist():
#                         d_blocks.append(T.getcolslice("DATA", blc=[ch,0], trc=[ch,-1])[:,None,:])
#                         f_blocks.append(T.getcolslice("FLAG", blc=[ch,0], trc=[ch,-1])[:,None,:])
#                         if has_ws:
#                             w_blocks.append(T.getcolslice("WEIGHT_SPECTRUM", blc=[ch,0], trc=[ch,-1])[:,None,:])
#                     DATA = np.concatenate(d_blocks, axis=1)
#                     FLAG = np.concatenate(f_blocks, axis=1)
#                     if has_ws:
#                         W = np.concatenate(w_blocks, axis=1)
#                     else:
#                         SIGMA = T.getcol("SIGMA")

#         # ---- row mask: only autocorr removal; no uvmin/uvmax here ----
#         row_mask = np.ones(UVW.shape[0], dtype=bool)
#         if not keep_autocorr:
#             row_mask &= (A1 != A2)
            
#         UVW  = UVW[row_mask]
#         DATA = DATA[row_mask]
#         FLAG = FLAG[row_mask]
#         if has_ws:
#             W = W[row_mask]
#         else:
#             SIGMA = SIGMA[row_mask]
    
#         nrow2, nchan_chk, npol = DATA.shape
#         if nchan_chk != nchan:
#             raise RuntimeError("Channel selection mismatch after masking.")

#         # ---- Stokes I combine ----
#         if npol == 1:
#             I  = DATA[..., 0]                # (nrow2, nchan)
#             fI = FLAG[..., 0]
#             if has_ws:
#                 eps = 1e-12
#                 sI = 1.0 / np.sqrt(np.maximum(W[..., 0], eps))
#             else:
#                 row_sig = SIGMA[:, 0]                         # (nrow2,)
#                 sI = np.repeat(row_sig[:, None], nchan, 1)    # (nrow2, nchan)
#         else:
#             p0, p1 = 0, -1
#             I  = 0.5 * (DATA[..., p0] + DATA[..., p1])
#             fI = (FLAG[..., p0] | FLAG[..., p1])
#             if has_ws:
#                 eps = 1e-12
#                 sig0 = 1.0 / np.sqrt(np.maximum(W[..., p0], eps))
#                 sig1 = 1.0 / np.sqrt(np.maximum(W[..., p1], eps))
#                 sI = 0.5 * np.sqrt(sig0**2 + sig1**2)
#             else:
#                 row_sig0 = SIGMA[:, p0]; row_sig1 = SIGMA[:, p1]
#                 row_sI   = 0.5 * np.sqrt(row_sig0**2 + row_sig1**2)
#                 sI = np.repeat(row_sI[:, None], nchan, 1)

#         # --- per-channel UV mask in wavelengths (matches old behavior) ---
#         # baseline length in meters for each (remaining) row
#         bl_m = np.sqrt((UVW**2).sum(axis=1))                  # (nrow2,)
        
#         # convert to wavelengths per channel
#         bl_lam = bl_m[:, None] * (frequency[None, :] / c_light.value)  # (nrow2, nchan)
        
#         in_range = (bl_lam >= uvmin) & (bl_lam <= uvmax)      # (nrow2, nchan)
        
#         # flag vis that are out-of-range for that channel
#         fI |= ~in_range
        
#         # (optional but nice): make those samples weightless
#         sI[~in_range] = np.inf
#         # (optional): zero their vis (won’t be used once flagged anyway)
#         # I[~in_range] = 0

#         # ---- to channel-major for this beam ----
#         I_cb  = I.transpose(1, 0).astype(np.complex64)   # (nchan, nvis_b)
#         sI_cb = sI.transpose(1, 0).astype(np.float32)    # (nchan, nvis_b)
#         fI_cb = fI.transpose(1, 0).astype(bool)          # (nchan, nvis_b)

#         # ---- coords in METERS (no channel duplication) ----
#         uu_list.append(UVW[:, 0].astype(np.float32))
#         vv_list.append(UVW[:, 1].astype(np.float32))
#         ww_list.append(UVW[:, 2].astype(np.float32))
#         I_list.append(I_cb); sI_list.append(sI_cb); fI_list.append(fI_cb)

#     # ---- pack to dense (nchan, nbeam, nvis_max) ----
#     nbeam = len(ms_list)
#     nvis = np.array([u.shape[0] for u in uu_list], dtype=np.int32)
#     nvis_max = int(nvis.max())

#     data_I  = np.zeros((nchan, nbeam, nvis_max), dtype=np.complex64)
#     sigma_I = np.zeros((nchan, nbeam, nvis_max), dtype=np.float32)
#     flag_I  = np.ones( (nchan, nbeam, nvis_max), dtype=bool)

#     uu = np.zeros((nbeam, nvis_max), dtype=np.float32)
#     vv = np.zeros_like(uu); ww = np.zeros_like(uu)

#     for b in range(nbeam):
#         nv = int(nvis[b])
#         uu[b, :nv] = uu_list[b]
#         vv[b, :nv] = vv_list[b]
#         ww[b, :nv] = ww_list[b]
#         data_I [:, b, :nv] = I_list [b]
#         sigma_I[:, b, :nv] = sI_list[b]
#         flag_I [:, b, :nv] = fI_list[b]    # padded tails remain flagged=True

#     centers = np.asarray(centers, dtype=object)

#     return VisIData(
#         frequency=frequency,
#         velocity=velocity,
#         centers=centers,
#         nvis=nvis,
#         uu=uu, vv=vv, ww=ww,            # stored in meters
#         data_I=data_I,
#         sigma_I=sigma_I,
#         flag_I=flag_I,
#     )

def read_ms_blocks_I(
    ms_root: str,
    uvmin: float = 0.0,
    uvmax: float = float("inf"),
    chan_sel=None,                   # None | slice | list[int] | np.ndarray[int]
    keep_autocorr: bool = False,
    prefer_weight_spectrum: bool = True,
    mode: str = "concat",            # "concat" | "separate"
    n_workers: int = 0,               # 0/1 = serial; >1 = parallel per-MS    
) -> "VisIData | list[VisIData]":
    """
    Load multiple blocks of observations located under ms_root.

    - If ms_root contains *.ms directly, it's treated as a single block.
    - Any immediate subdirectory of ms_root that contains *.ms is also a block.

    mode="concat":  returns one VisIData with beams from all blocks concatenated
    mode="separate": returns a list[VisIData], one per block
    """
    block_dirs = _list_block_dirs(ms_root)

    # Load each block independently with your optimized reader
    blocks: list[VisIData] = []
    for bdir in block_dirs:
        logger.info(f"[BLOCK] Loading block from: {bdir}")
        vi = read_ms_block_I(
            bdir,
            uvmin=uvmin,
            uvmax=uvmax,
            chan_sel=chan_sel,
            keep_autocorr=keep_autocorr,
            prefer_weight_spectrum=prefer_weight_spectrum,
            n_workers=n_workers,
        )
        blocks.append(vi)

    if mode == "separate":
        return blocks

    if mode != "concat":
        raise ValueError('mode must be "concat" or "separate"')

    # ---------------- concat path ----------------
    _check_same_freq_grid(blocks)  # ensure same channels across blocks

    # Concatenate beams; keep per-beam nvis; pad to global nvis_max
    nchan = blocks[0].frequency.size
    total_beams = sum(b.uu.shape[0] for b in blocks)
    global_nvis_max = max(int(b.nvis.max()) for b in blocks)

    # Allocate output
    data_I  = np.zeros((nchan, total_beams, global_nvis_max), dtype=np.complex64)
    sigma_I = np.zeros((nchan, total_beams, global_nvis_max), dtype=np.float32)
    flag_I  = np.ones( (nchan, total_beams, global_nvis_max), dtype=bool)

    uu = np.zeros((total_beams, global_nvis_max), dtype=np.float32)
    vv = np.zeros_like(uu); ww = np.zeros_like(uu)
    nvis = np.zeros((total_beams,), dtype=np.int32)
    centers = np.empty((total_beams,), dtype=object)

    # Copy block by block
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
            data_I [:, dst, :nv] = blk.data_I[:, j, :nv]
            sigma_I[:, dst, :nv] = blk.sigma_I[:, j, :nv]
            flag_I [:, dst, :nv] = blk.flag_I[:, j, :nv]
            # padded tails remain flagged=True
        b_off += nb

    return VisIData(
        frequency=blocks[0].frequency,
        velocity=blocks[0].velocity,
        centers=centers,
        nvis=nvis,
        uu=uu, vv=vv, ww=ww,
        data_I=data_I,
        sigma_I=sigma_I,
        flag_I=flag_I,
    )

# ------------------------- channel-slab iterator --------------------------

def iter_channel_slabs(
    ms_dir: str,
    uvmin: float = 0.0,
    uvmax: float = float("inf"),
    chan_sel=None,                 # None | slice | list[int] | np.ndarray[int]
    slab: int = 64,                # max channels per slab
    keep_autocorr: bool = False,
    prefer_weight_spectrum: bool = True,
    n_workers: int = 0,               # 0/1 = serial; >1 = parallel per-MS
) -> Iterator[Tuple[int, int, VisIData]]:
    """
    Yield contiguous channel slabs so you can stream a big cube with low RAM.

    Yields:
        (start, stop, visI)
        where start/stop are absolute channel indices into the SPW (Python slice semantics),
        and visI is a VisIData with shape (stop-start, nbeam, nvis_max).
    """
    ms_list = _list_ms(ms_dir)
    all_freq, _ = _freqs(ms_list[0])
    all_idx = np.arange(all_freq.size, dtype=int)

    # Normalize chan_sel -> explicit indices
    if chan_sel is None:
        sel_idx = all_idx
    elif isinstance(chan_sel, slice):
        sel_idx = all_idx[chan_sel]
    else:
        sel_idx = np.asarray(chan_sel, dtype=int)

    if sel_idx.size == 0:
        return  # nothing to do

    i = 0
    n = sel_idx.size
    while i < n:
        # grow a contiguous run up to 'slab' channels
        j = i + 1
        while j < n and sel_idx[j] == sel_idx[j-1] + 1 and (j - i) < slab:
            j += 1

        start = int(sel_idx[i])
        stop  = int(sel_idx[j-1] + 1)  # slice end (exclusive)

        visI = read_ms_block_I(
            ms_dir,
            uvmin=uvmin,
            uvmax=uvmax,
            chan_sel=slice(start, stop),
            keep_autocorr=keep_autocorr,
            prefer_weight_spectrum=prefer_weight_spectrum,
            n_workers=n_workers,            
        )
        yield start, stop, visI
        i = j

def iter_blocks_chan_beam_I(
    ms_root: str,
    uvmin: float = 0.0,
    uvmax: float = float("inf"),
    chan_sel=None,
    keep_autocorr: bool = False,
    prefer_weight_spectrum: bool = True,
    n_workers: int = 0,               # 0/1 = serial; >1 = parallel per-MS
):
    """
    Stream over (block_index, c, b, I, sI, uu, vv, ww) without concatenating.
    """
    block_dirs = _list_block_dirs(ms_root)
    for bi, bdir in enumerate(block_dirs):
        vis = read_ms_block_I(
            bdir,
            uvmin=uvmin,
            uvmax=uvmax,
            chan_sel=chan_sel,
            keep_autocorr=keep_autocorr,
            prefer_weight_spectrum=prefer_weight_spectrum,
            n_workers=n_workers,

        )
        for c, b, I, sI, uu, vv, ww in vis.iter_chan_beam_I():
            yield bi, c, b, I, sI, uu, vv, ww
        del vis

def iter_blocks_channel_slabs(
    ms_root: str,
    uvmin: float = 0.0,
    uvmax: float = float("inf"),
    chan_sel=None,
    slab: int = 64,
    keep_autocorr: bool = False,
    prefer_weight_spectrum: bool = True,
    n_workers: int = 0,               # 0/1 = serial; >1 = parallel per-MS    
):
    """
    Yield (bi, block_dir, c0, c1, visI) where visI is a VisIData slab for that block.
    """
    block_dirs = _list_block_dirs(ms_root)
    for bi, bdir in enumerate(block_dirs):
        for c0, c1, visI in iter_channel_slabs(
            bdir,
            uvmin=uvmin,
            uvmax=uvmax,
            chan_sel=chan_sel,
            slab=slab,
            keep_autocorr=keep_autocorr,
            prefer_weight_spectrum=prefer_weight_spectrum,
            n_workers=n_workers,
        ):
            yield bi, bdir, c0, c1, visI
            # caller should del visI when done to free RAM

def iter_blocks_chan_beam_via_slabs(
    ms_root: str,
    uvmin: float = 0.0,
    uvmax: float = float("inf"),
    chan_sel=None,
    slab: int = 64,
    keep_autocorr: bool = False,
    prefer_weight_spectrum: bool = True,
    n_workers: int = 0,               # 0/1 = serial; >1 = parallel per-MS
):
    """
    Yield (bi, c_abs, b, I, sI, uu, vv, ww), streaming through slabs per block.
    c_abs is the absolute channel index in the SPW.
    """
    block_dirs = _list_block_dirs(ms_root)
    for bi, bdir in enumerate(block_dirs):
        for c0, c1, visI in iter_channel_slabs(
            bdir,
            uvmin=uvmin,
            uvmax=uvmax,
            chan_sel=chan_sel,
            slab=slab,
            keep_autocorr=keep_autocorr,
            prefer_weight_spectrum=prefer_weight_spectrum,
        ):
            # iterate tiny chunks from the slab
            for c_rel in range(visI.data_I.shape[0]):
                c_abs = c0 + c_rel
                for b in range(visI.uu.shape[0]):
                    I, sI, uu, vv, ww = visI.slice_chan_beam_I(c_rel, b)
                    if I.size:
                        yield bi, c_abs, b, I, sI, uu, vv, ww
            del visI  # free slab memory

# ------------------------------- demo -------------------------------------

if __name__ == "__main__":
    # Example usage — adjust path + channels
    ms_dir = "/Users/antoine/Desktop/Synthesis/ivis/docs/tutorials/data_tutorials/ivis_data/msl_mw/"
    # ms_dir = "/Users/antoine/Desktop/Synthesis/ivis/docs/tutorials/data_tutorials/msdir2"

    # # Single shot load (channels 0..99)
    # visI = read_ms_block_I(
    #     ms_dir,
    #     uvmin=20.0,
    #     uvmax=5000.0,
    #     chan_sel=slice(0, 100),
    #     keep_autocorr=False,
    #     prefer_weight_spectrum=True,
    # )

    # print("VisIData loaded:")
    # print(f"  nchan   = {visI.frequency.size}")
    # print(f"  nbeam   = {visI.uu.shape[0]}")
    # print(f"  nvis_max= {visI.uu.shape[1]}")
    # print(f"  fmin/max= {visI.frequency.min():.3f} – {visI.frequency.max():.3f} Hz")

    # I, sI, uu, vv, ww = visI.slice_chan_beam_I(c=0, b=0)
    # print(f"Example slice (chan=0, beam=0): I={I.shape}, σ={sI.shape}, uu={uu.shape}")

    # # Streaming slabs example
    # for c0, c1, slab_vis in iter_channel_slabs(ms_dir, uvmin=20, uvmax=5000, chan_sel=slice(0, 128), slab=32):
    #     print(f"Slab [{c0}:{c1}) → {slab_vis.data_I.shape}")
    #     del slab_vis


    # concat
    print("Test #Concat all")
    vis_all = read_ms_blocks_I(
        ms_root=ms_dir,
        uvmin=20.0, uvmax=5000.0,
        chan_sel=slice(0, 4),
        keep_autocorr=False,
        prefer_weight_spectrum=True,
        mode="concat",
        n_workers=4,
    )
    for c, b, I, sI, uu, vv, ww in vis_all.iter_chan_beam_I():
        pass

    stop
    
    # # keep separate
    # vis_blocks = read_ms_blocks_I(
    #     ms_root=ms_dir,
    #     uvmin=0.0, uvmax=np.inf,
    #     chan_sel=slice(0, 64),
    #     mode="separate",
    # )
    
    # for bi, vis in enumerate(vis_blocks):
    #     print("block", bi, vis.data_I.shape)

    # Option A — Stream slabs per block (moderate RAM, simple)
    print("Test # Option A — Stream slabs per block (moderate RAM, simple)")
    for bi, bdir, c0, c1, visI in iter_blocks_channel_slabs(
        ms_root=ms_dir, uvmin=20, uvmax=5000, chan_sel=slice(0,8), slab=4
    ):
        logger.info(f"[block {bi}] {os.path.basename(bdir)} slab [{c0}:{c1}) -> {visI.data_I.shape}")
        del visI

    # Option B — Stream (block, channel, beam) inside slabs (lowest RAM)
    print("Test # Option B — Stream (block, channel, beam) inside slabs (lowest RAM)")
    for bi, c, b, I, sI, uu, vv, ww in iter_blocks_chan_beam_via_slabs(
        ms_root=ms_dir, uvmin=20, uvmax=5000, chan_sel=slice(0,8), slab=4
    ):
        # NUFFT/predict/imaging for just this (block,chan,beam) slice
        pass
