# -*- coding: utf-8 -*-
import os, re, glob, contextlib, numpy as np
from dataclasses import dataclass
from typing import Iterator, Tuple, List, Union, Sequence, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

from casacore.tables import table
from astropy.constants import c as c_light
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astropy.io.fits import Header

from ivis.logger import logger
from ivis.readers.base import Reader
from ivis.types import VisIData

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

    Accepts a single *.ms path (treated as a one-MS block).
    """
    # --- NEW: allow passing a single MS directly
    if ms_root.lower().endswith(".ms"):
        if not os.path.isdir(ms_root):
            raise FileNotFoundError(f"MS not found: {ms_root}")
        return [ms_root]

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
        raise FileNotFoundError(
            f"No *.ms found in {ms_root} or its immediate subdirectories."
        )
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

def _natkey(name: str):
    # Natural sort helper: "file2" < "file10"
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', name)]


def _beamkey_by_c10(path: str):
    """
    Prefer sorting by the number after 'C10_' in the filename, e.g. MW-C10_5_... -> 5.
    Fallback to a general natural sort by full filename.
    """
    name = Path(path).name
    m = re.search(r'[Cc]10[_-](\d+)', name)
    if m:
        return (0, int(m.group(1)), name.lower())
    # optional: also recognize 'BEAM_###'
    m2 = re.search(r'[Bb][Ee][Aa][Mm][_-]?(\d+)', name)
    if m2:
        return (0, int(m2.group(1)), name.lower())
    return (1, *_natkey(name))  # fallback: natural sort on the whole name


def _list_ms_sorted(ms_dir: str):
    """List .ms directories in the same order as raw `ls` (lexicographic).
    Accepts either a directory or a single .ms path.
    """
    # --- NEW: allow passing a single MS directly
    if ms_dir.lower().endswith(".ms"):
        if not os.path.isdir(ms_dir):
            raise FileNotFoundError(f"MS not found: {ms_dir}")
        return [ms_dir]

    # --- Original behavior (unchanged)
    items = []
    with os.scandir(ms_dir) as it:
        for de in it:
            if de.name.startswith('.'):
                continue
            if de.is_dir() and de.name.lower().endswith('.ms'):
                items.append(os.path.join(ms_dir, de.name))
    if not items:
        raise FileNotFoundError(f"No .ms found under {ms_dir}")
    return sorted(items)  # plain lexicographic, like `ls`


def _list_ms_sorted_natural(ms_dir: str):
    """List .ms directories and sort by beam number (C10_#), then natural name."""
    items = []
    with os.scandir(ms_dir) as it:
        for de in it:
            if de.name.startswith('.'):
                continue
            if de.is_dir() and de.name.lower().endswith('.ms'):
                items.append(os.path.join(ms_dir, de.name))
    if not items:
        raise FileNotFoundError(f"No .ms found under {ms_dir}")
    return sorted(items, key=_beamkey_by_c10)


def _list_block_dirs_sorted(ms_root: str):
    out = []
    with os.scandir(ms_root) as it:
        for de in it:
            if de.name.startswith('.'):
                continue
            if de.is_dir():
                out.append(os.path.join(ms_root, de.name))
    return sorted(out, key=lambda p: _natkey(Path(p).name))


def _phasecenter(ms_path: str) -> SkyCoord:
    with _quiet_tables():
        with table(f"{ms_path}/FIELD", readonly=True) as t:
            ra_rad, dec_rad = t.getcol("PHASE_DIR")[0, 0, :]
    ra_hms  = Angle(ra_rad, unit=u.rad).to_string(unit=u.hourangle, sep=":")
    dec_dms = Angle(dec_rad, unit=u.rad).to_string(unit=u.deg,       sep=":")
    return SkyCoord(ra_hms, dec_dms, unit=(u.hourangle, u.deg), frame="icrs")


def _freqs(ms_path: str, rest_freq: float | u.Quantity = 1.42040575177e9 * u.Hz):
    """
    Read frequencies from a MeasurementSet and compute corresponding velocities.

    Parameters
    ----------
    ms_path : str
        Path to the MeasurementSet.
    rest_freq : float or Quantity, optional
        Rest frequency of the spectral line [Hz]. Default: 1.42040575177 GHz (H I 21cm line).

    Returns
    -------
    freqs : ndarray (float64)
        Channel frequencies [Hz].
    vel   : ndarray (float64)
        Velocities relative to rest frequency [km/s].
    """
    with _quiet_tables():
        with table(f"{ms_path}/SPECTRAL_WINDOW", readonly=True) as t:
            freqs = np.atleast_1d(
                np.squeeze(t.getcol("CHAN_FREQ"))
            ).astype(np.float64)  # Hz

    # normalize rest_freq to Quantity
    rest_freq = rest_freq * u.Hz if np.isscalar(rest_freq) else rest_freq.to(u.Hz)

    vel_q = ((rest_freq - freqs * u.Hz) / rest_freq * c_light)   # m/s
    vel   = vel_q.to_value(u.km/u.s).astype(np.float64)          # km/s

    return freqs, vel

# -------- helpers --------

def _centers_equal(c0, c1, tol_deg: float) -> bool:
    """
    Compare 'center' objects with a tiny tolerance in degrees.
    Supports:
      - SkyCoord-like with .ra.deg/.dec.deg
      - (ra_deg, dec_deg) tuples/lists
      - Fallback to equality for strings/other identifiers.
    """
    try:
        if hasattr(c0, "ra") and hasattr(c0, "dec") and hasattr(c1, "ra") and hasattr(c1, "dec"):
            return (abs(float(c0.ra.deg)  - float(c1.ra.deg))  <= tol_deg and
                    abs(float(c0.dec.deg) - float(c1.dec.deg)) <= tol_deg)
        if isinstance(c0, (tuple, list)) and isinstance(c1, (tuple, list)) and len(c0) == 2 and len(c1) == 2:
            return (abs(float(c0[0]) - float(c1[0])) <= tol_deg and
                    abs(float(c0[1]) - float(c1[1])) <= tol_deg)
    except Exception:
        pass
    return c0 == c1

def _norm_radius(radius) -> Angle:
    """Accept Angle, Quantity, or float (deg) and return Angle."""
    if isinstance(radius, Angle):
        return radius
    if isinstance(radius, u.Quantity):
        return Angle(radius)
    # assume float in degrees
    return Angle(radius, unit=u.deg)

def _within_radius(center: SkyCoord, radius: Angle, coord: SkyCoord) -> bool:
    return coord.icrs.separation(center.icrs) <= radius

def _normalize_target_header_from_header(hdr: Header):
    """
    Returns (celestial WCS, (ny, nx)) from a FITS Header that may include
    FREQ/STOKES axes. Uses only the celestial part.
    """
    # Use celestial slice: works even if header advertises 3–4 axes
    wcs_cel = WCS(hdr).celestial
    nx = int(hdr["NAXIS1"])
    ny = int(hdr["NAXIS2"])
    return wcs_cel, (ny, nx)

def _coord_in_image(wcs_cel: WCS, shape: tuple, coord: SkyCoord) -> bool:
    """
    Returns True if coord projects inside image footprint.
    shape = (ny, nx). Uses zero-based pixel coords.
    """
    ny, nx = shape
    c_icrs = coord.icrs
    x, y = wcs_cel.world_to_pixel(c_icrs)  # origin=0 convention
    return (0 <= x <= nx - 1) and (0 <= y <= ny - 1)


def _normalize_beam_sel(beam_sel, nbeam_total: int) -> np.ndarray:
    """
    Return a sorted, unique array of beam indices in [0, nbeam_total).
    Accepts: None, int, slice, Sequence[int].
    """
    if beam_sel is None:
        return np.arange(nbeam_total, dtype=int)

    if isinstance(beam_sel, int):
        idx = np.array([beam_sel], dtype=int)

    elif isinstance(beam_sel, slice):
        idx = np.arange(nbeam_total, dtype=int)[beam_sel]

    else:
        idx = np.asarray(list(beam_sel), dtype=int)

    if idx.size == 0:
        raise ValueError("beam_sel selects 0 beams")

    if np.any(idx < 0) or np.any(idx >= nbeam_total):
        raise IndexError(f"beam_sel has indices outside [0, {nbeam_total-1}]")

    # unique + sorted (keeps deterministic ordering)
    return np.unique(idx)


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
    rest_freq: float = 1.42040575177e9, # HI rest frequency as default value in unit of Hz
    keep_autocorr: bool = False,
    prefer_weight_spectrum: bool = True,
    n_workers: int = 0,               # 0/1 = serial; >1 = parallel per-MS
    target_center: "SkyCoord | None" = None,
    target_radius: "Angle | u.Quantity | float | None" = None,  # float ⇒ degrees
    beam_sel=None,   # NEW: int | list[int] | slice | None
) -> "VisIData":
    """
    Load a directory of .ms files (one per beam) into an I-only, channel-major VisIData.

    Primary-beam order is preserved **even if** some beams are skipped: we do not
    remove beam slots; we simply avoid reading DATA for out-of-radius beams and
    leave their slots empty (nvis=0, flag=True).
    """
    # ---------------- 1) Deterministic beam list ----------------
    ms_list = _list_ms_sorted(ms_dir)          # or _list_ms_sorted_natural(ms_dir)
    nbeam_total = len(ms_list)

    sel = _normalize_beam_sel(beam_sel, nbeam_total) 
    ms_list = [ms_list[i] for i in sel]              

    logger.info(
        f"[BLOCK] Loading {len(ms_list)}/{nbeam_total} beam(s) from: {ms_dir} "
        # f"(beam_sel={sel.tolist()})"
    )

    # # ---------------- 1) Deterministic beam list (do NOT filter) ----------------
    # ms_list = _list_ms_sorted(ms_dir)
    # logger.info(f"[BLOCK] Loading {len(ms_list)} beam(s) from: {ms_dir}")

    # ---------------- 2) Channel selection (unchanged) ----------------
    all_freq, all_vel = _freqs(ms_list[0], rest_freq)
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

    # ---------------- 3) Decide which beams to actually read ----------------
    def _norm_radius(R):
        from astropy.coordinates import Angle
        import astropy.units as u
        if R is None: return None
        if isinstance(R, Angle): return R
        if hasattr(R, "unit"):   return Angle(R)          # Quantity
        return Angle(R, unit=u.deg)                       # float -> deg

    def _within_radius(cen, R, coord):
        return coord.icrs.separation(cen.icrs) <= R

    nbeam = len(ms_list)
    centers = [None] * nbeam
    keep_mask = [True] * nbeam  # default: keep
    R = _norm_radius(target_radius) if (target_center is not None and target_radius is not None) else None

    for i, ms in enumerate(ms_list):
        c = _phasecenter(ms)      # cheap FIELD read
        centers[i] = c
        if R is not None and not _within_radius(target_center, R, c):
            keep_mask[i] = False
            logger.info(f"[SKIP-READ] {os.path.basename(ms)} outside {R.to_string()} of "
                        f"{target_center.to_string('hmsdms')}")

    # ---------------- 4) Read only the kept beams (preserve positions) ----------------
    uu_list = [None] * nbeam
    vv_list = [None] * nbeam
    ww_list = [None] * nbeam
    I_list  = [None] * nbeam
    sI_list = [None] * nbeam
    fI_list = [None] * nbeam

    if n_workers and n_workers > 1:
        logger.info(f"[BLOCK] Parallel read with {n_workers} workers (order-preserving; selective)")
        ctx = get_context("fork")  # 'spawn' on Windows
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            futs = {ex.submit(_read_one_ms, ms_list[i], chan_idx, keep_autocorr, prefer_weight_spectrum): i
                    for i in range(nbeam) if keep_mask[i]}
            for fut in as_completed(futs):
                i = futs[fut]
                out = fut.result()
                uu_list[i] = out["uu"]; vv_list[i] = out["vv"]; ww_list[i] = out["ww"]
                I_list[i]  = out["I"];  sI_list[i] = out["sI"];  fI_list[i] = out["fI"]
    else:
        logger.info("[BLOCK] Serial read (order-preserving; selective)")
        for i in range(nbeam):
            if not keep_mask[i]:
                continue  # leave None -> empty slot
            out = _read_one_ms(ms_list[i], chan_idx, keep_autocorr, prefer_weight_spectrum)
            uu_list[i] = out["uu"]; vv_list[i] = out["vv"]; ww_list[i] = out["ww"]
            I_list[i]  = out["I"];  sI_list[i] = out["sI"];  fI_list[i] = out["fI"]

    # ---------------- 5) Pack to dense (nchan, nbeam, nvis_max) ----------------
    nvis = np.zeros((nbeam,), dtype=np.int32)
    for i in range(nbeam):
        nvis[i] = 0 if uu_list[i] is None else int(uu_list[i].shape[0])
    nvis_max = int(nvis.max())  # can be 0 if all beams skipped

    data_I  = np.zeros((nchan, nbeam, nvis_max), dtype=np.complex64)
    sigma_I = np.zeros((nchan, nbeam, nvis_max), dtype=np.float32)
    flag_I  = np.ones( (nchan, nbeam, nvis_max), dtype=bool)

    uu = np.zeros((nbeam, nvis_max), dtype=np.float32)
    vv = np.zeros_like(uu); ww = np.zeros_like(uu)

    # Per-channel UV mask and channel-major transpose
    for b in range(nbeam):
        if nvis[b] == 0:
            continue  # empty slot; already all flags=True
        UVW0 = uu_list[b]; UVW1 = vv_list[b]; UVW2 = ww_list[b]
        I    = I_list[b];  sI   = sI_list[b];  fI   = fI_list[b]   # (nrow_b, nchan)

        # baseline length in meters for each row
        bl_m   = np.sqrt((UVW0**2 + UVW1**2 + UVW2**2))[:, None]   # (nrow_b, 1)
        # convert to wavelengths per channel
        bl_lam = bl_m * (frequency[None, :] / c_light.value)       # (nrow_b, nchan)
        in_rng = (bl_m >= uvmin) & (bl_m <= uvmax)

        fI |= ~in_rng
        sI[~in_rng] = np.inf

        # to channel-major for this beam
        I_cb  = I.transpose(1, 0).astype(np.complex64)    # (nchan, nvis_b)
        sI_cb = sI.transpose(1, 0).astype(np.float32)     # (nchan, nvis_b)
        fI_cb = fI.transpose(1, 0).astype(bool)           # (nchan, nvis_b)

        nv = nvis[b]
        uu[b, :nv] = UVW0
        vv[b, :nv] = UVW1
        ww[b, :nv] = UVW2
        data_I [:, b, :nv] = I_cb
        sigma_I[:, b, :nv] = sI_cb
        flag_I [:, b, :nv] = fI_cb

    centers = np.asarray(centers, dtype=object)
    kept_count = int(np.count_nonzero(keep_mask)) if R is not None else nbeam
    logger.info(f"[BLOCK] Done: nchan={nchan}, nbeam={nbeam}, nvis_max={nvis_max} "
                f"(read {kept_count}/{nbeam} beams)")

    return VisIData(
        frequency=frequency,
        velocity=velocity,
        centers=centers,     # natural beam order by C10_#
        nvis=nvis,
        uu=uu, vv=vv, ww=ww, # meters (scaled to λ on demand)
        data_I=data_I,
        sigma_I=sigma_I,
        flag_I=flag_I,
    )

def read_ms_blocks_I(
    ms_root: str,
    uvmin: float = 0.0,
    uvmax: float = float("inf"),
    chan_sel=None,                      # None | slice | list[int] | np.ndarray[int]
    rest_freq: float = 1.42040575177e9, # HI rest frequency as default value in unit of Hz
    keep_autocorr: bool = False,
    prefer_weight_spectrum: bool = True,
    mode: str = "merge",                # "merge" | "stack" | "separate"
    n_workers: int = 0,                 # 0/1 = serial; >1 = parallel per-MS
    target_center: "SkyCoord | None" = None,
    target_radius: "Angle | u.Quantity | float | None" = None,
    center_tol_deg: float = 1e-12,      # tolerance for center equality when mode="merge"
    beam_sel=None, 
) -> "VisIData | List[VisIData]":
    """
    Load multiple ``blocks`` of observations located under ``ms_root``, then either:

    - ``merge``: concatenate vis per beam across blocks, assuming same beam order and centers
    - ``stack``: stack beams (Nblock × beams)
    - ``separate``: return a list of ``VisIData``, one per block

    Block discovery policy
    ----------------------
    - If ``ms_root`` itself contains ``*.ms`` directly, it's treated as a single block.
    - Any immediate subdirectory of ``ms_root`` that contains ``*.ms`` is also a block.

    Returns
    -------
    VisIData | list[VisIData]
        - ``merge``: beams equal to the number of unique centers (per order in block 0)  
        - ``stack``: beams equal to the sum of beams across blocks  
        - ``separate``: list of ``VisIData`` objects  
    """
    # ---------- discover and read blocks ----------
    block_dirs = _list_block_dirs(ms_root)
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
            beam_sel=beam_sel
        )
        blocks.append(vi)

    if mode == "separate":
        return blocks

    if mode not in ("merge", "stack"):
        raise ValueError('mode must be one of: "merge", "stack", "separate"')

    # ---------- common sanity: same spectral grid ----------
    _check_same_freq_grid(blocks)
    nchan = blocks[0].frequency.size

    # ---------- STACK: old behavior (Nblock × beams) ----------
    if mode == "stack":
        total_beams = sum(b.uu.shape[0] for b in blocks)
        global_nvis_max = max(int(b.nvis.max()) for b in blocks)

        data_I  = np.zeros((nchan, total_beams, global_nvis_max), dtype=np.complex64)
        sigma_I = np.zeros((nchan, total_beams, global_nvis_max), dtype=np.float32)
        flag_I  = np.ones( (nchan, total_beams, global_nvis_max), dtype=bool)

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
                data_I [:, dst, :nv] = blk.data_I[:, j, :nv]
                sigma_I[:, dst, :nv] = blk.sigma_I[:, j, :nv]
                flag_I [:, dst, :nv] = blk.flag_I[:, j, :nv]
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

    # ---------- MERGE: concatenate per-beam across blocks ----------
    # Sanity: same number of beams and same centers (order) across blocks
    nbeams = blocks[0].uu.shape[0]
    for bi, blk in enumerate(blocks[1:], start=1):
        if blk.uu.shape[0] != nbeams:
            raise ValueError(f"Block {bi} has {blk.uu.shape[0]} beams; expected {nbeams}.")
        for j in range(nbeams):
            c0, c1 = blocks[0].centers[j], blk.centers[j]
            if not _centers_equal(c0, c1, tol_deg=center_tol_deg):
                raise ValueError(f"Beam center mismatch at beam {j}: "
                                 f"block0={c0} vs block{bi}={c1}")

    # Merged nvis per beam = sum over blocks
    merged_nvis = np.zeros((nbeams,), dtype=np.int32)
    for j in range(nbeams):
        merged_nvis[j] = sum(int(blk.nvis[j]) for blk in blocks)
    global_nvis_max = int(merged_nvis.max())

    # Allocate outputs
    data_I  = np.zeros((nchan, nbeams, global_nvis_max), dtype=np.complex64)
    sigma_I = np.zeros((nchan, nbeams, global_nvis_max), dtype=np.float32)
    flag_I  = np.ones( (nchan, nbeams, global_nvis_max), dtype=bool)

    uu = np.zeros((nbeams, global_nvis_max), dtype=np.float32)
    vv = np.zeros_like(uu)
    ww = np.zeros_like(uu)
    nvis = merged_nvis.copy()
    centers = np.array(blocks[0].centers, dtype=object)  # preserve order

    # Concatenate per beam across blocks in read order (block0, block1, ...)
    for j in range(nbeams):
        w = 0
        for blk in blocks:
            nv = int(blk.nvis[j])
            if nv <= 0:
                continue
            uu[j, w:w+nv] = blk.uu[j, :nv]
            vv[j, w:w+nv] = blk.vv[j, :nv]
            ww[j, w:w+nv] = blk.ww[j, :nv]
            data_I [:, j, w:w+nv] = blk.data_I[:, j, :nv]
            sigma_I[:, j, w:w+nv] = blk.sigma_I[:, j, :nv]
            flag_I [:, j, w:w+nv] = blk.flag_I[:, j, :nv]
            w += nv
        # padded tail remains flag=True

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
    rest_freq: float = 1.42040575177e9, # HI rest frequency as default value in unit of Hz
    slab: int = 64,                # max channels per slab
    keep_autocorr: bool = False,
    prefer_weight_spectrum: bool = True,
    n_workers: int = 0,               # 0/1 = serial; >1 = parallel per-MS
    beam_sel=None
) -> Iterator[Tuple[int, int, VisIData]]:
    """
    Yield contiguous channel slabs so you can stream a big cube with low RAM.

    Yields:
        (start, stop, visI)
        where start/stop are absolute channel indices into the SPW (Python slice semantics),
        and visI is a VisIData with shape (stop-start, nbeam, nvis_max).
    """
    ms_list = _list_ms_sorted(ms_dir)
    all_freq, _ = _freqs(ms_list[0], rest_freq)
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
            rest_freq=rest_freq,
            keep_autocorr=keep_autocorr,
            prefer_weight_spectrum=prefer_weight_spectrum,
            n_workers=n_workers,            
            beam_sel=beam_sel
        )
        yield start, stop, visI
        i = j

def iter_blocks_chan_beam_I(
    ms_root: str,
    uvmin: float = 0.0,
    uvmax: float = float("inf"),
    chan_sel=None,
    rest_freq: float = 1.42040575177e9, # HI rest frequency as default value in unit of Hz
    keep_autocorr: bool = False,
    prefer_weight_spectrum: bool = True,
    n_workers: int = 0,               # 0/1 = serial; >1 = parallel per-MS
    beam_sel = None
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
            rest_freq=rest_freq,
            keep_autocorr=keep_autocorr,
            prefer_weight_spectrum=prefer_weight_spectrum,
            n_workers=n_workers,
            beam_sel=beam_sel
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
    n_workers: int = 0,
    concat: bool = False,  # NEW: if True, concat slabs across blocks before yielding
    beam_sel = None
):
    """
    Yield slabs for each block, or concatenated slabs if concat=True.

    If concat=False:
        Yields (bi, block_dir, c0, c1, visI) for each block slab.

    If concat=True:
        Yields (c0, c1, visI_concat) where visI_concat has all beams from all blocks
        for that channel range (like mode="concat" but streaming).
    """
    block_dirs = _list_block_dirs(ms_root)

    if not concat:
        for bi, bdir in enumerate(block_dirs):
            for c0, c1, visI in iter_channel_slabs(
                bdir,
                uvmin=uvmin,
                uvmax=uvmax,
                chan_sel=chan_sel,
                rest_freq=rest_freq,
                slab=slab,
                keep_autocorr=keep_autocorr,
                prefer_weight_spectrum=prefer_weight_spectrum,
                n_workers=n_workers,
                beam_sel=beam_sel
            ):
                yield bi, bdir, c0, c1, visI
                # caller should del visI when done
    else:
        # Accumulate slabs from each block for the same channel range
        from collections import defaultdict
        slab_accum = defaultdict(list)
        n_blocks = len(block_dirs)

        for bi, bdir in enumerate(block_dirs):
            for c0, c1, visI in iter_channel_slabs(
                bdir,
                uvmin=uvmin,
                uvmax=uvmax,
                chan_sel=chan_sel,
                rest_freq=rest_freq,
                slab=slab,
                keep_autocorr=keep_autocorr,
                prefer_weight_spectrum=prefer_weight_spectrum,
                n_workers=n_workers,
                beam_sel=beam_sel
            ):
                slab_accum[(c0, c1)].append(visI)

                # Once we have all blocks for this slab range -> concat & yield
                if len(slab_accum[(c0, c1)]) == n_blocks:
                    vis_concat = concat_visidata_slabs(slab_accum[(c0, c1)])
                    yield c0, c1, vis_concat
                    del slab_accum[(c0, c1)]


def concat_visidata_slabs(slabs: list[VisIData]) -> VisIData:
    """
    Concatenate a list of VisIData slabs (same channels, different beams)
    into one VisIData with all beams.
    """
    if not slabs:
        raise ValueError("No slabs to concatenate")

    # Frequency/velocity arrays are identical in all slabs
    freq = slabs[0].frequency
    vel  = slabs[0].velocity
    nchan = freq.shape[0]

    total_beams = sum(s.uu.shape[0] for s in slabs)
    global_nvis_max = max(int(s.nvis.max()) for s in slabs)

    data_I  = np.zeros((nchan, total_beams, global_nvis_max), dtype=np.complex64)
    sigma_I = np.zeros((nchan, total_beams, global_nvis_max), dtype=np.float32)
    flag_I  = np.ones( (nchan, total_beams, global_nvis_max), dtype=bool)

    uu = np.zeros((total_beams, global_nvis_max), dtype=np.float32)
    vv = np.zeros_like(uu)
    ww = np.zeros_like(uu)
    nvis = np.zeros((total_beams,), dtype=np.int32)
    centers = np.empty((total_beams,), dtype=object)

    b_off = 0
    for slab in slabs:
        nb = slab.uu.shape[0]
        for j in range(nb):
            nv = int(slab.nvis[j])
            dst = b_off + j
            nvis[dst] = nv
            centers[dst] = slab.centers[j]
            uu[dst, :nv] = slab.uu[j, :nv]
            vv[dst, :nv] = slab.vv[j, :nv]
            ww[dst, :nv] = slab.ww[j, :nv]
            data_I [:, dst, :nv] = slab.data_I[:, j, :nv]
            sigma_I[:, dst, :nv] = slab.sigma_I[:, j, :nv]
            flag_I [:, dst, :nv] = slab.flag_I[:, j, :nv]
        b_off += nb

    return VisIData(
        frequency=freq,
        velocity=vel,
        centers=centers,
        nvis=nvis,
        uu=uu, vv=vv, ww=ww,
        data_I=data_I,
        sigma_I=sigma_I,
        flag_I=flag_I,
    )


def iter_blocks_chan_beam_via_slabs(
    ms_root: str,
    uvmin: float = 0.0,
    uvmax: float = float("inf"),
    chan_sel=None,
    slab: int = 64,
    keep_autocorr: bool = False,
    prefer_weight_spectrum: bool = True,
    n_workers: int = 0,               # 0/1 = serial; >1 = parallel per-MS
    beam_sel = None
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
            rest_freq=rest_freq,
            slab=slab,
            keep_autocorr=keep_autocorr,
            prefer_weight_spectrum=prefer_weight_spectrum,
            beam_sel=beam_sel
        ):
            # iterate tiny chunks from the slab
            for c_rel in range(visI.data_I.shape[0]):
                c_abs = c0 + c_rel
                for b in range(visI.uu.shape[0]):
                    I, sI, uu, vv, ww = visI.slice_chan_beam_I(c_rel, b)
                    if I.size:
                        yield bi, c_abs, b, I, sI, uu, vv, ww
            del visI  # free slab memory


class CasacoreReader:
    """
    Concrete Reader backed by casacore.
    Delegates to the module-level functions defined above.
    """

    def __init__(self, *, prefer_weight_spectrum: bool = True,
                 keep_autocorr: bool = False, n_workers: int = 0):
        self.prefer_weight_spectrum = prefer_weight_spectrum
        self.keep_autocorr = keep_autocorr
        self.n_workers = n_workers

    # --- optional helpers (not required by the Protocol) ---
    def list_ms(self, ms_dir: str) -> List[str]:
        return _list_ms_sorted(ms_dir)

    def freq_grid(self, ms_dir: str, rest_freq: float):
        msl = self.list_ms(ms_dir)
        if not msl:
            raise FileNotFoundError(f"No .ms found in {ms_dir}")
        freqs, _vel = _freqs(msl[0], rest_freq)
        return freqs

    # --- Protocol methods ---
    # def read_block_I(self, ms_dir: str, **kwargs) -> VisIData:
    #     return read_ms_block_I(
    #         ms_dir,
    #         uvmin=kwargs.get("uvmin", 0.0),
    #         uvmax=kwargs.get("uvmax", float("inf")),
    #         chan_sel=kwargs.get("chan_sel"),
    #         keep_autocorr=kwargs.get("keep_autocorr", self.keep_autocorr),
    #         prefer_weight_spectrum=kwargs.get("prefer_weight_spectrum", self.prefer_weight_spectrum),
    #         n_workers=kwargs.get("n_workers", self.n_workers),
    #         target_center=kwargs.get("target_center"),
    #         target_radius=kwargs.get("target_radius"),
    #     )

    def read_blocks_I(self, ms_root: str, **kwargs) -> Union["VisIData", List["VisIData"]]:
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
        )

# ------------------------------- demo -------------------------------------

if __name__ == "__main__":
    # Example usage — adjust path + channels
    # ms_dir = "/Users/antoine/Desktop/Synthesis/ivis/docs/tutorials/data_tutorials/ivis_data/msl_mw/"
    ms_dir = "/Users/antoine/Desktop/Synthesis/ivis/docs/tutorials/data_tutorials/msdir2"

    # # single shot load (channels 0..99)
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


    # concat or merge
    print("Test #Concat all")
    I: VisIData = read_ms_blocks_I(
        ms_root=ms_dir,
        uvmin=20.0, uvmax=5000.0,
        chan_sel=slice(0, 4),
        rest_freq=1.42040575177e9, #HI rest frequency in Hz
        keep_autocorr=False,
        prefer_weight_spectrum=True,
        mode="merge",
        n_workers=4,
    )
    for c, b, Ib, sI, uu, vv, ww in I.iter_chan_beam_I():
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
    
    for c0, c1, visI in iter_blocks_channel_slabs(
            ms_root=ms_dir,
            uvmin=20,
            uvmax=5000,
            chan_sel=slice(0,128),
            slab=16,
            concat=True,
            n_workers=4,
    ):
        # Count all unflagged visibilities in this slab
        total_vis = np.count_nonzero(~visI.flag_I)
        logger.info(
            f"Concat slab [{c0}:{c1}) -> {visI.data_I.shape}, total unflagged vis={total_vis}"
        )
        
        for rel_c in range(visI.frequency.shape[0]):
            logger.info(
                f"Get single channel {rel_c} from slab [{c0}:{c1}) -> {visI.data_I.shape}"
            )
            chan_vis = visI.single_channel(rel_c, copy=False)
            
        del visI

    stop
    
    for bi, bdir, c0, c1, visI in iter_blocks_channel_slabs(
            ms_root=ms_dir,
            uvmin=20,
            uvmax=5000,
            chan_sel=slice(0,128),
            slab=16,
            # concat=True,
    ):
        logger.info(f"[block {bi}] {os.path.basename(bdir)} slab [{c0}:{c1}) -> {visI.data_I.shape}")
        for rel_c in range(visI.frequency.shape[0]):
            logger.info(f"Get single channel {rel_c} from slab [{c0}:{c1}) -> {visI.data_I.shape}")
            chan_vis = visI.single_channel(rel_c, copy=False)
        del visI


    # # Option B — Stream (block, channel, beam) inside slabs (lowest RAM)
    # print("Test # Option B — Stream (block, channel, beam) inside slabs (lowest RAM)")
    # for bi, c, b, I, sI, uu, vv, ww in iter_blocks_chan_beam_via_slabs(
    #     ms_root=ms_dir, uvmin=20, uvmax=5000, chan_sel=slice(0,8), slab=4
    # ):
    #     # NUFFT/predict/imaging for just this (block,chan,beam) slice
    #     pass

    # def read_ms_block_I_no_beam_selection_from_center_and_radius(
#     ms_dir: str,
#     uvmin: float = 0.0,               # in wavelengths (desired)
#     uvmax: float = float("inf"),      # in wavelengths (desired)
#     chan_sel=None,                    # None | slice | list[int] | np.ndarray[int]
#     keep_autocorr: bool = False,
#     prefer_weight_spectrum: bool = True,
#     n_workers: int = 0,               # 0/1 = serial; >1 = parallel per-MS
#     target_center: "SkyCoord | None" = None,
#     target_radius: "Angle | u.Quantity | float | None" = None,  # float ⇒ degrees
# ) -> "VisIData":
#     """
#     Load a directory of .ms files (one per beam) into an I-only, channel-major VisIData.
#     The beams are packed in a **deterministic, natural order by beam index** inferred
#     from filenames like '...-C10_5_...'. This matches what humans expect from a
#     sorted listing (1,2,3,4,5,...), and is preserved under parallel reads.
#     """
#     # 1) Deterministic, human-expected order
#     ms_list = _list_ms_sorted(ms_dir)
#     logger.info(f"[BLOCK] Loading {len(ms_list)} beam(s) from: {ms_dir}")

#     # --- NEW: center+radius filter ---
#     if (target_center is not None) and (target_radius is not None):
#         R = _norm_radius(target_radius)
#         kept, skipped = [], []
#         for ms in ms_list:
#             c = _phasecenter(ms)  # cheap FIELD read
#             if _within_radius(target_center, R, c):
#                 kept.append(ms)
#             else:
#                 skipped.append(ms)
#         for s in skipped:
#             logger.info(f"[SKIP] {os.path.basename(s)}: outside {R.to_string()} of center")
#         ms_list = kept
#         if not ms_list:
#             raise ValueError(f"No beams within {R.to_string()} of {target_center.to_string('hmsdms')}")

#     # 2) Channel selection
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

#     frequency = all_freq[chan_idx]     # (nchan,)
#     velocity  = all_vel [chan_idx]     # (nchan,)
#     nchan = int(frequency.size)

#     # 3) Read per-MS into fixed slots (preserve the chosen order)
#     nbeam = len(ms_list)
#     centers = [None] * nbeam
#     uu_list = [None] * nbeam
#     vv_list = [None] * nbeam
#     ww_list = [None] * nbeam
#     I_list  = [None] * nbeam
#     sI_list = [None] * nbeam
#     fI_list = [None] * nbeam

#     if n_workers and n_workers > 1:
#         logger.info(f"[BLOCK] Parallel read with {n_workers} workers (order-preserving)")
#         ctx = get_context("fork")  # 'spawn' on Windows
#         with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
#             futs = {ex.submit(_read_one_ms, ms, chan_idx, keep_autocorr, prefer_weight_spectrum): i
#                     for i, ms in enumerate(ms_list)}
#             for fut in as_completed(futs):
#                 i = futs[fut]
#                 out = fut.result()
#                 centers[i] = out["center"]
#                 uu_list[i] = out["uu"]; vv_list[i] = out["vv"]; ww_list[i] = out["ww"]
#                 I_list[i]  = out["I"];  sI_list[i] = out["sI"];  fI_list[i] = out["fI"]
#     else:
#         logger.info("[BLOCK] Serial read (order-preserving)")
#         for i, ms in enumerate(ms_list):
#             out = _read_one_ms(ms, chan_idx, keep_autocorr, prefer_weight_spectrum)
#             centers[i] = out["center"]
#             uu_list[i] = out["uu"]; vv_list[i] = out["vv"]; ww_list[i] = out["ww"]
#             I_list[i]  = out["I"];  sI_list[i] = out["sI"];  fI_list[i] = out["fI"]

#     # 4) Pack to dense (nchan, nbeam, nvis_max)
#     nvis = np.array([u.shape[0] for u in uu_list], dtype=np.int32)
#     nvis_max = int(nvis.max())

#     data_I  = np.zeros((nchan, nbeam, nvis_max), dtype=np.complex64)
#     sigma_I = np.zeros((nchan, nbeam, nvis_max), dtype=np.float32)
#     flag_I  = np.ones( (nchan, nbeam, nvis_max), dtype=bool)

#     uu = np.zeros((nbeam, nvis_max), dtype=np.float32)
#     vv = np.zeros_like(uu); ww = np.zeros_like(uu)

#     # Per-channel UV mask and channel-major transpose
#     for b in range(nbeam):
#         UVW0 = uu_list[b]; UVW1 = vv_list[b]; UVW2 = ww_list[b]
#         I    = I_list[b];  sI   = sI_list[b];  fI   = fI_list[b]   # (nrow_b, nchan)

#         # baseline length in meters for each row
#         bl_m   = np.sqrt((UVW0**2 + UVW1**2 + UVW2**2))[:, None]   # (nrow_b, 1)
#         # convert to wavelengths per channel
#         bl_lam = bl_m * (frequency[None, :] / c_light.value)       # (nrow_b, nchan)
#         in_rng = (bl_lam >= uvmin) & (bl_lam <= uvmax)

#         fI |= ~in_rng
#         sI[~in_rng] = np.inf

#         # to channel-major for this beam
#         I_cb  = I.transpose(1, 0).astype(np.complex64)    # (nchan, nvis_b)
#         sI_cb = sI.transpose(1, 0).astype(np.float32)     # (nchan, nvis_b)
#         fI_cb = fI.transpose(1, 0).astype(bool)           # (nchan, nvis_b)

#         nv = int(UVW0.shape[0])
#         uu[b, :nv] = UVW0
#         vv[b, :nv] = UVW1
#         ww[b, :nv] = UVW2
#         data_I [:, b, :nv] = I_cb
#         sigma_I[:, b, :nv] = sI_cb
#         flag_I [:, b, :nv] = fI_cb

#     centers = np.asarray(centers, dtype=object)
#     logger.info(f"[BLOCK] Done: nchan={nchan}, nbeam={nbeam}, nvis_max={nvis_max}")

#     return VisIData(
#         frequency=frequency,
#         velocity=velocity,
#         centers=centers,     # natural beam order by C10_#
#         nvis=nvis,
#         uu=uu, vv=vv, ww=ww, # meters (scaled to λ on demand)
#         data_I=data_I,
#         sigma_I=sigma_I,
#         flag_I=flag_I,
#     )

# def read_ms_block_I_no_parrallel(
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
#     ms_list = _list_ms_sorted(ms_dir)

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

# def read_ms_blocks_I(
#     ms_root: str,
#     uvmin: float = 0.0,
#     uvmax: float = float("inf"),
#     chan_sel=None,                   # None | slice | list[int] | np.ndarray[int]
#     keep_autocorr: bool = False,
#     prefer_weight_spectrum: bool = True,
#     mode: str = "concat",            # "concat" | "separate"
#     n_workers: int = 0,               # 0/1 = serial; >1 = parallel per-MS    
# ) -> "VisIData | list[VisIData]":
#     """
#     Load multiple blocks of observations located under ms_root.

#     - If ms_root contains *.ms directly, it's treated as a single block.
#     - Any immediate subdirectory of ms_root that contains *.ms is also a block.

#     mode="concat":  returns one VisIData with beams from all blocks concatenated
#     mode="separate": returns a list[VisIData], one per block
#     """
#     block_dirs = _list_block_dirs(ms_root)

#     # Load each block independently with your optimized reader
#     blocks: list[VisIData] = []
#     for bdir in block_dirs:
#         logger.info(f"[BLOCK] Loading block from: {bdir}")
#         vi = read_ms_block_I(
#             bdir,
#             uvmin=uvmin,
#             uvmax=uvmax,
#             chan_sel=chan_sel,
#             keep_autocorr=keep_autocorr,
#             prefer_weight_spectrum=prefer_weight_spectrum,
#             n_workers=n_workers,
#         )
#         blocks.append(vi)

#     if mode == "separate":
#         return blocks

#     if mode != "concat":
#         raise ValueError('mode must be "concat" or "separate"')

#     # ---------------- concat path ----------------
#     _check_same_freq_grid(blocks)  # ensure same channels across blocks

#     # Concatenate beams; keep per-beam nvis; pad to global nvis_max
#     nchan = blocks[0].frequency.size
#     total_beams = sum(b.uu.shape[0] for b in blocks)
#     global_nvis_max = max(int(b.nvis.max()) for b in blocks)

#     # Allocate output
#     data_I  = np.zeros((nchan, total_beams, global_nvis_max), dtype=np.complex64)
#     sigma_I = np.zeros((nchan, total_beams, global_nvis_max), dtype=np.float32)
#     flag_I  = np.ones( (nchan, total_beams, global_nvis_max), dtype=bool)

#     uu = np.zeros((total_beams, global_nvis_max), dtype=np.float32)
#     vv = np.zeros_like(uu); ww = np.zeros_like(uu)
#     nvis = np.zeros((total_beams,), dtype=np.int32)
#     centers = np.empty((total_beams,), dtype=object)

#     # Copy block by block
#     b_off = 0
#     for blk in blocks:
#         nb = blk.uu.shape[0]
#         for j in range(nb):
#             nv = int(blk.nvis[j])
#             dst = b_off + j
#             nvis[dst] = nv
#             centers[dst] = blk.centers[j]
#             uu[dst, :nv] = blk.uu[j, :nv]
#             vv[dst, :nv] = blk.vv[j, :nv]
#             ww[dst, :nv] = blk.ww[j, :nv]
#             data_I [:, dst, :nv] = blk.data_I[:, j, :nv]
#             sigma_I[:, dst, :nv] = blk.sigma_I[:, j, :nv]
#             flag_I [:, dst, :nv] = blk.flag_I[:, j, :nv]
#             # padded tails remain flagged=True
#         b_off += nb

#     return VisIData(
#         frequency=blocks[0].frequency,
#         velocity=blocks[0].velocity,
#         centers=centers,
#         nvis=nvis,
#         uu=uu, vv=vv, ww=ww,
#         data_I=data_I,
#         sigma_I=sigma_I,
#         flag_I=flag_I,
#     )
