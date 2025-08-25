# ivis/types.py
from __future__ import annotations
from dataclasses import dataclass, fields, MISSING
from typing import Iterator, Tuple
import numpy as np
from astropy.constants import c as c_light

# -------------------- I-only container (channel-major) --------------------
@dataclass
class VisIData():
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
    data_I:  np.ndarray       # (nchan, nbeam, nvis_max) complex64 [Jy]
    sigma_I: np.ndarray       # (nchan, nbeam, nvis_max) float32 [Jy]
    flag_I:  np.ndarray       # (nchan, nbeam, nvis_max) bool


    _units = {
        "frequency": "[Hz]",
        "velocity":  "[km/s]",
        "centers":   "",
        "nvis":      "",
        "uu":        "[m]",
        "vv":        "[m]",
        "ww":        "[m]",
        "data_I":    "[Jy]",
        "sigma_I":   "[Jy]",
        "flag_I":    "",
    }
    
    def __repr__(self):
        lines = [f"{self.__class__.__name__}("]
        for f in fields(self):
            val = getattr(self, f.name)
            unit = self._units.get(f.name, "")
            if isinstance(val, np.ndarray):
                flat = val.ravel()
                preview = np.array2string(flat[:3], separator=", ", threshold=5)
                if flat.size > 3:
                    preview = preview[:-1] + ", ...]"
                lines.append(
                    f"  {f.name:<10} {unit:<7}: array{val.shape}, dtype={val.dtype}, sample={preview}"
                )
            else:
                lines.append(f"  {f.name:<10} {unit:<7}: {val!r}")
        lines.append(")")
        return "\n".join(lines)


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


    def single_channel(self, c: int, copy: bool = False) -> "VisIData":
        """
        Return a VisIData containing only channel c (shape (1, nbeam, nvis_max)).
        By default returns views (no copy); set copy=True for independent arrays.
        """
        # channel slice
        cs = slice(c, c+1)
        
        def maybe(a):
            return a[cs].copy() if copy else a[cs]
        
        return VisIData(
            frequency=self.frequency[cs].copy() if copy else self.frequency[cs],
            velocity=self.velocity[cs].copy()   if copy else self.velocity[cs],
            centers=self.centers,                  # same object array (beams)
            nvis=self.nvis.copy() if copy else self.nvis,
            uu=self.uu.copy() if copy else self.uu,
            vv=self.vv.copy() if copy else self.vv,
            ww=self.ww.copy() if copy else self.ww,
            data_I=maybe(self.data_I),
            sigma_I=maybe(self.sigma_I),
            flag_I=maybe(self.flag_I),
        )
    
    def iter_single_channel(self, copy: bool = False):
        """
        Iterate over channels, yielding (c, VisIData_with_only_that_channel).
        """
        for c in range(self.frequency.shape[0]):
            yield c, self.single_channel(c, copy=copy)

