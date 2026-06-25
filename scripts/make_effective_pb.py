#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits


def build_effective_pb(pb_cube: np.ndarray, mode: str) -> np.ndarray:
    if pb_cube.ndim != 3:
        raise ValueError(
            f"Expected a 3D primary beam cube with shape (nbeam, ny, nx), got {pb_cube.shape}"
        )

    pb_cube = np.nan_to_num(pb_cube.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if mode == "mean":
        effpb = np.mean(pb_cube, axis=0)
    elif mode == "rss":
        effpb = np.sqrt(np.sum(pb_cube**2, axis=0))
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    peak = float(np.nanmax(effpb))
    if peak <= 0.0:
        raise ValueError("Effective primary beam has non-positive peak; cannot normalize.")

    return effpb / peak


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a normalized effective primary beam from a beam cube already "
            "reprojected onto the target header."
        )
    )
    parser.add_argument("input_fits", help="Input FITS beam cube, e.g. reproj_pb.fits")
    parser.add_argument(
        "-o",
        "--output",
        default="effective_pb.fits",
        help="Output FITS filename (default: effective_pb.fits)",
    )
    parser.add_argument(
        "--mode",
        choices=("rss", "mean"),
        default="mean",
        help="Combination mode: mean=average(pb), rss=sqrt(sum(pb^2)). Default: mean",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_fits)
    output_path = Path(args.output)

    with fits.open(input_path) as hdul:
        pb_cube = hdul[0].data
        header = hdul[0].header.copy()

    effpb = build_effective_pb(pb_cube, mode=args.mode)

    header["NAXIS"] = 2
    for key in ("NAXIS3", "CTYPE3", "CRPIX3", "CRVAL3", "CDELT3", "CUNIT3"):
        if key in header:
            del header[key]

    fits.PrimaryHDU(effpb.astype(np.float32), header=header).writeto(output_path, overwrite=True)
    print(f"Wrote normalized effective PB to {output_path}")


if __name__ == "__main__":
    main()
