#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from ivis.models.operators.reprojection import backward_reprojection_manual


def combine_effective_pb(
    pb_cube: np.ndarray,
    grid_cube: np.ndarray,
    image_shape: tuple[int, int],
    mode: str,
) -> np.ndarray:
    if pb_cube.ndim != 3:
        raise ValueError(
            f"Expected pb cube with shape (nbeam, ny, nx), got {pb_cube.shape}"
        )
    if grid_cube.ndim != 5 or grid_cube.shape[1] != 1 or grid_cube.shape[-1] != 2:
        raise ValueError(
            "Expected grid cube with shape (nbeam, 1, ny, nx, 2), "
            f"got {grid_cube.shape}"
        )
    if pb_cube.shape[0] != grid_cube.shape[0]:
        raise ValueError(
            f"Beam count mismatch: pb has {pb_cube.shape[0]}, grid has {grid_cube.shape[0]}"
        )

    device = "cpu"
    cache_store: dict[object, object] = {}
    accum = np.zeros(image_shape, dtype=np.float32)

    for i in range(pb_cube.shape[0]):
        pb_local = np.nan_to_num(pb_cube[i].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        back = backward_reprojection_manual(
            z2d=pb_local,
            grid=grid_cube[i],
            image_shape=image_shape,
            device=device,
            cache_store=cache_store,
        )
        back_np = back.real.detach().cpu().numpy().astype(np.float32)
        if mode == "mean":
            accum += back_np
        elif mode == "rss":
            accum += back_np**2
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    if mode == "mean":
        effpb = accum / float(pb_cube.shape[0])
    else:
        effpb = np.sqrt(accum)

    peak = float(np.nanmax(effpb))
    if peak <= 0.0:
        raise ValueError("Effective primary beam has non-positive peak; cannot normalize.")
    return effpb / peak


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mosaic local primary beams onto a larger target header using grid_interp.fits, "
            "then write a normalized effective primary beam."
        )
    )
    parser.add_argument("pb_fits", help="Input local PB cube, e.g. reproj_pb.fits")
    parser.add_argument("grid_fits", help="Input interpolation grid cube, e.g. grid_interp.fits")
    parser.add_argument(
        "template_fits",
        help="FITS file whose header defines the large target map to write onto",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="effective_pb_large.fits",
        help="Output FITS filename (default: effective_pb_large.fits)",
    )
    parser.add_argument(
        "--mode",
        choices=("mean", "rss"),
        default="mean",
        help="Combine back-projected beams with mean or rss. Default: mean",
    )
    return parser.parse_args()


def extract_celestial_template_header(template_path: Path) -> tuple[fits.Header, tuple[int, int]]:
    with fits.open(template_path) as hdul:
        header = hdul[0].header.copy()

    if "NAXIS1" not in header or "NAXIS2" not in header:
        raise ValueError(f"Template FITS {template_path} does not define NAXIS1/NAXIS2.")

    image_shape = (int(header["NAXIS2"]), int(header["NAXIS1"]))

    # Keep only the celestial WCS when the template is a cube.
    celestial_header = WCS(header).celestial.to_header()
    out_header = fits.Header()
    out_header["NAXIS"] = 2
    out_header["NAXIS1"] = image_shape[1]
    out_header["NAXIS2"] = image_shape[0]
    for key, value in celestial_header.items():
        out_header[key] = value

    return out_header, image_shape


def main() -> None:
    args = parse_args()

    with fits.open(Path(args.pb_fits)) as hdul:
        pb_cube = hdul[0].data

    with fits.open(Path(args.grid_fits)) as hdul:
        grid_cube = hdul[0].data

    header, image_shape = extract_celestial_template_header(Path(args.template_fits))

    effpb = combine_effective_pb(pb_cube, grid_cube, image_shape=image_shape, mode=args.mode)

    fits.PrimaryHDU(effpb.astype(np.float32), header=header).writeto(
        Path(args.output), overwrite=True
    )
    print(f"Wrote normalized mosaiced effective PB to {args.output}")


if __name__ == "__main__":
    main()
