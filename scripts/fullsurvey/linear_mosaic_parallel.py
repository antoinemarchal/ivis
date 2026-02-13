import os
import glob
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===============================
# PATHS / PARAMS (edit if needed)
# ===============================
path = "/totoro/anmarchal/data/gaskap/fullsurvey/products/merge/"
path_beams = "/totoro/anmarchal/data/gaskap/fullsurvey/holography_beams/merge/"

TEMPLATE = path+"output_chan_765_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_new_PB_Nw_0.fits"
OUT_MOSAIC = path+"output_chan_795_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_LINEAR.fits"
OUT_PBEFF  = path+"output_chan_795_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_LINEAR.fits"

IMG_GLOB = path+"output_chan_795_linear/*.fits"
PB_GLOB  = path_beams+"*.fits"

PB_MIN = 0.0
NPROC  = 12


def _chunkify(lst, n):
    n = max(1, int(n))
    return [lst[i::n] for i in range(n)]


def worker_chunk(args):
    """
    Process a chunk of tiles and return partial accumulators.
    Returns ONCE per worker (num, wsum) to keep IPC/memory stable.
    """
    img_list, pb_list, hdr_out, shape_out, pb_min = args

    # avoid oversubscription inside each process
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    wcs_out = WCS(hdr_out).celestial
    ny, nx = shape_out

    num  = np.zeros((ny, nx), dtype=np.float64)
    wsum = np.zeros((ny, nx), dtype=np.float64)

    for img_fn, pb_fn in zip(img_list, pb_list):

        # --- image: take [0] exactly like your script
        with fits.open(img_fn, memmap=True) as hi:
            img2d = hi[0].data[0]
            img_hdu = fits.PrimaryHDU(img2d, WCS(hi[0].header).celestial.to_header())
            img, fpi = reproject_interp(
                img_hdu, wcs_out, shape_out=shape_out, return_footprint=True
            )

        # --- PB: already 2D in your case
        with fits.open(pb_fn, memmap=True) as hp:
            pb2d = hp[0].data
            pb_hdu = fits.PrimaryHDU(pb2d, WCS(hp[0].header).celestial.to_header())
            pb, fpb = reproject_interp(
                pb_hdu, wcs_out, shape_out=shape_out, return_footprint=True
            )

        ok = (fpi > 0) & (fpb > 0)
        if pb_min > 0:
            ok &= (pb >= pb_min)

        w = pb * pb
        num[ok]  += w[ok] * img[ok]
        wsum[ok] += w[ok]

    return num, wsum


def main():
    print("=== linear_mosaic_parallel.py ===", flush=True)

    imgs = sorted(glob.glob(IMG_GLOB))
    pbs  = sorted(glob.glob(PB_GLOB))
    if len(imgs) == 0:
        raise RuntimeError(f"No images found: {IMG_GLOB}")
    if len(pbs) == 0:
        raise RuntimeError(f"No PB files found: {PB_GLOB}")
    if len(imgs) != len(pbs):
        raise RuntimeError(f"len(imgs)={len(imgs)} != len(pbs)={len(pbs)} (check sorting)")

    print(f"Found {len(imgs)} tiles", flush=True)

    # --- target grid (2D)
    with fits.open(TEMPLATE, memmap=True) as h:
        hdr_out = h[0].header
        shape_out = h[0].data.shape[-2:]

    ny, nx = shape_out
    print(f"Output grid: {ny} x {nx}", flush=True)
    print(f"NPROC={NPROC} PB_MIN={PB_MIN}", flush=True)

    # split work into NPROC chunks (skip empties)
    img_chunks = _chunkify(imgs, NPROC)
    pb_chunks  = _chunkify(pbs,  NPROC)
    jobs = [(ic, pc, hdr_out, shape_out, PB_MIN) for ic, pc in zip(img_chunks, pb_chunks) if len(ic) > 0]

    num_tot  = np.zeros((ny, nx), dtype=np.float64)
    wsum_tot = np.zeros((ny, nx), dtype=np.float64)

    with ProcessPoolExecutor(max_workers=min(NPROC, len(jobs))) as ex:
        futures = [ex.submit(worker_chunk, j) for j in jobs]
        done = 0
        for f in as_completed(futures):
            num, wsum = f.result()
            num_tot  += num
            wsum_tot += wsum
            done += 1
            print(f"Completed {done}/{len(futures)} workers", flush=True)

    # --- outputs
    mosaic = np.full((ny, nx), np.nan, dtype=np.float32)
    good = wsum_tot > 0
    mosaic[good] = (num_tot[good] / wsum_tot[good]).astype(np.float32)

    # remove nans like your script
    mosaic[~np.isfinite(mosaic)] = 0.0

    pb_eff = np.zeros((ny, nx), dtype=np.float32)
    pb_eff[good] = np.sqrt(wsum_tot[good]).astype(np.float32)

    fits.writeto(OUT_MOSAIC, mosaic, hdr_out, overwrite=True)
    fits.writeto(OUT_PBEFF,  pb_eff, hdr_out, overwrite=True)

    print("Saved:", OUT_MOSAIC, OUT_PBEFF, flush=True)


if __name__ == "__main__":
    main()
