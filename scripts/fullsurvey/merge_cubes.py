from astropy.io import fits
import numpy as np

def mergeN(files, out):
    hs = [fits.open(f, memmap=True) for f in files]
    try:
        datas = [np.float32(h[0].data) for h in hs]
        hdr = hs[0][0].header.copy()
        hdr["NAXIS3"] = sum(d.shape[0] for d in datas)
        fits.writeto(out, np.concatenate(datas, axis=0), hdr, overwrite=True)
    finally:
        for h in hs: h.close()
    return out

path="/totoro/anmarchal/data/gaskap/fullsurvey/products/merge/"
files = [
  path+"output_chan_735_30_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_Nw_0.fits",
  path+"output_chan_765_30_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_Nw_0.fits",
  path+"output_chan_795_30_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_Nw_0.fits",
  path+"output_chan_825_30_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_Nw_0.fits",
  path+"output_chan_855_30_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_Nw_0.fits",
]
mergeN(files, path+"output_merged_all.fits")
