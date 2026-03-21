            # def process(self, model=None, units="Jy/arcsec^2", disk=False):
    #     """
    #     Run imaging optimization.
    #     """
    #     if model is None:
    #         raise ValueError("Must pass a model instance to `process()`.")
    #     if not hasattr(model, "loss"):
    #         raise TypeError("Model must implement `.loss()`.")

    #     # --- Image/grid params ---
    #     cell_size = (self.hdr["CDELT2"] * u.deg).to(u.arcsec)
    #     shape = (self.hdr["NAXIS2"], self.hdr["NAXIS1"])
    #     tapper = dutils.apodize(0.98, shape)

    #     # --- FFT beam for reg ---
    #     kernel_map = dutils.laplacian(shape)
    #     fftkernel = abs(fft2(kernel_map))

    #     bmaj_pix = self.beam_sd.major.to(u.deg).value / cell_size.to(u.deg).value
    #     beam = dutils.gauss_beam(bmaj_pix, shape, FWHM=True)
    #     fftbeam = abs(fft2(beam))

    #     # --- FFT single-dish ---
    #     fftsd = cell_size.value**2 * tfft2(torch.from_numpy(np.float32(self.sd))).numpy()

    #     # --- Bounds ---
    #     param_shape = self.init_params.shape
    #     if self.positivity:
    #         bounds = dutils.ROHSA_bounds(param_shape, lb_amp=0, ub_amp=np.inf)
    #     else:
    #         bounds = dutils.ROHSA_bounds(param_shape, lb_amp=-np.inf, ub_amp=np.inf)

    #     # --- Precompute params dict ---
    #     params = dict(
    #         vis_data=self.vis_data,
    #         pb=np.asarray(self.pb, dtype=np.float32),
    #         fftbeam=np.asarray(fftbeam, dtype=np.float32),
    #         fftsd=np.asarray(fftsd, dtype=np.complex64),
    #         tapper=np.asarray(tapper, dtype=np.float32),
    #         lambda_sd=self.lambda_sd,
    #         fftkernel=np.asarray(fftkernel, dtype=np.float32),
    #         cell_size=cell_size.value,
    #         grid_array=np.asarray(self.grid, dtype=np.float32),
    #         beam_workers=self.beam_workers
    #     )

    #     device = self.device

    #     # --- Closure for optimizer ---
    #     def objective_flat(x):
    #         # Pass through **params so loss() sees vis_data, pb, fftbeam, etc.
    #         return model.loss(x, shape=param_shape, device=device, **params)

    #     # --- Optimize ---
    #     options = dict(maxiter=self.max_its, maxfun=int(1e6), iprint=25)

    #     logger.info("Starting optimisation (using LBFGS-B)")
    #     opt_output = optimize.minimize(
    #         objective_flat,
    #         self.init_params.ravel().astype(np.float32),
    #         jac=True,
    #         tol=1e-8,
    #         bounds=bounds,
    #         method="L-BFGS-B",
    #         options=options
    #     )

    #     result = np.reshape(opt_output.x, self.init_params.shape)
    #     logger.warning("Multiply by 2 for ASKAP if needed.")

    #     # --- Unit conversion ---
    #     if units == "Jy/arcsec^2":
    #         return result
    #     elif units == "Jy/beam":
    #         beam_r = Beam(4.2857 * cell_size, 4.2857 * cell_size, 1.e-12 * u.deg)
    #         return result * beam_r.sr.to(u.arcsec**2).value
    #     elif units == "K":
    #         nu = self.vis_data.frequency[0] * u.Hz
    #         beam_r = Beam(3 * cell_size, 3 * cell_size, 1.e-12 * u.deg)
    #         result_Jy = result * beam_r.sr.to(u.arcsec**2).value
    #         return (result_Jy * u.Jy).to(u.K, u.brightness_temperature(nu, beam_r)).value
    #     else:
    #         logger.error("Unknown unit type.")
    #         return result


    # def process(self, model=None, units="Jy/arcsec^2",
    #         history_size=10, dtype=torch.float32,
    #         down_factor=4, coarse_its=25, fine_its=5):
    # import numpy as np, torch
    # import torch.nn.functional as F
    # if model is None: raise ValueError("Must pass a model instance to `process()`.")
    
    # def f32c(a):
    #     arr = np.asarray(a, dtype=np.float32)            # allow copy if needed (NumPy 2.0 safe)
    #     if not arr.flags.c_contiguous:
    #         arr = np.ascontiguousarray(arr)              # ensure C-contiguous for torch.from_numpy
    #     return arr
    
    # def resize_img(x, hw):  # x: (H,W) or (C,H,W) -> same rank with new (H,W)
    #     t = torch.from_numpy(f32c(x))
    #     if t.ndim == 2:
    #         return F.interpolate(t[None,None], size=hw, mode="bilinear", align_corners=True)[0,0].numpy()
    #     if t.ndim == 3:
    #         return F.interpolate(t[:,None], size=hw, mode="bilinear", align_corners=True)[:,0].numpy()
    #     raise ValueError(f"img rank {t.ndim}")
    
    # def grid_to_nchw(g):  # -> (N,2,H,W) and tag
    #     g = f32c(g)
    #     if g.ndim == 3 and g.shape[-1]==2:   return torch.from_numpy(g).permute(2,0,1)[None], ("hw2", g.shape)
    #     if g.ndim == 4 and g.shape[-1]==2:   return torch.from_numpy(g).permute(0,3,1,2),     ("bhw2", g.shape)
    #     if g.ndim == 5 and g.shape[1]==1 and g.shape[-1]==2:
    #         gs = np.squeeze(g,1);            return torch.from_numpy(gs).permute(0,3,1,2),     ("n1hw2", g.shape)
    #     if g.ndim == 4 and g.shape[1]==2:    return torch.from_numpy(g),                      ("n2hw", g.shape)
    #     raise ValueError(f"grid shape {g.shape}")
    
    # def nchw_to_grid(t, tag):  # (N,2,H,W) -> original rank; ensure batch=1 for hw2
    #     kind,_ = tag
    #     if kind=="hw2":   return t.permute(0,2,3,1)[0].cpu().numpy()[None,...]
    #     if kind=="bhw2":  return t.permute(0,2,3,1).cpu().numpy()
    #     if kind=="n1hw2": return t.permute(0,2,3,1).cpu().numpy()[:,None,...]
    #     if kind=="n2hw":  return t.cpu().numpy()
    #     raise ValueError(f"tag {kind}")
    
    # def resize_grid(g, hw):
    #     t,tag = grid_to_nchw(g)
    #     s = F.interpolate(t.float(), size=hw, mode="bilinear", align_corners=True)
    #     return nchw_to_grid(s, tag)
    
    # init = f32c(self.init_params)
    # H,W = init.shape[-2], init.shape[-1]
    # Hc,Wc = max(1,H//down_factor), max(1,W//down_factor)
    
    # hdr0, pb0, grid0, init0, its0 = self.hdr, self.pb, self.grid, self.init_params, self.max_its
    
    # # downsample
    # init_ds = resize_img(init, (Hc,Wc))
    # pb_ds   = [resize_img(p,(Hc,Wc)) for p in (pb0 if isinstance(pb0,(list,tuple)) else [pb0])]
    # grid_ds = [resize_grid(g,(Hc,Wc)) for g in (grid0 if isinstance(grid0,(list,tuple)) else [grid0])]
    
    # # coarse
    # self.hdr = dutils.downsample_hdr(hdr0, down_factor)
    # self.pb  = pb_ds if isinstance(pb0,(list,tuple)) else pb_ds[0]
    # self.grid= grid_ds if isinstance(grid0,(list,tuple)) else grid_ds[0]
    # self.init_params = init_ds
    # self.max_its = coarse_its
    # coarse = self.process_standalone(model=model, units="Jy/arcsec^2", history_size=history_size, dtype=dtype)
    
    # # upsample to full
    # ct = torch.from_numpy(f32c(coarse))
    # if ct.ndim==2:
    #     init_up = F.interpolate(ct[None,None], size=(H,W), mode="bilinear", align_corners=True)[0,0].cpu().numpy()
    # else:
    #     init_up = F.interpolate(ct[:,None], size=(H,W), mode="bilinear", align_corners=True)[:,0].cpu().numpy()
    
    # # fine
    # self.hdr, self.pb, self.grid = hdr0, pb0, grid0
    # self.init_params, self.max_its = init_up, fine_its
    # out = self.process_standalone(model=model, units=units, history_size=history_size, dtype=dtype)
    
    # self.max_its = its0
    # return out


