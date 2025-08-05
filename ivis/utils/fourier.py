# -*- coding: utf-8 -*-
"""
Power Spectrum and Image Processing Utilities
---------------------------------------------

This module provides tools to analyze the spatial frequency content of 2D images,
including power spectrum estimation, cross-spectra, autocorrelation functions,
and image preprocessing utilities such as apodization and padding.

Originally adapted from a package by J.-F. Robitaille.

Functions
---------

- ``powspec(image, reso=1, autocorr=False, **kwargs)``
    Computes the azimuthally averaged power spectrum of a 2D image, optionally returning
    the autocorrelation or cross-spectrum if a second image is passed via ``im2``.

- ``cross_spec(image, image2, reso=1)``
    Computes the cross-power spectrum between two 2D images.

- ``kgrid(image)``
    Returns a normalized radial spatial frequency grid (k) corresponding to the FFT bins
    for a given image shape.

- ``apodize(radius, shape)``
    Returns a 2D cosine taper mask to apodize the edges of an image.

- ``apodize_1d(radius, shape)``
    Returns a 1D cosine taper to apodize the edges of a 1D array (e.g. spectrum).

- ``padding(input, fact)``
    Pads a 2D image symmetrically by a given fractional amount (``fact``), centering
    the original image in a larger zero-filled array.

- ``depad(input, fact)``
    Crops a previously padded 2D image, removing the borders added by ``padding``.

Author: Adapted from JF Robitaille package by Antoine Marchal (mostly added docstrings for RTD). 
"""

import numpy as np
import torch

def powspec_torch(image, pixel_size_arcsec=1.0, nbins=None, autocorr=False, im2=None):
    device = image.device
    H, W = image.shape
    if nbins is None:
        nbins = max(H, W) // 2

    dx_arcmin = pixel_size_arcsec / 60.0
    fx = torch.fft.fftshift(torch.fft.fftfreq(W, d=dx_arcmin)).to(device)
    fy = torch.fft.fftshift(torch.fft.fftfreq(H, d=dx_arcmin)).to(device)
    kx, ky = torch.meshgrid(fx, fy, indexing='xy')
    k_radius = torch.sqrt(kx**2 + ky**2)

    ft1 = torch.fft.fft2(image)
    if im2 is not None:
        ft2 = torch.fft.fft2(im2)
        ps2d = torch.real(ft1 * torch.conj(ft2)) / (H * W)
    else:
        ps2d = torch.real(ft1 * torch.conj(ft1)) / (H * W)

    if autocorr:
        ps2d = torch.fft.ifft2(ps2d).real

    ps_flat = torch.fft.fftshift(ps2d).flatten()
    k_flat = k_radius.flatten()

    kmin = k_flat[k_flat > 0].min()
    kmax = k_flat.max()
    bins = torch.linspace(kmin, kmax, nbins + 1, device=device)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    inds = torch.bucketize(k_flat, bins) - 1

    power_1d = torch.zeros(nbins, device=device)
    for i in range(nbins):
        mask = inds == i
        if torch.any(mask):
            power_1d[i] = ps_flat[mask].mean()

    return bin_centers, power_1d


def powspec(image, reso=1, autocorr=False, **kwargs):
    """
    Compute the azimuthally averaged 1D power spectrum or autocorrelation of a 2D image.

    Parameters
    ----------
    image : ndarray
        Input 2D image.
    reso : float, optional
        Pixel size in spatial units (e.g. arcmin).
    autocorr : bool, optional
        If True, return the autocorrelation function instead of the power spectrum.
    im2 : ndarray, optional (passed via kwargs)
        Second image for computing the cross-spectrum.

    Returns
    -------
    tab_k : ndarray
        Radial spatial frequency values.
    spec_k : ndarray
        Azimuthally averaged power spectrum.
    """
    na = float(image.shape[1])
    nb = float(image.shape[0])
    nf = np.max(np.array([na, nb]))

    k_crit = nf / 2
    bins = np.arange(k_crit + 1)

    imft = np.fft.fft2(image)

    if 'im2' in kwargs:
        im2 = kwargs.get('im2')
        im2ft = np.fft.fft2(im2)
        ps2D = imft * np.conj(im2ft) / (na * nb)
    else:
        ps2D = np.abs(imft) ** 2 / (na * nb)

    del imft

    if autocorr:
        ps2D = np.fft.ifft2(ps2D).real

    x, y = np.meshgrid(np.arange(na), np.arange(nb))

    if na % 2 == 0:
        x = (x - (na / 2)) / na
        shiftx = na / 2
    else:
        x = (x - (na - 1) / 2) / na
        shiftx = (na - 1) / 2 + 1

    if nb % 2 == 0:
        y = (y - (nb / 2)) / nb
        shifty = nb / 2
    else:
        y = (y - (nb - 1) / 2) / nb
        shifty = (nb - 1) / 2 + 1

    k_mat = np.sqrt(x ** 2 + y ** 2) * nf
    k_mat = np.roll(k_mat, int(shiftx), axis=1)
    k_mat = np.roll(k_mat, int(shifty), axis=0)
    k_mod = np.round(k_mat, decimals=0)

    hval, _ = np.histogram(k_mod, bins=bins)

    kval = np.zeros(int(k_crit))
    kpow = np.zeros(int(k_crit))

    for j in range(int(k_crit)):
        kval[j] = np.sum(k_mod[k_mod == float(j)]) / hval[j]
        kpow[j] = np.sum(ps2D[k_mod == float(j)]) / hval[j]

    spec_k = kpow[1:np.size(hval) - 1]

    if not autocorr:
        tab_k = kval[1:np.size(hval) - 1] / (k_crit * 2. * reso)
    else:
        tab_k = kval[1:np.size(hval) - 1] * reso

    return tab_k, spec_k

def kgrid(image):
    """
    Compute the 2D spatial frequency grid (radius only) for a given image.

    Parameters
    ----------
    image : ndarray
        Input 2D image.

    Returns
    -------
    kmat : ndarray
        2D array of radial spatial frequencies (normalized).
    """
    na = float(image.shape[1])
    nb = float(image.shape[0])
    x, y = np.meshgrid(np.arange(na), np.arange(nb))

    if na % 2 == 0:
        x = (x - (na / 2)) / na
    else:
        x = (x - (na - 1) / 2) / na

    if nb % 2 == 0:
        y = (y - (nb / 2)) / nb
    else:
        y = (y - (nb - 1) / 2) / nb

    return np.sqrt(x ** 2 + y ** 2)

def cross_spec(image, image2, reso=1):
    """
    Compute the 1D cross power spectrum between two 2D images.

    Parameters
    ----------
    image : ndarray
        First 2D image.
    image2 : ndarray
        Second 2D image.
    reso : float, optional
        Pixel size in spatial units (e.g. arcmin).

    Returns
    -------
    spec_k : ndarray
        Azimuthally averaged cross power spectrum.
    """
    na = float(image.shape[1])
    nb = float(image.shape[0])
    nf = np.max(np.array([na, nb]))
    k_crit = nf / 2
    bins = np.arange(k_crit + 1)

    imft = np.fft.fft2(image)
    im2ft = np.fft.fft2(image2)
    ps2D = imft * np.conj(im2ft) / (na * nb)
    del imft

    x, y = np.meshgrid(np.arange(na), np.arange(nb))

    if na % 2 == 0:
        x = (x - (na / 2)) / na
        shiftx = na / 2
    else:
        x = (x - (na - 1) / 2) / na
        shiftx = (na - 1) / 2 + 1

    if nb % 2 == 0:
        y = (y - (nb / 2)) / nb
        shifty = nb / 2
    else:
        y = (y - (nb - 1) / 2) / nb
        shifty = (nb - 1) / 2 + 1

    k_mat = np.sqrt(x ** 2 + y ** 2) * nf
    k_mat = np.roll(k_mat, int(shiftx), axis=1)
    k_mat = np.roll(k_mat, int(shifty), axis=0)
    k_mod = np.round(k_mat, decimals=0)

    hval, _ = np.histogram(k_mod, bins=bins)

    kval = np.zeros(int(k_crit))
    kpow = np.zeros(int(k_crit))

    for j in range(int(k_crit)):
        kval[j] = np.sum(k_mod[k_mod == float(j)]) / hval[j]
        kpow[j] = np.sum(ps2D[k_mod == float(j)]) / hval[j]

    return kpow[1:np.size(hval) - 1]

def apodize_1d(radius, shape):
    """
    Create a 1D cosine taper window for edge apodization.

    Parameters
    ----------
    radius : float
        Fractional taper width (0 < radius < 1).
    shape : int
        Total length of the 1D array.

    Returns
    -------
    tap1d_x : ndarray
        1D tapering function.
    """
    if not (0 < radius < 1):
        raise ValueError("radius must be between 0 and 1")

    nx = shape
    nj = np.fix(radius * nx)
    dnj = int(nx - nj)

    tap1d_x = np.ones(nx)
    tap1d_x[0:dnj] = np.cos(3 * np.pi / 2 + np.pi / 2 * np.arange(dnj) / (dnj - 1))
    tap1d_x[nx - dnj:] = np.cos(np.pi / 2 * np.arange(dnj) / (dnj - 1))

    return tap1d_x

def apodize(radius, shape):
    """
    Create a 2D cosine taper window for edge apodization.

    Parameters
    ----------
    radius : float
        Fractional taper width (0 < radius < 1).
    shape : tuple of int
        Shape of the 2D image (ny, nx).

    Returns
    -------
    tapper : ndarray
        2D tapering function.
    """
    if not (0 < radius < 1):
        raise ValueError("radius must be between 0 and 1")

    ny, nx = shape
    dni = int(nx - np.fix(radius * nx))
    dnj = int(ny - np.fix(radius * ny))

    tap1d_x = np.ones(nx)
    tap1d_y = np.ones(ny)

    tap1d_x[0:dni] = np.cos(3 * np.pi / 2 + np.pi / 2 * np.arange(dni) / (dni - 1))
    tap1d_x[nx - dni:] = np.cos(np.pi / 2 * np.arange(dni) / (dni - 1))
    tap1d_y[0:dnj] = np.cos(3 * np.pi / 2 + np.pi / 2 * np.arange(dnj) / (dnj - 1))
    tap1d_y[ny - dnj:] = np.cos(np.pi / 2 * np.arange(dnj) / (dnj - 1))

    return np.outer(tap1d_y, tap1d_x)

def padding(input, fact):
    """
    Pad a 2D image by a given fractional amount with zeros.

    Parameters
    ----------
    input : ndarray
        Original 2D image.
    fact : float
        Padding fraction (e.g. 0.2 adds 20% on each side).

    Returns
    -------
    output : ndarray
        Padded image.
    """
    y, x = int(input.shape[0] * (1. + fact)), int(input.shape[1] * (1. + fact))
    width = input.shape[1]
    height = input.shape[0]

    output = np.zeros((y, x))
    xpos = int(x / 2 - width / 2)
    ypos = int(y / 2 - height / 2)

    output[ypos:height + ypos, xpos:width + xpos] = input
    return output

def depad(input, fact):
    """
    Crop a padded 2D image to its original size.

    Parameters
    ----------
    input : ndarray
        Padded 2D image.
    fact : float
        Padding fraction used during padding.

    Returns
    -------
    output : ndarray
        Cropped image matching original dimensions.
    """
    y, x = int(input.shape[0] / (1. + fact)) + 1, int(input.shape[1] / (1. + fact)) + 1
    width = input.shape[1]
    height = input.shape[0]

    xpos = int(width / 2 - x / 2)
    ypos = int(height / 2 - y / 2)

    return input[ypos:y + ypos, xpos:x + xpos]
