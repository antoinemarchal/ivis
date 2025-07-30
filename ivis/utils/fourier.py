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

- powspec(image, reso=1, autocorr=False, **kwargs)
    Computes the azimuthally averaged power spectrum of a 2D image, optionally returning
    the autocorrelation or cross-spectrum if a second image is passed via `im2`.

- cross_spec(image, image2, reso=1)
    Computes the cross-power spectrum between two 2D images.

- kgrid(image)
    Returns a normalized radial spatial frequency grid (k) corresponding to the FFT bins
    for a given image shape.

- apodize(radius, shape)
    Returns a 2D cosine taper mask to apodize the edges of an image.

- apodize_1d(radius, shape)
    Returns a 1D cosine taper to apodize the edges of a 1D array (e.g. spectrum).

- padding(input, fact)
    Pads a 2D image symmetrically by a given fractional amount (`fact`), centering
    the original image in a larger zero-filled array.

- depad(input, fact)
    Crops a previously padded 2D image, removing the borders added by `padding`.

Author: Taken from JF Robitaille package
"""

import numpy as np

def powspec(image, reso=1, autocorr=False, **kwargs):
	"""
	Calculate the power spectrum of a 2D image

	Parameters
	----------
	image : array_like
		Input array, must 2-dimentional and real
	reso : float, optional
		Resolution of the image in pixel^-1
		
	autocorr : If True, calculate the autocorrelation function
		instead of the power spectrum. (False by default)
		
	Optional keywords
	------------------
	im2 : array_like
		Second input array of the same size than image.
		If called, the function returns the cross-spectrum
		between 'image' and 'im2'

	Returns
	-------

	tab_k : Array of spatial scales used for the decomposition
	spec_k: The power spectrum
	"""
	
	na=float(image.shape[1])
	nb=float(image.shape[0])
	nf=np.max(np.array([na,nb]))

	k_crit = nf/2    
	k_min = 1

	bins = np.arange(k_crit+1)

	#Fourier transform & 2D power spectrum
	#---------------------------------------------

	imft=np.fft.fft2(image)
	
	if 'im2' in kwargs:
		im2 = kwargs.get('im2')
		im2ft = np.fft.fft2(im2)
		
		ps2D = imft * np.conj(im2ft) / (na*nb)

	else:
		ps2D = np.abs( imft )**2 / (na*nb)

	del imft
    
	if autocorr == True:
		ps2D = np.fft.ifft2(ps2D).real

	#Set-up kbins
	#---------------------------------------------

	x=np.arange(na)
	y=np.arange(nb)
	x,y=np.meshgrid(x,y)

	if (na % 2) == 0:
		x = (1.*x - ((na)/2.) ) / na
		shiftx = (na)/2.
	else:
		x = (1.*x - (na-1)/2.)/ na
		shiftx = (na-1.)/2.+1

	if (nb % 2) == 0:
		y = (1.*y - ((nb/2.)) ) / nb
		shifty = (nb)/2.
	else:
		y = (1.*y - (nb-1)/2.)/ nb
		shifty = (nb-1.)/2+1

	k_mat = np.sqrt(x**2 + y**2)
	k_mat = k_mat * nf 
	#k_mod = np.round(np.fft.fftshift(k_mat),decimals=0)
	
	k_mat= np.roll(k_mat,int(shiftx), axis=1)
	k_mat= np.roll(k_mat,int(shifty), axis=0)
	k_mod = np.round(k_mat,decimals=0)

	hval, rbin = np.histogram(k_mod,bins=bins)

	#Average values in same k bin
	#---------------------------------------------

	kval = np.zeros(int(k_crit))
	kpow = np.zeros(int(k_crit))

	for j in range(int(k_crit)):

		kval[j] = np.sum(k_mod[k_mod == float(j)]) / hval[j]
		kpow[j] = np.sum(ps2D[k_mod == float(j)]) / hval[j]

	spec_k = kpow[1:np.size(hval)-1]

	if autocorr == False:
		tab_k = kval[1:np.size(hval)-1] / (k_crit * 2.* reso)
	else:
		tab_k = kval[1:np.size(hval)-1] * reso

	return tab_k, spec_k


def kgrid(image):

	na=float(image.shape[1])
	nb=float(image.shape[0])
	
	x=np.arange(na)
	y=np.arange(nb)
	x,y=np.meshgrid(x,y)

	if (na % 2) == 0:
		x = (1.*x - ((na)/2.) ) / na
		shiftx = (na)/2.
	else:
		x = (1.*x - (na-1)/2.)/ na
		shiftx = (na-1.)/2.+1

	if (nb % 2) == 0:
		y = (1.*y - ((nb/2.)) ) / nb
		shifty = (nb)/2.
	else:
		y = (1.*y - (nb-1)/2.)/ nb
		shifty = (nb-1.)/2+1

	kmat = np.sqrt(x**2 + y**2)
    
	return kmat


def cross_spec(image, image2, reso=1):
	"""
	Calculate the power spectrum of a 2D image

	Parameters
	----------
	image : array_like
		Input array, must 2-dimentional and real
	reso : float, optional
		Resolution of the image in pixel^-1
		
	autocorr : If True, calculate the autocorrelation function
		instead of the power spectrum. (False by default)
		
	Optional keywords
	------------------
	im2 : array_like
		Second input array of the same size than image.
		If called, the function returns the cross-spectrum
		between 'image' and 'im2'

	Returns
	-------

	tab_k : Array of spatial scales used for the decomposition
	spec_k: The power spectrum
	"""
	
	na=float(image.shape[1])
	nb=float(image.shape[0])
	nf=np.max(np.array([na,nb]))

	k_crit = nf/2    
	k_min = 1

	bins = np.arange(k_crit+1)

	#Fourier transform & 2D power spectrum
	#---------------------------------------------
	imft=np.fft.fft2(image)	
	im2ft = np.fft.fft2(image2)	

	ps2D = imft * np.conj(im2ft) / (na*nb)
		
	del imft
    
	#Set-up kbins
	#---------------------------------------------

	x=np.arange(na)
	y=np.arange(nb)
	x,y=np.meshgrid(x,y)

	if (na % 2) == 0:
		x = (1.*x - ((na)/2.) ) / na
		shiftx = (na)/2.
	else:
		x = (1.*x - (na-1)/2.)/ na
		shiftx = (na-1.)/2.+1

	if (nb % 2) == 0:
		y = (1.*y - ((nb/2.)) ) / nb
		shifty = (nb)/2.
	else:
		y = (1.*y - (nb-1)/2.)/ nb
		shifty = (nb-1.)/2+1

	k_mat = np.sqrt(x**2 + y**2)
	k_mat = k_mat * nf 
	#k_mod = np.round(np.fft.fftshift(k_mat),decimals=0)
	
	k_mat= np.roll(k_mat,int(shiftx), axis=1)
	k_mat= np.roll(k_mat,int(shifty), axis=0)
	k_mod = np.round(k_mat,decimals=0)

	hval, rbin = np.histogram(k_mod,bins=bins)

	#Average values in same k bin
	#---------------------------------------------

	kval = np.zeros(int(k_crit))
	kpow = np.zeros(int(k_crit))

	for j in range(int(k_crit)):

		kval[j] = np.sum(k_mod[k_mod == float(j)]) / hval[j]
		kpow[j] = np.sum(ps2D[k_mod == float(j)]) / hval[j]

	spec_k = kpow[1:np.size(hval)-1]

	return spec_k

def apodize_1d(radius, shape):
    """                                                                                                                                                                                                                                                               
    Create edges apodization tapper 1d
    
    Parameters
    ----------
    nx : integer
    size of the tapper
    radius : float
    radius must be lower than 1 and greater than 0.
    
    Returns
    -------
    
    tapper : numpy array ready to multiply on your spectrum
    to apodize edges     
    """
    nx = shape

    if (radius >= 1) or (radius <= 0.):
        print('Error: radius must be lower than 1 and greater than 0.')
        return

    nj = np.fix(radius*nx)
    dnj = int(nx-nj)

    tap1d_x = np.ones(nx)

    tap1d_x[0:dnj] = (np.cos(3. * np.pi/2. + np.pi/2. * (1.* np.arange( dnj )/(dnj-1)) ))
    tap1d_x[nx-dnj:] = (np.cos(0. + np.pi/2. * (1.* np.arange(dnj)/(dnj-1)) ))

    return tap1d_x


def apodize(radius, shape):
    """
    Create edges apodization tapper
    
    Parameters
    ----------
    nx, ny : integers
    size of the tapper
    radius : float
    radius must be lower than 1 and greater than 0.
    
    Returns
    -------
    
    tapper : numpy array ready to multiply on your image
    to apodize edges
    """
    ny = shape[0]
    nx = shape[1]

    if (radius >= 1) or (radius <= 0.):
        print('Error: radius must be lower than 1 and greater than 0.')
        return
        
    ni = np.fix(radius*nx)
    dni = int(nx-ni)
    nj = np.fix(radius*ny)
    dnj = int(ny-nj)
    
    tap1d_x = np.ones(nx)
    tap1d_y = np.ones(ny)
    
    tap1d_x[0:dni] = (np.cos(3. * np.pi/2. + np.pi/2.* (1.* np.arange(dni)/(dni-1)) ))
    tap1d_x[nx-dni:] = (np.cos(0. + np.pi/2. * (1.* np.arange(dni)/(dni-1)) ))
    tap1d_y[0:dnj] = (np.cos(3. * np.pi/2. + np.pi/2. * (1.* np.arange( dnj )/(dnj-1)) ))
    tap1d_y[ny-dnj:] = (np.cos(0. + np.pi/2. * (1.* np.arange(dnj)/(dnj-1)) ))
    
    tapper = np.zeros((ny, nx))
    
    for i in range(nx):
        tapper[:,i] = tap1d_y
                        
    for i in range(ny):
        tapper[i,:] = tapper[i,:] * tap1d_x
        
    return tapper


def padding(input, fact):                
    y, x = int(input.shape[0]*(1.+fact)), int(input.shape[1]*(1.+fact)) 
    
    width = input.shape[1]
    height = input.shape[0]
    
    output = np.zeros((y,x))
    
    xpos = int(x/2 - width/2)
    ypos = int(y/2 - height/2)
    
    output[ypos:height+ypos,xpos:width+xpos] = input                

    return output
    
        
def depad(input, fact):
    y, x = int(input.shape[0]/(1.+fact))+1, int(input.shape[1]/(1.+fact))+1 

    width = input.shape[1]
    height = input.shape[0]
    
    output = np.zeros((y,x))
    
    xpos = int(width/2 - x/2)
    ypos = int(height/2 - y/2)
    
    output = input[ypos:y+ypos,xpos:x+xpos]                

    return output
