Single Frequency Model (ClassicIViS)
====================================

.. raw:: html

   <div style="text-align: justify; color: #dddddd; font-size: 16px; line-height: 1.6;">

This page walks you through the Single Frequency model implemented in IViS.

.. raw:: html

   </div>

Interferometric Data Model
--------------------------

.. raw:: html

   <div style="text-align: justify; color: #dddddd; font-size: 16px; line-height: 1.6;">

The visibility measured between antennas at positions :math:`\mathbf{r}_i` and :math:`\mathbf{r}_j` is

.. raw:: html

   </div>

.. math::

    V_{i,j} = \langle E(\mathbf{r}_i, t) E^*(\mathbf{r}_j, t) \rangle_t \tag{1}

.. raw:: html

   <div style="text-align: justify; color: #dddddd; font-size: 16px; line-height: 1.6;">

To model the set of visibilities for each beam :math:`k`, we first reproject a local field from the global sky model :math:`I(r)` using an orthographic SIN projection. The header of each primary beam in ``BEAMS`` defines the local projection.

.. raw:: html

   </div>

.. math::

    I'_k(\ell, m) \xleftarrow{\text{SIN projection}} I(r) \tag{5}

.. raw:: html

   <div style="text-align: justify; color: #dddddd; font-size: 16px; line-height: 1.6;">

This is the step where mosaicking is natively handled. The starting coordinate system of the sky image (i.e., the WCS of :math:`I(r)`) is entirely arbitrary — it can be Cartesian, HEALPix, or any valid projection. What matters is that each beam is reprojected consistently into its own local frame before forward modeling.

.. raw:: html

   </div>

After projection, the local image is multiplied by the known primary beam :math:`A_k(\ell, m)`.  
This step naturally incorporates the direction-dependent effect (DDE) of the primary beam.

.. math::

    A_k(\ell, m) \times I'_k(\ell, m)

.. raw:: html

   <div style="text-align: justify; color: #dddddd; font-size: 16px; line-height: 1.6;">

For each beam :math:`k`, the model visibilities are obtained using the interferometric measurement equation:

.. raw:: html

   </div>

.. math::

    \tilde{V}_k(I'_k; u, v, w) = \iint A_k(\ell, m) \, I'_k(\ell, m) \, \frac{e^{-2\pi i [u\ell + v m + w(\sqrt{1 - \ell^2 - m^2} - 1)]}}{\sqrt{1 - \ell^2 - m^2}} \, d\ell \, dm \tag{3}

Under the small-angle approximation, this simplifies to:

.. math::

    \tilde{V}_k(I'_k; u, v) \approx \iint A_k(\ell, m) \, I'_k(\ell, m) \, e^{-2\pi i [u\ell + v m]} \, d\ell \, dm \tag{4}

.. raw:: html

   <div style="text-align: justify; color: #dddddd; font-size: 16px; line-height: 1.6;">

A non-uniform FFT (NuFFT, using the fiNuFFT implementation) is used to evaluate model visibilities at irregular :math:`(u,v)` coordinates — a process often referred to as *degridding*.  
This avoids interpolation onto a regular grid and circumvents gridding artifacts, while enabling fast computation.

This concept is not new and was implemented in the MPol package developed by Ian Czekala, which I learned about during a presentation at the NRAO 2024 workshop on synthesis imaging for radio interferometry.

.. raw:: html

   </div>

   
Cost Function
-------------

The residual visibilities for each beam k is

.. math::

    L_{1,k}(I'_k) = \tilde{V}_k(I'_k) - V_k \tag{7}

and the estimated parameter map :math:`I(r)` is defined as the minimizer of a cost function that includes the sum of the squares of the residual

.. math::

    J_k(I'_k) = \frac{1}{2} \sum_{u,v} \left( \frac{L_{1,k}(I'_k)}{\Sigma_{1,k}} \right)^2 \tag{8}

summed over the N beams

.. math::

    J(I) = \sum_k^N J_k(I'_k) \tag{9}

where :math:`\Sigma_{1,k}` is the standard deviation of the noise, provided in the measurement set of beam :math:`k` in column ``SIGMA``. This sum over the :math:`k` beams is what makes the deconvolution “joint”. 


.. math::

    Q(I) = J(I) + \lambda_r R(I)

The total cost function is a regularized non-linear least-square criterion, and the minimizer is

.. math::

    \hat{I}(r) = \arg \min_I Q(I) \tag{15}

.. raw:: html

   <div style="text-align: justify; color: #dddddd; font-size: 16px; line-height: 1.6;">

Here, :math:`\lambda_r` is a tunable hyperparameter that controls the strength of the regularization.
It balances data fidelity and any statistical prior that can be introduce in to cost function such as, e.g., smoothness of the reconstructed image.
This very general form is what we hope will make IViS a modular sofware where anyone can design their own cost function. 

In IViS base layer, the regularization term :math:`R(I)` is a Laplacian filter, which penalizes local pixel-to-pixel variations in the image intensity.
This encourages spatial smoothness and suppresses small-scale noise, especially in diffuse emission regions.
Unlike the Maximum Entropy Method (MEM), this approach does not maximize an entropy functional — instead, it imposes smoothness via a quadratic penalty.
In this case, 

.. raw:: html

   </div>

.. math::

    R(I) = \frac{1}{2} \| D I(r) \|_2^2 \tag{13}

.. math::
      
    d = \begin{bmatrix}
        0 & -1 & 0 \\
        -1 & 4 & -1 \\
        0 & -1 & 0
    \end{bmatrix}

.. raw:: html

   <div style="text-align: justify; color: #dddddd; font-size: 16px; line-height: 1.6;">

where :math:`D` is the matrix that performs the convolution with the kernel :math:`d`.

.. raw:: html

   </div>

Adding single dish data
-----------------------
To natively build the short spacing correction into IViS, we added the second data fidelity term. This idea was first introduced by Stanimirivic et al 2002. 

.. math::

    L_2(I) = \tilde{T_b}(I) - T_b

and :math:`I(r)` is defined as the minimizer of a cost function that is the sum of :math:`Q(I)` and

.. math::

    K(I) = \frac{1}{2} \left\| L_2(I) \right\|_{\Sigma_2}^2

where :math:`\Sigma_2` is the standard deviation of the noise in the single-dish data, usually measured from empty channel maps where no signal is detected. 

.. math::

    Q_{\mathrm{tot}}(\mathbf{I}) = Q(\mathbf{I}) + \lambda K(\mathbf{I}) + \lambda_r R(\mathbf{I})

where a new yper-parameter is introduced to tune the balance between the three terms. 


Optimization Strategy
---------------------

.. math::

    I^{(k+1)} = I^{(k)} - \alpha^{(k)} H^{-1}_{(k)} \nabla Q_{\text{tot}}(I^{(k)}) \tag{16}

.. raw:: html

   <div style="text-align: justify; color: #dddddd; font-size: 16px; line-height: 1.6;">

This optimizer allows constraints such as :math:`I(r) \geq 0`, which can be enabled when negative flux is not expected (e.g., when short-spacing information is available).

Notes:

- Gradients computed via **PyTorch autograd**
- Uses `pytorch-finufft` for GPU-accelerated NuFFT
- Avoids data gridding entirely
- Performs degridding via direct evaluation of model visibilities at irregular :math:`(u,v)`
- Residuals are not added back to the model (unlike in CLEAN)

.. raw:: html

   </div>
   
