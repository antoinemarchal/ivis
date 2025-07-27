Single Frequency Model
======================

.. raw:: html

   <div style="text-align: justify; color: #dddddd; font-size: 16px; line-height: 1.6;">

This page walks you through the Single Frequency model implemented in IViS.

.. raw:: html

   </div>

Interferometric Data Model
--------------------------

.. raw:: html

   <div style="text-align: justify; color: #dddddd; font-size: 16px; line-height: 1.6;">

The visibility measured between antennas at positions :math:`\mathbf{r}_i` and :math:`\mathbf{r}_j` is:

.. raw:: html

   </div>

.. math::

    V_{i,j} = \langle E(\mathbf{r}_i, t) E^*(\mathbf{r}_j, t) \rangle_t \tag{1}

.. raw:: html

   <div style="text-align: justify; color: #dddddd; font-size: 16px; line-height: 1.6;">

Each beam :math:`k` models visibilities as:

.. raw:: html

   </div>

.. math::

    \tilde{V}_k(I'_k; u, v, w) = \iint A_k(\ell, m) \, I'_k(\ell, m) \, \frac{e^{-2\pi i [u\ell + v m + w(\sqrt{1 - \ell^2 - m^2} - 1)]}}{\sqrt{1 - \ell^2 - m^2}} \, d\ell \, dm \tag{3}

.. math::

    \tilde{V}_k(I'_k; u, v) \approx \iint A_k(\ell, m) \, I'_k(\ell, m) \, e^{-2\pi i [u\ell + v m]} \, d\ell \, dm \tag{4}

.. math::

    I'_k(\ell, m) \xleftarrow{\text{SIN projection}} I(r) \tag{5}

.. math::

    A_k(\ell, m) \times I'_k(\ell, m)

.. raw:: html

   <div style="text-align: justify; color: #dddddd; font-size: 16px; line-height: 1.6;">

This is the step where mosaicking is natively handled: the global sky model :math:`I(r)` is reprojected into the local coordinate frame of each beam :math:`k`, using an orthographic SIN projection.

The starting coordinate system of the sky image (i.e., the WCS of :math:`I(r)`) is entirely arbitrary — it can be Cartesian, HEALPix, or any valid projection.
What matters is that each beam is reprojected consistently into its own local frame before forward modeling.

This approach enables joint imaging of multiple overlapping fields in a natural and consistent way.

A non-uniform FFT (NuFFT) is used to evaluate model visibilities at irregular :math:`(u,v)` coordinates — a process often referred to as degridding. This avoids the need to interpolate the data onto a regular grid and circumvents gridding artifacts, while enabling fast computation.

.. raw:: html

   </div>

Cost Function
-------------

.. math::

    L_{1,k}(I'_k) = \tilde{V}_k(I'_k) - V_k \tag{7}

.. math::

    J_k(I'_k) = \frac{1}{2} \sum_{u,v} \left( \frac{L_{1,k}(I'_k)}{\Sigma_{1,k}} \right)^2 \tag{8}

.. math::

    Q(I) = \sum_k J_k(I'_k) \tag{9}

.. math::

    d = \begin{bmatrix}
        0 & -1 & 0 \\
        -1 & 4 & -1 \\
        0 & -1 & 0
    \end{bmatrix}

.. math::

    R(I) = \frac{1}{2} \| D I(r) \|_2^2 \tag{13}

.. math::

    Q_{\text{tot}}(I) = Q(I) + \lambda_r R(I) \tag{14'}

.. math::

    \hat{I}(r) = \arg \min_I Q_{\text{tot}}(I) \tag{15}

.. raw:: html

   <div style="text-align: justify; color: #dddddd; font-size: 16px; line-height: 1.6;">

Here, :math:`\lambda_r` is a tunable hyperparameter that controls the strength of the spatial regularization.
It balances data fidelity with smoothness in the reconstructed image.
The regularization term :math:`R(I)` is based on a Laplacian filter, which penalizes local pixel-to-pixel variations in the image intensity.
This encourages spatial smoothness and suppresses small-scale noise, especially in diffuse emission regions.
Unlike maximum entropy methods, this approach does not maximize an entropy functional — instead, it imposes smoothness via a quadratic penalty that is simple, effective, and differentiable.

.. raw:: html

   </div>

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
   
