Gaussian Multi-Frequency Synthesis (GMFS)
=========================================

.. raw:: html

   <div style="text-align: justify; color: #dddddd; font-size: 16px; line-height: 1.6;">

This page walks you through the Gaussian Multi-Frequency Synthesis (GMFS) model implemented in IViS.

As far as I know, this approach is new and has not been implemented before. It is conceptually related to Multi-Frequency Synthesis (MFS), from which it takes its name, in the sense that it uses a physical model to describe the frequency axis.

In this case, the model is Gaussian (not a Taylor expansion or power law), tailored to represent diffuse emission lines.

The base model to describe the line is that of ROHSA (Marchal et al. 2019), a spatially regularized multi-Gaussian decomposition algorithm developed to model hyperspectral data.

With GMFS, our goal is to see if we can recover faint low-brightness emission buried within the noise, and to quantify its amount in terms of SNR.

IViS provides a natural framework for this methodological development, which was in fact its original purpose (DECRA project 2025 – not funded).

I recommend that this project be part of either a PhD or Postdoc research effort. Please contact me if interested.

This development could have a significant impact, particularly in light of recent results from the MHONGOOSE survey, which show that the low surface brightness emission predicted by simulations is not observed by MeerKAT, despite its high sensitivity.

.. raw:: html

   </div>

