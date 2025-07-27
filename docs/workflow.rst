Workflow
========

.. raw:: html

   <p style="color:#dddddd; font-size: 16px; line-height: 1.6;">
   The IViS imaging pipeline performs non-linear joint deconvolution of interferometric and single-dish data using a regularized optimization approach.
   </p>

.. raw:: html

   <p style="color:#dddddd;">
   At the top level, a controller script (e.g., <code style="color:#5c6bc0;">pipeline.py</code>) instantiates a <code style="color:#ffb74d;">DataProcessor</code>, which loads visibilities from calibrated Measurement Sets (<code>.ms</code>), reprojects <code style="color:#ba68c8;">primary beam</code> models, and optionally includes a <code style="color:#9575cd;">single-dish</code> map.
   </p>

.. raw:: html

   <p style="color:#dddddd;">
   These inputs are assembled into a <code style="color:#ff8a65;">VisData</code> structure and interpolation grids, then passed to the <code style="color:#4fc3f7;">Imager</code>. The <code style="color:#4fc3f7;">Imager</code> constructs a model of the sky brightness and simulates visibilities using a forward operator that incorporates beam effects and Fourier transforms.
   </p>

.. raw:: html

   <p style="color:#dddddd;">
   It evaluates a loss function via <code style="color:#f06292;">mod_loss.objective()</code>, combining residuals and optional priors (such as Laplacian spatial regularization or single-dish consistency). Optimization is performed using the <code style="color:#ffcc80;">L-BFGS-B</code> algorithm from <code style="color:#f5f5f5;">scipy.optimize</code>.
   </p>

.. raw:: html

   <p style="color:#dddddd;">
   The final image cube is written to disk in physical units such as <code>Jy/beam</code>, <code>Jy/arcsec^2</code>, or <code>K</code>. This workflow supports <span style="color:#4db6ac;">GPU acceleration</span> and is designed to scale to large mosaics.
   </p>

.. raw:: html

   <p style="color:#aaaaaa;">
   The flowchart below summarizes the key modules and their data flow.
   </p>

.. graphviz::

   digraph ivis_workflow {
       rankdir=TB;
       bgcolor="#1e1e1e";
       fontcolor="white";
       fontsize=16;
       nodesep=1.0;
       ranksep=1.2;

       node [
           shape=box,
           style=filled,
           fontname="Helvetica",
           fontcolor="white",
           fillcolor="#1e1e1e"
       ];

       PIPELINE  [label="Runs DataProcessor\nand Imager", shape=box3d, color="#5c6bc0"];
       DATAPROC  [label="DataProcessor\n(ivis.io.data_processor)", color="#ffb74d"];
       VISDATA   [label="VisData\n(dataclass)", color="#ff8a65"];
       PBGRID    [label="compute_pb_and_grid()", color="#ba68c8"];
       SD        [label="read_sd()", color="#9575cd"];
       IMAGER    [label="Imager\n(ivis.imager)\n(uses L-BFGS-B)", color="#4fc3f7"];
       MODLOSS   [label="mod_loss.objective()\n(ivis.utils.mod_loss)", color="#f06292"];
       IMAGE     [label="Optimized Image Cube\n(saved to disk)", color="#b0bec5"];

       subgraph cluster_pipeline {
           label="ivis.pipeline (build your own)";
           style=dashed;
           fontcolor="#5c6bc0";
           color="#5c6bc0";
           PIPELINE;
       }

       edge [
           color="white",
           fontcolor="white",
           fontsize=14
       ];

       PIPELINE -> DATAPROC [label="calls"];
       PIPELINE -> IMAGER [label="calls"];
       DATAPROC -> VISDATA [label="returns"];
       DATAPROC -> PBGRID [label="generates"];
       DATAPROC -> SD [label="reads"];
       VISDATA -> IMAGER [label="input visibilities"];
       PBGRID -> IMAGER [label="input PB + Grid"];
       SD -> IMAGER [label="input SD map"];
       IMAGER -> MODLOSS [label="calls"];
       MODLOSS -> IMAGER [label="returns ∇loss"];
       IMAGER -> IMAGE [label="writes"];
   }
