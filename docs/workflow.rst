Workflow
========

.. raw:: html

   <p style="color:#dddddd; font-size: 16px; line-height: 1.6; text-align: justify;">
   The IViS imaging pipeline performs non-linear joint deconvolution of interferometric and optional single-dish data using a regularized optimization approach.
   </p>

.. raw:: html

   <p style="color:#dddddd; text-align: justify;">
   At the top level, a controller script (e.g., <code style="color:#5c6bc0;">pipeline.py</code>) instantiates a <code style="color:#ffb74d;">DataProcessor</code>, which loads visibilities from calibrated Measurement Sets (<code>.ms</code>) and reprojects <code style="color:#ba68c8;">primary beam</code> models. A single-dish map may also be provided for hybrid deconvolution.
   </p>

.. raw:: html

   <p style="color:#dddddd; text-align: justify;">
   These inputs are assembled into a <code style="color:#ff8a65;">VisData</code> structure and interpolation grids, then passed to the <code style="color:#4fc3f7;">Imager</code>. A <code style="color:#fbc02d;">model class</code> is chosen in the pipeline script and passed explicitly to the <code>Imager.process()</code> method.
   </p>

.. raw:: html

   <p style="color:#dddddd; text-align: justify;">
   The model’s <code style="color:#fbc02d;">forward()</code> method simulates visibilities given image parameters, and the <code style="color:#fbc02d;">loss()</code> method defines the cost function and gradient. Optimization is performed using the <code style="color:#ffcc80;">L-BFGS-B</code> algorithm from <code style="color:#f5f5f5;">scipy.optimize</code>.
   </p>

.. raw:: html

   <p style="color:#dddddd; text-align: justify;">
   The final image cube is written to disk in physical units such as <code>Jy/beam</code>, <code>Jy/arcsec^2</code>, or <code>K</code>. This workflow supports <span style="color:#4db6ac;">GPU acceleration</span> and is designed to scale to large mosaics.
   </p>

The flowchart below summarizes the key modules and their data flow.

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

       PIPELINE  [label="Runs DataProcessor,\nselects model,\ncalls Imager", shape=box3d, color="#5c6bc0"];
       DATAPROC  [label="DataProcessor\n(ivis.io.data_processor)", color="#ffb74d"];
       VISDATA   [label="VisData\n(dataclass)", color="#ff8a65"];
       PBGRID    [label="compute_pb_and_grid()", color="#ba68c8"];
       IMAGER    [label="Imager\n(ivis.imager)\n(uses L-BFGS-B)", color="#4fc3f7"];
       MODEL     [label="ClassicIViS or other\n(ivis.models)", color="#fbc02d"];
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
       PIPELINE -> MODEL [label="selects model"];
       PIPELINE -> IMAGER [label="passes model + data"];
       DATAPROC -> VISDATA [label="returns"];
       DATAPROC -> PBGRID [label="generates"];
       VISDATA -> IMAGER [label="input visibilities"];
       PBGRID -> IMAGER [label="input PB + Grid"];
       MODEL -> IMAGER [label="provides loss/forward"];
       IMAGER -> IMAGE [label="writes"];
   }
