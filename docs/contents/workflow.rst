IViS Workflow
=============

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
   The final image cube is written to disk in physical units of <code>Jy/arcsec^2</code> or <code>K</code>. This workflow supports <span style="color:#4db6ac;">GPU acceleration</span> and is designed to scale to large mosaics.
   </p>

The flowchart below summarizes the key modules and their data flow.

.. graphviz::

   digraph ivis_workflow {
       rankdir=TB;
       bgcolor="white";
       fontcolor="black";
       fontsize=22;
       nodesep=0.35;
       ranksep=1.2;
       ratio=compress;

       node [
           shape=box,
           style=filled,
           fontname="Helvetica-Bold",
           fontsize=20,
           fontcolor="black",
           fillcolor="white",
           margin=0.25,
           penwidth=2
       ];

       // --- Nodes ---
       PATHS      [label="Paths & WCS", color="#b0bec5"];
       PARAMS     [label="User parameters\n(λ, iterations, devices, ...)", color="#b0bec5"];

       CASAREAD   [label="CasacoreReader()\nread visibilities", color="#ffb74d"];
       VISDATA    [label="VisIData\n(dataclass)", color="#ff8a65"];

       DATAPROC   [label="DataProcessor\n(PB + grid)", color="#ba68c8"];
       PBGRID     [label="PB + grid FITS", color="#ba68c8"];

       MODEL      [label="Classic3D\n(model class)", color="#fbc02d"];
       IMAGER     [label="Imager3D\n(optimization loop)", color="#4fc3f7"];

       IMAGE      [label="Image cube\n(K or Jy arcsec⁻²)", color="#b0bec5"];

       // --- Links: ONLY label size increased ---
       edge [
           color="black",
           fontcolor="black",
           fontsize=22     // bigger link labels only
       ];

       // --- Data flow ---
       PATHS    -> CASAREAD   [label="path_ms"];
       PATHS    -> DATAPROC   [label="beams, header"];

       CASAREAD -> VISDATA    [label="returns"];
       VISDATA  -> IMAGER     [label="visibilities"];

       DATAPROC -> PBGRID     [label="read/compute"];
       PBGRID   -> IMAGER     [label="pb + grid"];

       MODEL    -> IMAGER     [label="forward + loss"];
       IMAGER   -> IMAGE      [label="writes"];

       // --- Invisible spine to enforce vertical ordering ---
       PATHS    -> CASAREAD  [style=invis];
       CASAREAD -> DATAPROC  [style=invis];
       DATAPROC -> IMAGER   [style=invis];

       // --- Clusters ---
       subgraph cluster_inputs {
           label="Configuration script";
           fontsize=18;
           style=dashed;
           color="#b0bec5";
           fontcolor="#5c6bc0";
           penwidth=2;
           PATHS; PARAMS;
       }

       subgraph cluster_io {
           label="Data Ingestion";
           fontsize=18;
           style=dashed;
           color="#ffb74d";
           fontcolor="#ffb74d";
           penwidth=2;
           CASAREAD; VISDATA;
       }

       subgraph cluster_pb {
           label="Pre-processing";
           fontsize=18;
           style=dashed;
           color="#ba68c8";
           fontcolor="#ba68c8";
           penwidth=2;
           DATAPROC; PBGRID;
       }

       subgraph cluster_recon {
           label="Optimization";
           fontsize=18;
           style=dashed;
           color="#4fc3f7";
           fontcolor="#4fc3f7";
           penwidth=2;
           MODEL; IMAGER; IMAGE;
       }
   }

..
   .. graphviz::

      digraph ivis_workflow {
	  rankdir=TB;
	  bgcolor="white";
	  fontcolor="black";
	  fontsize=22;
	  nodesep=0.35;
	  ranksep=1.2;
	  ratio=compress;

	  node [
	      shape=box,
	      style=filled,
	      fontname="Helvetica-Bold",
	      fontsize=20,
	      fontcolor="black",
	      fillcolor="white",
	      margin=0.25,
	      penwidth=2
	  ];

	  // --- Nodes ---
	  PIPELINE   [label="IViS workflow", shape=box3d, color="#5c6bc0"];

	  PATHS      [label="Paths & WCS", color="#b0bec5"];
	  PARAMS     [label="User parameters\n(λ, iterations, devices, ...)", color="#b0bec5"];

	  CASAREAD   [label="CasacoreReader()\nread visibilities", color="#ffb74d"];
	  VISDATA    [label="VisIData\n(dataclass)", color="#ff8a65"];

	  DATAPROC   [label="DataProcessor\n(PB + grid)", color="#ba68c8"];
	  PBGRID     [label="PB + grid FITS", color="#ba68c8"];

	  MODEL      [label="Classic3D\n(model class)", color="#fbc02d"];
	  IMAGER     [label="Imager3D\n(optimization loop)", color="#4fc3f7"];

	  IMAGE      [label="Image cube\n(K or Jy arcsec⁻²)", color="#b0bec5"];

	  edge [
	      color="black",
	      fontcolor="black",
	      fontsize=18
	  ];

	  // --- Pipeline controls (only config-level) ---
	  PIPELINE -> PATHS      [label="define"];
	  PIPELINE -> PARAMS     [label="set"];
	  PIPELINE -> MODEL      [label="select"];

	  // --- Data flow ---
	  PATHS    -> CASAREAD   [label="path_ms"];
	  PATHS    -> DATAPROC   [label="beams, header"];

	  CASAREAD -> VISDATA    [label="returns"];
	  VISDATA  -> IMAGER     [label="visibilities"];

	  DATAPROC -> PBGRID     [label="read/compute"];
	  PBGRID   -> IMAGER     [label="pb + grid"];

	  MODEL    -> IMAGER     [label="forward + loss"];
	  IMAGER   -> IMAGE      [label="writes"];

	  // --- Invisible spine to enforce vertical ordering ---
	  PIPELINE -> PATHS     [style=invis];
	  PATHS    -> CASAREAD  [style=invis];
	  CASAREAD -> DATAPROC  [style=invis];
	  DATAPROC -> IMAGER   [style=invis];

	  // --- Clusters ---
	  subgraph cluster_inputs {
	      label="Configuration script";
	      fontsize=18;
	      style=dashed;
	      color="#b0bec5";
	      fontcolor="#5c6bc0";
	      penwidth=2;
	      PATHS; PARAMS;
	  }

	  subgraph cluster_io {
	      label="Data Ingestion";
	      fontsize=18;
	      style=dashed;
	      color="#ffb74d";
	      fontcolor="#ffb74d";
	      penwidth=2;
	      CASAREAD; VISDATA;
	  }

	  subgraph cluster_pb {
	      label="Pre-processing";
	      fontsize=18;
	      style=dashed;
	      color="#ba68c8";
	      fontcolor="#ba68c8";
	      penwidth=2;
	      DATAPROC; PBGRID;
	  }

	  subgraph cluster_recon {
	      label="Optimization";
	      fontsize=18;
	      style=dashed;
	      color="#4fc3f7";
	      fontcolor="#4fc3f7";
	      penwidth=2;
	      MODEL; IMAGER; IMAGE;
	  }
      }

..
   .. graphviz::

      digraph ivis_workflow {
	  rankdir=LR;             // left-to-right layout
	  bgcolor="white";
	  fontcolor="black";
	  fontsize=22;            // global font size
	  nodesep=0.8;
	  ranksep=0.8;

	  node [
	      shape=box,
	      style=filled,
	      fontname="Helvetica-Bold",
	      fontsize=20,
	      fontcolor="black",
	      fillcolor="white",
	      margin=0.25,
	      penwidth=2           // thicker node borders
	  ];

	  // --- Main pipeline ---
	  PIPELINE   [label="Build your own pipeline", shape=box3d, color="#5c6bc0"];

	  // --- Inputs / config ---
	  PATHS      [label="Paths & WCS", color="#b0bec5"];
	  PARAMS     [label="User parameters\n(λ, iterations, devices, ...)", color="#b0bec5"];

	  // --- I/O stage ---
	  CASAREAD   [label="CasacoreReader()\nread visibilities", color="#ffb74d"];
	  VISDATA    [label="VisIData\n(dataclass)", color="#ff8a65"];

	  // --- PB + grid stage ---
	  DATAPROC   [label="DataProcessor\n(PB + grid)", color="#ba68c8"];
	  PBGRID     [label="PB + grid FITS", color="#ba68c8"];

	  // --- Imager & model ---
	  IMAGER     [label="Imager3D\n(optimization loop)", color="#4fc3f7"];
	  MODEL      [label="Classic3D\n(model class)", color="#fbc02d"];

	  // --- Output ---
	  IMAGE      [label="Image cube\n(K or Jy arcsec⁻²)", color="#b0bec5"];

	  // --- Edges (arrows) ---
	  edge [
	      color="black",
	      fontcolor="black",
	      fontsize=18
	  ];

	  PIPELINE -> PATHS      [label="define"];
	  PIPELINE -> PARAMS     [label="set"];
	  PIPELINE -> CASAREAD   [label="import"];
	  PIPELINE -> DATAPROC   [label="import"];
	  PIPELINE -> MODEL      [label="select"];
	  PIPELINE -> IMAGER     [label="assemble"];

	  PATHS    -> CASAREAD   [label="path_ms"];
	  PATHS    -> DATAPROC   [label="beams, header"];

	  CASAREAD -> VISDATA    [label="returns"];
	  VISDATA  -> IMAGER     [label="visibilities"];
	  DATAPROC -> PBGRID     [label="read/compute"];
	  PBGRID   -> IMAGER     [label="pb + grid"];
	  MODEL    -> IMAGER     [label="forward + loss"];
	  IMAGER   -> IMAGE      [label="writes"];

	  // --- Optional grouping for readability ---
	  subgraph cluster_inputs {
	      label="Inputs & Config";
	      fontsize=18;
	      style=dashed;
	      color="#b0bec5";
	      fontcolor="#5c6bc0";
	      penwidth=2;
	      PATHS; PARAMS;
	  }

	  subgraph cluster_io {
	      label="I/O";
	      fontsize=18;
	      style=dashed;
	      color="#ffb74d";
	      fontcolor="#ffb74d";
	      penwidth=2;
	      CASAREAD; VISDATA;
	  }

	  subgraph cluster_pb {
	      label="PB & Grid";
	      fontsize=18;
	      style=dashed;
	      color="#ba68c8";
	      fontcolor="#ba68c8";
	      penwidth=2;
	      DATAPROC; PBGRID;
	  }

	  subgraph cluster_recon {
	      label="Reconstruction";
	      fontsize=18;
	      style=dashed;
	      color="#4fc3f7";
	      fontcolor="#4fc3f7";
	      penwidth=2;
	      IMAGER; MODEL; IMAGE;
	  }
      }   
