Workflow
========

The following diagram shows the data and model flow in IViS:

.. graphviz::

   digraph ivis_workflow {
       rankdir=TB;
       bgcolor="#1e1e1e";
       fontcolor="white";
       fontsize=14;
       nodesep=0.6;
       ranksep=1.0;
       node [shape=box, style=filled, fontname="Helvetica", fontcolor="white", color="white", fillcolor="#2e2e2e"];

       subgraph cluster_pipeline {
           label="ivis.pipeline (external script)";
           style=dashed;
           fontcolor="white";
           color="white";
           PIPELINE [label="Runs DataProcessor\nand Imager", shape=box3d, fillcolor="#37474f"];
       }

       DATAPROC [label="DataProcessor\n(ivis.io.data_processor)"];
       VISDATA [label="VisData\n(dataclass)"];
       PBGRID [label="compute_pb_and_grid()"];
       SD [label="read_sd()"];

       IMAGER [label="Imager\n(ivis.imager)\n(uses L-BFGS-B)"];
       MODLOSS [label="mod_loss.objective()\n(ivis.utils.mod_loss)"];
       IMAGE [label="Optimized Image Cube\n(saved to disk)"];

       edge [color="white", fontcolor="white", fontsize=12];

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
