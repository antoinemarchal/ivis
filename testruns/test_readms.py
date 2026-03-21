# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
 
from deconv.core import DataVisualizer, DataProcessor, Imager
from deconv.utils import dutils
from deconv import logger

plt.ion()

if __name__ == '__main__':    
    #path data
    path_ms = "/priv/avatar/amarchal/gaskap/nicolas/"
    
    path_beams = "./" #directory of primary beams
    path_sd = "./" #path single-dish data - dummy here
    pathout = "./" #path where data will be packaged and stored

    data_processor = DataProcessor(path_ms, path_beams, path_sd, pathout)

    vis_data = data_processor.read_vis_from_scratch(uvmin=0, uvmax=np.inf,
                                                    target_frequency=None,
                                                    target_channel=950,
                                                    extension=".ms",
                                                    blocks='single',
                                                    max_workers=1)

    
plt.figure()
plt.plot(vis_data.uu/1.e3, vis_data.vv/1.e3, ".m", ms=0.05)
plt.xlabel(r"u ($k\lambda$)")
plt.ylabel(r"v ($k\lambda$)")


