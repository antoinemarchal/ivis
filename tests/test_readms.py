# -*- coding: utf-8 -*-
import numpy as np
from astropy import units as u
from astropy.constants import c
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
from radio_beam import Beam
import torch
from tqdm import tqdm as tqdm
import queue  # Standard Python queue for passing data between threads
import threading
import multiprocessing
import time

from deconv.core import DataVisualizer, DataProcessor, Imager
from deconv.utils import dutils
from deconv import logger

import marchalib as ml #remove

plt.ion()

if __name__ == '__main__':    
    #path data
    path_ms = "/priv/avatar/amarchal/gaskap/nicolas/"
    
    path_beams = "./" #directory of primary beams
    path_sd = "./" #path single-dish data - dummy here
    pathout = "./" #path where data will be packaged and stored

    data_processor = DataProcessor(path_ms, path_beams, path_sd, pathout)

    vis_data = data_processor.read_vis_from_scratch(uvmin=0, uvmax=7000,
                                                    target_frequency=None,
                                                    target_channel=950,
                                                    extension=".ms",
                                                    blocks='single',
                                                    max_workers=1)

    
