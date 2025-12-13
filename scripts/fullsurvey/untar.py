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

from ivis.io import DataProcessor
from ivis.logger import logger

plt.ion()

if __name__ == '__main__':    
    #path data
    path_ms = "/totoro/anmarchal/data/gaskap/fullsurvey/incoming/79266/"
    
    path_beams = "./" #directory of primary beams
    path_sd = "./" #path single-dish data - dummy here
    pathout = "/totoro/anmarchal/data/gaskap/fullsurvey/incoming/79266/" #path where data will be packaged and stored

    #create data processor
    data_processor = DataProcessor(path_ms, path_beams, path_sd, pathout)
    
    #PRE-COMPUTE DATA
    #untardir and fixms
    data_processor.untardir(max_workers=4, clear=False) #warning clean=True will clear the .tar files

