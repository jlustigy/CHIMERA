# Import standard libraries
import os, sys
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
import astropy.units as u
import pickle
import multiprocessing
import emcee, corner

import chimera
import pysynphot

import coronagraph as cg
import smart
import smarter; smarter.utils.plot_setup()

HERE = os.path.abspath(os.path.split(__file__)[0])

# Important to import "free" version of code
import run_wasp43_free_mcmc as exopie

"""
GLOBALS
"""

# Define data and xsecs
unique_tag = "allin1"
data_tag = "w43b_exopie_free_data_%s" %unique_tag
if os.path.exists(data_tag+".npz"):
    print("Loading Synthetic Data and xsecs...")
    data = np.load(data_tag+".npz", allow_pickle=True)
    wl=data["wl"]
    y_binned=data["y_binned"]
    y_meas=data["y_meas"]
    y_err=data["y_err"]
    XSECS=data["xsecs"]
else:
    # Make fake dataset
    print("Generating Synthetic Data and xsecs...")
    wl, y_binned, y_meas, y_err, XSECS = exopie.generate_data(savetag = data_tag, niriss = True, nirspec = True, mirilrs = True)

use_random_noise = False
if use_random_noise:
    pass
else:
    y_meas = y_binned

# Set globals in original script with other functions
exopie.wl = wl
exopie.y_meas = y_meas
exopie.y_binned = y_binned
exopie.y_err = y_err
exopie.XSECS = XSECS

if __name__ == "__main__":

    tag = "test_%s" %unique_tag
    nsteps = 5000
    ncpu = multiprocessing.cpu_count()
    nwalkers = 10*len(exopie.THETA0)

    # Run OE retrieval to initialize MCMC walker near posterior well
    popt, pcov, perr, p0 = exopie.perform_initial_optimization(tag, nwalkers)

    # Run MCMC inference
    exopie.run_mcmc(tag, processes = ncpu, nsteps = nsteps, p0 = p0)
