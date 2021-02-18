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

import run_wasp43_mcmc as exopie

"""
GLOBALS
"""

# Define data and xsecs
data_tag = "w43b_exopie_mirilrs_data"
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
    wl, y_binned, y_meas, y_err, XSECS = exopie.generate_data(savetag = data_tag, niriss = False, nirspec = False, mirilrs = True)

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

#"""
# Define Priors
exopie.PRIORS = [
    #smarter.priors.GaussianPrior(THETA_DEFAULTS["Teff"], 174.75, theta_name = "Teff", theta0 = THETA_DEFAULTS["Teff"]),
    #smarter.priors.GaussianPrior(-0.05, 0.17, theta_name = "logMH", theta0 = THETA_DEFAULTS["logMH"]),
    #smarter.priors.GaussianPrior(4.646, 0.052, theta_name = "logg", theta0 = THETA_DEFAULTS["logg"]),
    #smarter.priors.GaussianPrior(THETA_DEFAULTS["Rstar"], 0.0554, theta_name = "Rstar", theta0 = THETA_DEFAULTS["Rstar"]),
    #smarter.priors.GaussianPrior(THETA_DEFAULTS["d"], 0.3269, theta_name = "d", theta0 = THETA_DEFAULTS["d"]),
    smarter.priors.UniformPrior(300., 3000., theta_name = "Tirr", theta0 = THETA_DEFAULTS["Tirr"]),
    smarter.priors.UniformPrior(-3.0, 0.0, theta_name = "logKir", theta0 = THETA_DEFAULTS["logKir"]),
    smarter.priors.UniformPrior(-3.0, 1.0, theta_name = "logg1", theta0 = THETA_DEFAULTS["logg1"]),
    #smarter.priors.UniformPrior(0.0, 500.0, theta_name = "Tint", theta0 = THETA_DEFAULTS["Tint"]),
    smarter.priors.UniformPrior(-1.5, 3.0, theta_name = "logMet", theta0 = THETA_DEFAULTS["logMet"]),   # valid range is -1.5 - 3.0
    smarter.priors.UniformPrior(-1.0, 0.3, theta_name = "logCtoO", theta0 = THETA_DEFAULTS["logCtoO"]), # valid range is -1.0 - 0.3
    smarter.priors.UniformPrior(-6.0, 1.5, theta_name = "logPQCarbon", theta0 = THETA_DEFAULTS["logPQCarbon"]), # valid range -6.0 - 1.5
    smarter.priors.UniformPrior(-6.0, 1.5, theta_name = "logPQNitrogen", theta0 = THETA_DEFAULTS["logPQNitrogen"]),
    smarter.priors.GaussianPrior(0.93, 0.08, theta_name = "Rp", theta0 = THETA_DEFAULTS["Rp"]),
]

# Get state vector parameter names
exopie.THETA_NAMES = [prior.theta_name for prior in Pexopie.RIORS]

# Get Truth values
exopie.THETA0 = [prior.theta0 for prior in exopie.PRIORS]
#"""

if __name__ == "__main__":

    tag = "test_mirilrs2_fixedFs"
    nsteps = 5000
    ncpu = multiprocessing.cpu_count()
    nwalkers = 10*len(exopie.THETA0)

    # Run OE retrieval to initialize MCMC walker near posterior well
    popt, pcov, perr, p0 = exopie.perform_initial_optimization(tag, nwalkers)

    # Run MCMC inference
    exopie.run_mcmc(tag, processes = ncpu, nsteps = nsteps, p0 = p0)
