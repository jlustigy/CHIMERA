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
from smarter.utils import nsig_intervals

import run_wasp43_mcmc

THETA_DEFAULTS = run_wasp43_mcmc.THETA_DEFAULTS

# Define Priors
run_wasp43_mcmc.PRIORS = [
    smarter.priors.GaussianPrior(THETA_DEFAULTS["Teff"], 174.75, theta_name = "Teff", theta0 = THETA_DEFAULTS["Teff"]),
    smarter.priors.GaussianPrior(-0.05, 0.17, theta_name = "logMH", theta0 = THETA_DEFAULTS["logMH"]),
    smarter.priors.GaussianPrior(4.646, 0.052, theta_name = "logg", theta0 = THETA_DEFAULTS["logg"]),
    smarter.priors.GaussianPrior(THETA_DEFAULTS["Rstar"], 0.0554, theta_name = "Rstar", theta0 = THETA_DEFAULTS["Rstar"]),
    smarter.priors.GaussianPrior(THETA_DEFAULTS["d"], 0.3269, theta_name = "d", theta0 = THETA_DEFAULTS["d"]),
    smarter.priors.UniformPrior(300., 3000., theta_name = "Tirr", theta0 = THETA_DEFAULTS["Tirr"]),
    smarter.priors.UniformPrior(-3.0, 0.0, theta_name = "logKir", theta0 = THETA_DEFAULTS["logKir"]),
    smarter.priors.UniformPrior(-3.0, 1.0, theta_name = "logg1", theta0 = THETA_DEFAULTS["logg1"]),
    #smarter.priors.UniformPrior(0.0, 500.0, theta_name = "Tint", theta0 = THETA_DEFAULTS["Tint"]),
    smarter.priors.UniformPrior(-1.5, 3.0, theta_name = "logMet", theta0 = THETA_DEFAULTS["logMet"]),   # valid range is -1.5 - 3.0
    smarter.priors.UniformPrior(-1.0, 0.3, theta_name = "logCtoO", theta0 = THETA_DEFAULTS["logCtoO"]), # valid range is -1.0 - 0.3
    smarter.priors.UniformPrior(-6.0, 1.5, theta_name = "logPQCarbon", theta0 = THETA_DEFAULTS["logPQCarbon"]), # valid range -6.0 - 1.5
    smarter.priors.UniformPrior(-6.0, 1.5, theta_name = "logPQNitrogen", theta0 = THETA_DEFAULTS["logPQNitrogen"]),
    smarter.priors.UniformPrior(0.5, 1.5, theta_name = "Rp", theta0 = THETA_DEFAULTS["Rp"]), # Using uniform priors here to simulate non-transiting planet
]

# Get state vector parameter names
run_wasp43_mcmc.THETA_NAMES = [prior.theta_name for prior in run_wasp43_mcmc.PRIORS]

# Get Truth values
run_wasp43_mcmc.THETA0 = [prior.theta0 for prior in run_wasp43_mcmc.PRIORS]

if __name__ == "__main__":

    # Run MCMC (v1.0 data)
    #run_mcmc("test1", processes = 5, nsteps = 100, nwalkers = 10)
    #run_mcmc("test2", processes = 5, nsteps = 2000, nwalkers = 10)
    #run_mcmc("test3", processes = 3, nsteps = 500, nwalkers = 12)

    ncpu = multiprocessing.cpu_count()
    nwalkers = 10*len(run_wasp43_mcmc.THETA0)

    # Run MCMC (v2.0 data)
    #run_mcmc("test4", processes = 3, nsteps = 100, nwalkers = 12)  # 4 parameters
    #run_mcmc("test5", processes = 3, nsteps = 200, nwalkers = 24)  # 4 parameters + 4 more stellar = 8 parameters
    #run_mcmc("test6", processes = 3, nsteps = 200, nwalkers = 27)  # 4 parameters + 4 more stellar + Rp = 9 parameters
    #run_mcmc("test7", processes = 4, nsteps = 1000, nwalkers = 30)  # 4 parameters + 4 more stellar + Rp + 4 planet atm = 13 parameters
    run_wasp43_mcmc.run_mcmc("test10", processes = ncpu, nsteps = 5000, nwalkers = nwalkers)  # 4 parameters + 4 more stellar + Rp + 4 planet atm = 13 parameters
