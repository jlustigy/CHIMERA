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

import pymultinest as pmn

from run_wasp43_mcmc import *

def loglike_pmn(theta):
    try:
        ll = loglike(theta)[0]
        return ll
    except pysynphot.exceptions.ParameterOutOfBounds:
        pass
    return -np.inf

# Define general prior transformations
def prior_pmn(cube):
    """
    This prior function transforms general priors to the unit cube
    """
    return smarter.priors.get_prior_unit_cube(cube, PRIORS)

if __name__ == "__main__":

    tag = "test7"
    n_params = len(PRIORS)

    try: os.mkdir(tag)
    except OSError: pass

    # name of the output files
    prefix = os.path.join(tag, "%i-" %n_params)

    #pmn.run(loglike_pmn, prior_pmn, n_params, outputfiles_basename=prefix, resume=False, verbose=True,n_live_points=400)
    # run MultiNest
    result = pmn.solve(LogLikelihood=loglike_pmn, Prior=prior_pmn,
                       n_dims=n_params, outputfiles_basename=prefix,
                       n_live_points=400, sampling_efficiency=0.8,
                       evidence_tolerance=0.5, n_iter_before_update=10,
                       importance_nested_sampling=True,
                       multimodal=True, mode_tolerance = -1e90,
                       const_efficiency_mode=False,
                       max_modes=100, seed=-1, verbose=True, resume=True,
                       max_iter=0, n_clustering_params = None)

    print()
    print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
    print()
    print('parameter values:')
    for name, col in zip(parameters, result['samples'].transpose()):
        print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

    a = pmn.Analyzer(n_params = n_params, outputfiles_basename=prefix)
    s = a.get_stats()
    output=a.get_equal_weighted_posterior()
    pickle.dump(output, open(os.path.join(prefix, "%s.pic" %tag),"wb"))
