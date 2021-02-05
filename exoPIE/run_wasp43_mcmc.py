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
import dill

import chimera
import pysynphot

import coronagraph as cg
import smart
import smarter; smarter.utils.plot_setup()

from smarter.utils import nsig_intervals

HERE = os.path.abspath(os.path.split(__file__)[0])

def plot_mcmc_trace(tag, labels = None, iteration = None, plt_path = "", bins = 20,
                    iburn0 = 0, accept_quantiles = [0.1, 0.5, 0.9],
                    saveplot = True, chain_alpha = 0.3, show_quantiles = False,
                    emcee_backend = None):

    """
    Plot the evolution of MCMC chains as a function of iteration in a
    "trace plot".

    Parameters
    ----------
    tag : str
        Name of simulation
    labels : list
        List of string labels
    iteration : int
        Iteration number to append to tag
    plt_path : str
        Location to save plots
    bins : int
        Number of histogram bins
    iburn0 : int
        Initial burn-in cut
    accept_quantiles : list
        Thin chains by quantile
    saveplot : bool
        Set to save figure
    chain_alpha : float
        Chain plot alpha
    show_quantiles : bool
        Plot the quantile lines and zoomed hist (default is `False`)

    Returns
    -------
    fig : mpl.Figure
        Trace plot
    keep_mask : numpy.array of bool
        Mask for thinning chains
    """

    #tag = "../scripts/"+retrieval.tag;

    # Get parameter labels
    #labels = retrieval.forward.theta_names

    if emcee_backend is None:

        if iteration is not None:
            mcmc_tag = tag+str(iteration)+".h5"
        else:
            mcmc_tag = tag

        # Open MCMC output file and get the chain
        reader = emcee.backends.HDFBackend(mcmc_tag, read_only=True)

    else:

        # Override and use provided emcee object
        reader = emcee_backend

    chain = reader.get_chain()

    # Get dimensions of chain
    ndim = chain.shape[2]
    nwalk = chain.shape[1]

    # Create fake labels if None provided
    if labels is None:
        labels = ["$x_{%i}$" %(i+1) for i in range(ndim)]

    # Get flat chain
    flatchain = reader.get_chain(flat=True, discard = iburn0)
    theta_med = np.median(flatchain, axis = 0)

    # Calculate convergence diagnostics
    acceptance_fraction = 100*np.mean(reader.accepted / float(reader.iteration))
    avg_autocor_time = np.mean(reader.get_autocorr_time(tol = 0))

    # Print diagnostics
    print("Mean acceptance fraction: %0.2f%%" %(acceptance_fraction))
    print("Mean autocorrelation time: %.3f steps" %(avg_autocor_time))

    # Derive ranges
    xrange = []
    for i in range(flatchain.shape[1]):
        q_l, q_50, q_h, q_m, q_p = nsig_intervals(flatchain[:,i], quantiles=accept_quantiles)
        xrange.append((q_l, q_h))

    # Plot the chains w/out the burn-in
    fig = plt.figure(figsize=(12, int(3*ndim)))
    fig.subplots_adjust(bottom=0.05, top=0.95, hspace=0.1)
    axc = [plt.subplot2grid((ndim, 10), (n, 0), colspan=8, rowspan=1)
           for n in range(ndim)]
    axh = [plt.subplot2grid((ndim, 10), (n, 8), colspan=2,
                            rowspan=1, sharey=axc[n]) for n in range(ndim)]

    # Create empty list for chains to keep
    ikeep = []

    # Loop over free params
    for i, label in enumerate(labels):

        # Remove x ticks if not the last plot
        if i < (len(labels) - 1):
            axc[i].set_xticklabels([])

        axc[i].set_ylabel(label, fontsize=24)

        # Draw confidence intervals
        if show_quantiles:
            axc[i].axhline(xrange[i][0], ls = "dashed", c="k")
            axc[i].axhline(xrange[i][1], ls = "dashed", c="k")
            axh[i].axhline(xrange[i][0], ls = "dashed", c="k")
            axh[i].axhline(xrange[i][1], ls = "dashed", c="k")

        # Loop over walkers
        ik = []   # List of walkers to keep
        for k in range(nwalk):

            # if the median walker value is outside of confidence interval:
            if show_quantiles and ((np.median(chain[iburn0:, k, i]) > xrange[i][1]) or (np.median(chain[iburn0:, k, i]) < xrange[i][0])):

                # Plot in grey
                axc[i].plot(chain[iburn0:, k, i], alpha=chain_alpha, lw=1, c = "grey")

                # Set ikeep to false
                ik.append(False)

            else:

                # Plot normally and keep
                axc[i].plot(chain[iburn0:, k, i], alpha=chain_alpha, lw=1, color = "C%i" %(k%10))
                ik.append(True)

        # Append ikeep
        ik = np.array(ik)
        ikeep.append(ik)

        # Plot the collapsed histogram
        color = "grey"
        axh[i].hist(chain[iburn0:, ik, i].flatten(), bins=bins,
                    orientation="horizontal", alpha = 0.25, histtype='stepfilled',
                    edgecolor = 'none', color=color, lw=2)
        axh[i].hist(chain[iburn0:, ik, i].flatten(), bins=bins,
                    orientation="horizontal", histtype='step',
                    fill=False, color=color, lw=1)

        if show_quantiles:
            # Plot the zoomed hist
            axh_twin = axh[i].twinx().twiny()
            axh_twin.hist(chain[iburn0:, ik, i].flatten(), bins=3*bins,
                        orientation="vertical", histtype='step',
                        fill=True, color='k', lw=0, alpha = 0.25)
            # Plot tweaks
            plt.setp(axh_twin.get_yticklabels(), visible=False)
            plt.setp(axh_twin.get_xticklabels(), visible=False)
            axh_twin.set_xlim(xrange[i])
            axh_twin.set_yticks([])

        # Plot tweaks
        xlim0 = axc[i].get_xlim()
        axc[i].set_xlim(left=0,right=xlim0[1])
        plt.setp(axh[i].get_yticklabels(), visible=False)
        plt.setp(axh[i].get_xticklabels(), visible=False)

    # determine mask for which walkers to keep
    ikeep = np.array(ikeep)
    keep_mask = np.array(np.floor(np.sum(ikeep, axis = 0) / ikeep.shape[0]), dtype=bool)

    # Final plot tweaks
    for ax in fig.axes:
        ax.xaxis.label.set_fontsize(18)
        #ax.yaxis.label.set_fontsize(18)
        ax.title.set_fontsize(18)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(18)

    axc[-1].set_xlabel("MCMC Steps")

    if saveplot:
        fig.savefig(os.path.join(plt_path, "trace.png"), bbox_inches = "tight")
    else:
        plt.show()

    return fig, keep_mask

"""
FORWARD MODEL & LIKELIHOOD
"""

# Initialize as empty
THETA_NAMES = []

# Initial conditions
THETA_DEFAULTS = {
    "Rp" : 0.93,
    "Rstar" : 0.6629, #0.6,
    "M" : 1.776,
    "D" : 0.014,
    "Tirr" : 1000,
    "logKir" : -1.5,
    "logg1" : -0.7,
    "Tint" : 200,
    "logMet" : 0.0, #2.0,
    "logCtoO": -0.26, #-1.0,
    "logPQCarbon" : -5.5,
    "logPQNitrogen" : -5.5,
    "logKzz" : 7,
    "fsed" : 2.0,
    "logPbase" : -1.0,
    "logCldVMR" : -5.5,
    "logKcld" : -40,
    "logRayAmp" : -30,
    "RaySlope" : 0,
    "Teff" : 4305.0, #4400.,
    "logMH" : -0.05,
    "logg" : 4.646,
    "d" : 86.7467#80.0
}

def run_pie_model_general(theta):
    """
    """

    # Create dictionary
    theta_dict = dict(zip(THETA_NAMES, theta))

    # Unpack parameters to retrieve if they are in state vector
    # otherwise use the default fixed values

    #setup "input" parameters. We are defining our 1D atmosphere with these
    #the parameters
    #planet/star system params--xRp is the "Rp" free parameter, M right now is fixed, but could be free param
    Rp = theta_dict.get("Rp", THETA_DEFAULTS["Rp"])      # Planet radius in Jupiter Radii--this will be forced to be 10 bar radius--arbitrary (scaling to this is free par)
    Rstar = theta_dict.get("Rstar", THETA_DEFAULTS["Rstar"]) # Stellar Radius in Solar Radii
    M = theta_dict.get("M", THETA_DEFAULTS["M"])       # Mass in Jupiter Masses
    D=theta_dict.get("D", THETA_DEFAULTS["D"])        # Semimajor axis in AU--for reflected light component

    #TP profile params (3--Guillot 2010, Parmentier & Guillot 2013--see Line et al. 2013a for implementation)
    Tirr=theta_dict.get("Tirr", THETA_DEFAULTS["Tirr"])       # Irradiation temperature as defined in Guillot 2010
    logKir=theta_dict.get("logKir", THETA_DEFAULTS["logKir"])   # TP profile IR opacity (log there-of) controlls the "vertical" location of the gradient
    logg1=theta_dict.get("logg1", THETA_DEFAULTS["logg1"])     # single channel Vis/IR (log) opacity. Controls the delta T between deep T and TOA T
    Tint=theta_dict.get("Tint", THETA_DEFAULTS["Tint"])        # interior temperature...this would be the "effective temperature" if object were not irradiated

    #Composition parameters---assumes "chemically consistent model" described in Kreidberg et al. 2015
    logMet=theta_dict.get("logMet", THETA_DEFAULTS["logMet"])                # Metallicity relative to solar log--solar is 0, 10x=1, 0.1x = -1: valid range is -1.5 - 3.0
    logCtoO=theta_dict.get("logCtoO", THETA_DEFAULTS["logCtoO"])            # log C-to-O ratio: log solar is -0.26: valid range is -1.0 - 0.3
    logPQCarbon=theta_dict.get("logPQCarbon", THETA_DEFAULTS["logPQCarbon"])     # CH4, CO, H2O Qunech pressure--forces CH4, CO, and H2O to constant value at quench pressure value: valid range -6.0 - 1.5
    logPQNitrogen=theta_dict.get("logPQNitrogen", THETA_DEFAULTS["logPQNitrogen"]) # N2, NH3 Quench pressure--forces N2 and NH3 to ""

    #Ackerman & Marley 2001 Cloud parameters--physically motivated with Mie particles
    logKzz=theta_dict.get("logKzz", THETA_DEFAULTS["logKzz"])          # log Kzz (cm2/s)--valid range: 2 - 11 -- higher values make larger particles
    fsed=theta_dict.get("fsed", THETA_DEFAULTS["fsed"])            # sediminetation efficiency--valid range: 0.5 - 5--lower values make "puffier" more extended cloud
    logPbase=theta_dict.get("logPbase", THETA_DEFAULTS["logPbase"])   # cloud base pressure--valid range: -6.0 - 1.5
    logCldVMR=theta_dict.get("logCldVMR", THETA_DEFAULTS["logCldVMR"]) # cloud condensate base mixing ratio (e.g, see Fortney 2005)--valid range: -15 - -2.0

    #simple 'grey+rayleigh' parameters just in case you don't want to use a physically motivated cloud
    #(most are just made up anyway since we don't really understand all of the micro-physics.....)
    logKcld = theta_dict.get("logKcld", THETA_DEFAULTS["logKcld"])      # uniform in altitude and in wavelength "grey" opacity (it's a cross-section)--valid range: -50 - -10
    logRayAmp = theta_dict.get("logRayAmp", THETA_DEFAULTS["logRayAmp"])  # power-law haze amplitude (log) as defined in des Etangs 2008 "0" would be like H2/He scat--valid range: -30 - 3
    RaySlope = theta_dict.get("RaySlope", THETA_DEFAULTS["RaySlope"])      # power law index 4 for Rayleigh, 0 for "gray".  Valid range: 0 - 6

    # Stellar parameters
    Teff = theta_dict.get("Teff", THETA_DEFAULTS["Teff"])     # Stellar effective temperature [K]
    logMH = theta_dict.get("logMH", THETA_DEFAULTS["logMH"])   # Stellar metallicity
    logg = theta_dict.get("logg", THETA_DEFAULTS["logg"])     # Stellar log gravity
    d = theta_dict.get("d", THETA_DEFAULTS["d"])            # Stellar distance

    #unpacking parameters to retrieve (these override the fixed values above)
    #Teff, Tirr, logKir, logg1 = theta

    #stuffing all variables into state vector array
    x=np.array([Tirr, logKir,logg1, Tint, logMet, logCtoO, logPQCarbon,logPQNitrogen, Rp, Rstar, M, D, logKzz, fsed,logPbase,logCldVMR, logKcld, logRayAmp, RaySlope, Teff, logMH, logg, d])

    #gas scaling factors to mess with turning on various species
    #set to "0" to turn off a gas. Otherwise keep set at 1
    #thermochemical gas profile scaling factors
    # 0   1    2    3   4    5    6     7    8    9   10    11   12   13    14   15   16   17   18  19 20   21
    #H2O  CH4  CO  CO2 NH3  N2   HCN   H2S  PH3  C2H2 C2H6  Na    K   TiO   VO   FeH  H    H2   He   e- h-  mmw
    gas_scale=np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1., 1., 1.]) #can be made free params if desired (won't affect mmw)#can be made free params if desired (won't affect mmw)

    # Run chimera forward model
    foo = chimera.fx_emis_pie(x, -1, gas_scale, XSECS)
    y_binned,y_mod,wno,atm,Ftoa,Fstar,Fstar_TOA,Fup_therm,Fup_ref= foo

    # Convert fluxes to Earth distances
    Fstar_earth = Fstar*( Rstar * u.Rsun.in_units(u.m) / (d * u.pc.in_units(u.m)))**2
    #Fplan_earth = Ftoa * ( Rp * u.Rjup.in_units(u.m) / (d * u.pc.in_units(u.m)))**2
    Fplan_therm_earth = Fup_therm * ( Rp * u.Rjup.in_units(u.m) / (d * u.pc.in_units(u.m)))**2

    # Sum flux components
    Fobs = Fstar_earth + Fplan_therm_earth

    return Fobs, Fstar_earth, Fplan_therm_earth, atm

#defining log-likelihood function
# log-likelihood
def loglike(theta):

    y_binned, y_star, y_planet, atm = run_pie_model_general(theta)

    loglikelihood=-0.5*np.nansum((y_meas-y_binned)**2/y_err**2)  #your typical "quadratic" or "chi-square"

    return loglikelihood, [y_binned, y_star, y_planet]

def neg_loglike(theta):
    ll, blobs = loglike(theta)
    return -ll

def logprob(theta):

    lp = smarter.priors.get_lnprior(theta, PRIORS)

    if np.isfinite(lp):
        ll, blobs = loglike(theta)
        return ll + lp

    return -np.inf

def neg_logprob(theta):
    return -logprob(theta)

def logprob_blobs(theta):

    lp = smarter.priors.get_lnprior(theta, PRIORS)

    if np.isfinite(lp):
        try:
            ll, blobs = loglike(theta)
            return ll + lp, blobs
        except pysynphot.exceptions.ParameterOutOfBounds:
            pass

    return -np.inf, [np.nan*np.ones_like(wl), np.nan*np.ones_like(wl), np.nan*np.ones_like(wl)]

"""
MAKE FAKE DATA
"""

def load_kevins_errors(niriss = True, nirspec = True):
    """
    Load some PandExo outputs for WASP-43b using Kevin's files.
    """

    # Get NIRISS Data
    handle  = open(os.path.join(HERE, 'jwst_inputs/niriss_soss-W43-r100.p'), 'rb')
    model   = pickle.load(handle)
    obstime = model['RawData']['electrons_out'][0]/model['RawData']['e_rate_out'][0]/3600
    #print(obstime)
    wave = model['FinalSpectrum']['wave']
    spectrum = model['FinalSpectrum']['spectrum']
    error = model['FinalSpectrum']['error_w_floor']
    randspec = model['FinalSpectrum']['spectrum_w_rand']
    snr = model['RawData']['electrons_out']/np.sqrt(model['RawData']['var_out'])
    #print(wave.min(), wave.max())

    # Get NIRSpec Data
    handle  = open(os.path.join(HERE, 'jwst_inputs/nirspec_g395h-W43-r100.p'), 'rb')
    #handle  = open('niriss_soss-W43.p', 'rb')
    model_g395  = pickle.load(handle)
    #print(model['RawData']['electrons_out'][0]/model['RawData']['e_rate_out'][0]/3600)
    wave_g395 = model_g395['FinalSpectrum']['wave']
    spectrum_g395 = model_g395['FinalSpectrum']['spectrum']
    error_g395 = model_g395['FinalSpectrum']['error_w_floor']
    randspec_g395 = model_g395['FinalSpectrum']['spectrum_w_rand']
    #snr_g395 = model_g395['RawData']['electrons_out']/np.sqrt(model_g395['RawData']['var_out'])
    snr_g395 = 1./error_g395
    #print(wave_g395.min(), wave_g395.max())

    if niriss and nirspec:
        # Conbine datas
        wave2   = np.concatenate((wave, wave_g395))
        snr2    = np.concatenate(( snr,  snr_g395))
    elif niriss:
        wave2 = wave
        snr2 = snr
    elif nirspec:
        wave2 = wave_g395
        snr2 = snr_g395
    else:
        print("Error: Must use one of the instruments")

    m = snr2 > 100
    wave2 = wave2[m]
    snr2 = snr2[m]

    return wave2, snr2

def generate_data(niriss = True, nirspec = True, savetag = "w43b_exopie_data", seed=42):
    """
    """

    # Load Kevin's SNR files
    x, snr = load_kevins_errors(niriss = niriss, nirspec = nirspec)

    # Given a datafile, bin to CHIMERA's R=100 grid (This feels DUMB!)
    wnomin = np.floor(np.min(1e4 / x))
    wnomax = np.ceil(np.max(1e4 / x))

    # Prep CHIMERA inputs
    global XSECS
    observatory='JWST'
    directory = os.path.join(os.getcwd(), '..','ABSCOEFF_CK')
    XSECS=chimera.xsects(wnomin, wnomax, observatory, directory,stellar_file=None)
    XSECS=list(XSECS)  # This appears necessary to set the stellar flux *within* fx

    # Chimera Grids
    wno = XSECS[2]
    wl = 1e4 / wno

    # Derive bin widths
    dwl = wl[:-1] - wl[1:]        # Bin spacing
    mwl = 0.5*(wl[:-1] + wl[1:])  # Bin centers
    dwl = sp.interpolate.interp1d(mwl, dwl, fill_value="extrapolate")(wl)

    # Run model with default inputs
    Fobs, Fstar_earth, Fplan_therm_earth, atm = run_pie_model_general([])

    # Rebin data to CHIMERA grid
    y_meas_binned, y_err_binned = cg.downbin_spec_err(np.ones_like(snr), 1.0/snr, x, wl[::-1], dlam=dwl[::-1])

    # Final binned model and error
    y_binned = Fobs
    y_err = Fobs * y_err_binned[::-1]

    # Calculate Gaussian noise
    np.random.seed(seed=seed)   # User seed
    gaus = np.random.randn(len(wl))
    np.random.seed(None)   # User seed

    # Add gaussian noise to observed data
    y_meas = Fobs + y_err * gaus

    # Save datafile
    #"""
    np.savez(savetag+".npz",
             wl=wl, y_binned=y_binned,
             y_meas=y_meas, y_err=y_err,
             xsecs=XSECS)
    #"""

    # Plot
    fig, axes = plt.subplots(2,1, figsize = (14, 10))
    ax= axes[0]
    ax2 = axes[1]

    ax.set_xlim(wl.min(), wl.max())
    ax2.set_xlim(wl.min(), wl.max())
    ax2.set_xlabel("Wavelength [$\mu$m]")
    ax.set_ylabel("Observed Flux [W/m$^2$/m]")

    ax.plot(wl, y_binned, label = "Star + Planet", color = "C0")
    #ax.plot(wl, y_planet, label = "Planet thermal", color = "C3")
    ax.errorbar(wl, y_meas, yerr=y_err, fmt = ".k", label = "1 hr observations \n(NIRISS + NIRSpec G395)")
    ax.plot(wl, Fstar_earth + 1000*Fplan_therm_earth, label = "Star + 1000x Planet thermal", color = "C1")
    #ax.plot(wave2, y_planet2, color = "C3", label = "Planet Model")

    ax.set_yscale("log")
    ax.legend(fontsize = 16, loc = "lower left")

    ax.axvspan(wl.min(), 1.6, color="C2", alpha = 0.1)
    ax.text(1.2, 1.3e-7, "Reference\nWavelength\nRange", ha = "center", va = "top")

    ax.text(4.0, 2e-7, "Planetary\nInfrared\nExcess", ha = "center", va = "center")
    ax.annotate("", xy=(4.0, 2e-8), xytext=(4.0, 1e-7), arrowprops=dict(arrowstyle="->"))

    ax2.set_ylabel("Planet/Star Flux $(F_p/F_s)$ [ppm]")
    ax2.plot(wl, 1e6*(y_binned/Fstar_earth-1.0), color = "w", lw = 4.0, zorder = 100)
    ax2.plot(wl, 1e6*(y_binned/Fstar_earth-1.0), color = "C3", label = "Planet-to-Star Flux ($F_{pl}/F_{star}$)", zorder = 100)
    ylim = ax2.get_ylim()
    ax2.errorbar(wl, 1e6*(y_meas/Fstar_earth-1.0), yerr = 1e6*(y_err/Fstar_earth), fmt=".k", label = "Data/Stellar Model ($F_{obs} / F_{star} - 1$)")
    ax2.set_ylim(ylim[0], ylim[1])
    ax2.legend(fontsize = 16)
    #ax2.set_yscale("log")

    fig.savefig(savetag+".png", bbox_inches = "tight")

    return wl, y_binned, y_meas, y_err, XSECS

def model_wrapper(x, *theta):
    y_binned, y_star, y_planet, atm = run_pie_model_general(theta)
    return y_binned

def run_curve_fit(diff_length_scale = 0.1):
    """
    """

    print("Running curve fit...")

    # Prepare bounds
    bounds = smarter.priors.get_theta_bounds(PRIORS)
    lowers = [b[0] for b in bounds]
    uppers = [b[1] for b in bounds]
    bounds = (lowers, uppers)

    # Prepare step size for finite difference Jacobians
    bs = np.array(bounds)
    # Spacing between bounds
    delta = (bs[1:,:] - bs[:-1,:])
    # Use 1/10th of parameter length scale
    diff_step = diff_length_scale * delta.squeeze()

    # Run curve fit
    popt, pcov = sp.optimize.curve_fit(model_wrapper, wl, y_binned, sigma = y_err, p0 = THETA0,
                                       absolute_sigma = True, bounds = bounds, diff_step = diff_step)

    # Compute one standard deviation errors on the parameters
    perr = np.sqrt(np.diag(pcov))

    for i in range(len(THETA_NAMES)):
        print("%s = %.2e ± %.2e" %(THETA_NAMES[i], popt[i], perr[i]))

    return popt, pcov

def perform_initial_optimization(tag, nwalkers):
    """
    """

    # Run initial optimization
    popt, pcov = run_curve_fit()

    # Get one sigma posterior uncertainties
    perr = np.sqrt(np.diag(pcov))

    # Construct initial walker states using initial posterior estimates
    # Make gaussian ball
    gball = []
    # Loop over parameters
    for i in range(len(THETA_NAMES)):
        bnds = PRIORS[i].get_bounds()
        # if the initial likelihood is a delta function or larger than the bounds
        if (perr[i] < 1e-15) or (perr[i] > (bnds[1] - bnds[0])):
            # Use the prior
            gball.append(PRIORS[i])
        # if the prior is a gaussian
        elif hasattr(PRIORS[i], "sigma"):
            # If the prior is more constraining than the likelihood
            if (PRIORS[i].sigma < perr[i]):
                # Use the prior
                gball.append(PRIORS[i])
            else:
                # Use the gaussian posterior
                gball.append(smarter.priors.GaussianPrior(popt[i], perr[i], theta_name = THETA_NAMES[i], theta0=THETA0[i]))
        else:
            # Use the gaussian posterior
            gball.append(smarter.priors.GaussianPrior(popt[i], perr[i], theta_name = THETA_NAMES[i], theta0=THETA0[i]))

    # Get random samples from each of your parameters to initialize the walkers
    p0 = []
    # Loop until we have enough valid walker starting states
    while len(p0) < nwalkers:
        # Draw a random state vector
        theta_tmp = [dim.random_sample() for dim in gball]
        # If the state vector returns a finite log-prior probability
        if np.isfinite(smarter.priors.get_lnprior(theta_tmp, PRIORS)):
            # Add it as an initial walker state
            p0.append(theta_tmp)
    # Convert to np array
    p0 = np.array(p0)

    # Save initial optimization results
    with open(tag+"_optim.pkl", 'wb') as file:
        dill.dump((popt, pcov, perr, p0), file)

    return popt, pcov, perr, p0

def run_mcmc(tag, processes = 1, p0 = None, nsteps = 1000, nwalkers = 32,
             cache = True, overwrite = False):

    if processes > 1:
        # Use standard python multiprocessing
        Pool = multiprocessing.Pool
        pool_kwargs = {"processes" : processes}
    else:
        import schwimmbad
        Pool = schwimmbad.SerialPool
        pool_kwargs = {}

    # Open a pool
    with Pool(**pool_kwargs) as pool:

        # Randomly initialize walkers if no initial state provided
        if p0 is None:
            # Get n random samples from each of your parameter prior
            p0 = np.vstack([prior.random_sample(nwalkers) for prior in PRIORS]).T

        # Set the number of walkers and dimensionality based on
        # the initial state
        nwalkers, ndim = p0.shape

        # Create HDF5 backend to save chains?
        if cache:

            emcee_backend = str(tag) + "_emcee.h5"

            if os.path.exists(emcee_backend) and not overwrite:

                # Create an HDF5 backend
                backend = emcee.backends.HDFBackend(emcee_backend)

                # Get last sample to resume from
                p0 = backend.get_last_sample().coords
                print("Resuming MCMC using existing HDF5 backend...")

            else:

                # Create an HDF5 backend
                backend = emcee.backends.HDFBackend(emcee_backend)
                print("Created new HDF5 backend...")

                # Reset the backend
                backend.reset(nwalkers, ndim)

        else:

            # Only keep last sampler object in memory
            backend = None

        # Set the number of walkers and dimensionality based on
        # the initial state
        # REDOING this in case load from backend changed p0
        nwalkers, ndim = p0.shape

        # Set up MCMC sample object
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob_blobs,
                                        backend = backend,
                                        pool = pool)

        print("Created new EnsembleSampler...")

        # Run MCMC!
        #for _ in sampler.sample(p0, iterations = nsteps):
        #    pass
        result = sampler.run_mcmc(p0, nsteps, progress=True)

        print("MCMC finished.")

    # End of pool
    return

"""
GLOBALS
"""

# Define data and xsecs
data_tag = "w43b_exopie_data"
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
    wl, y_binned, y_meas, y_err, XSECS = generate_data(savetag = data_tag)

use_random_noise = False
if use_random_noise:
    pass
else:
    y_meas = y_binned

# Define Priors
PRIORS = [
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
    smarter.priors.GaussianPrior(0.93, 0.08, theta_name = "Rp", theta0 = THETA_DEFAULTS["Rp"]),
]

# Get state vector parameter names
THETA_NAMES = [prior.theta_name for prior in PRIORS]

# Get Truth values
THETA0 = [prior.theta0 for prior in PRIORS]

if __name__ == "__main__":

    # Run MCMC (v1.0 data)
    #run_mcmc("test1", processes = 5, nsteps = 100, nwalkers = 10)
    #run_mcmc("test2", processes = 5, nsteps = 2000, nwalkers = 10)
    #run_mcmc("test3", processes = 3, nsteps = 500, nwalkers = 12)

    # Run MCMC (v2.0 data)
    #run_mcmc("test4", processes = 3, nsteps = 100, nwalkers = 12)  # 4 parameters
    #run_mcmc("test5", processes = 3, nsteps = 200, nwalkers = 24)  # 4 parameters + 4 more stellar = 8 parameters
    #run_mcmc("test6", processes = 3, nsteps = 200, nwalkers = 27)  # 4 parameters + 4 more stellar + Rp = 9 parameters
    run_mcmc("test7", processes = 4, nsteps = 1000, nwalkers = 30)  # 4 parameters + 4 more stellar + Rp + 4 planet atm = 13 parameters
