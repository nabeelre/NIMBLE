import matplotlib.pyplot as plt
from multiprocess import Pool
from datetime import datetime
import scipy.optimize
import scipy.special
# import pandas as pd
import numpy as np
import corner
import emcee
import agama
# import time
import sys
import os

# import latte_helpers as latte
import auridesi_helpers as auridesi
import iron_helpers as iron
from deconv import getSurveyFootprintBoundary, parse_knots

np.set_printoptions(linewidth=200, precision=6, suppress=True)
np.random.seed(42)

SUBSAMPLE = False
VERBOSE = True

Grrl = 0.58   # TODO
DMerr = 0.24   # TODO? scatter in abs.magnitude
# assume that the error in distance modulus is purely due to intrinsic scatter in abs.mag
# (neglect photometric measurement errors, which are likely much smaller even at G=20.7)
bmin = 30.0   # min galactic latitude for the selection box (in degrees)
decmin = -35.0   # min declination for the selection box (degrees)
d2r = np.pi/180  # conversion from degrees to radians

min_knot = 5   # kpc
max_knot = 40  # kpc
num_knots = 5

min_r = 1   # kpc
max_r = 100  # kpc


def write_csv(data, filename, column_titles):
    np.savetxt(
        fname=figs_path+filename, X=data, delimiter=',', header=column_titles
    )


def medina24rrl_rho(radii):  # TODO, replace with BHB profile
    # https://ui.adsabs.harvard.edu/abs/2024MNRAS.531.4762M/abstract
    R_break = 18.0
    slope_inner = -2.05
    slope_outer = -4.47
    A1 = 0.67
    A2 = 1.52

    if isinstance(radii, float) or isinstance(radii, int):
        radii = [radii]

    log10dens = np.zeros(len(radii))
    for i, r in enumerate(radii):
        if r < R_break:
            log10dens[i] = A1 + slope_inner*np.log10(r/8)
        else:
            log10dens[i] = A2 + slope_outer*np.log10(r/8)

    return 10**log10dens


def medina24rrl_dlnrho(log_radii):  # TODO, replace with BHB profile
    # https://ui.adsabs.harvard.edu/abs/2024MNRAS.531.4762M/abstract
    R_break = 18.0
    slope_inner = -2.05
    slope_outer = -4.47

    if isinstance(log_radii, float) or isinstance(log_radii, int):
        log_radii = [log_radii]

    dlnrho_dlnr = np.zeros(len(log_radii))
    for i, lr in enumerate(log_radii):
        if np.exp(lr) < R_break:
            dlnrho_dlnr[i] = slope_inner
        else:
            dlnrho_dlnr[i] = slope_outer

    return dlnrho_dlnr


def parse_args(argv):
    kind = argv[1]
    global min_knot, max_knot, num_knots, min_r, max_r, Gmax, Gmin, bmin
    knot_override = None

    if kind == "auridesi":
        halonum = argv[2].lower()
        lsrdeg = argv[3].upper()

        assert (halonum in ["06", "16", "21", "23", "24", "27"])
        assert (lsrdeg in ["030", "120", "210", "300"])

        print(f"\033[1;33m**** RUNNING AURIDESI {halonum} {lsrdeg} ****\033[0m")

        figs_path = f"results/BHB/auridesi/deconv_{halonum}_{lsrdeg}/"
        true_path = f"data/auriga/H{halonum}/Au{halonum}_true.csv"

        load_params = (halonum, lsrdeg)
        load_fnc = auridesi.load_BHB

        Gmax = 19.0
        auridesi.Gmax = Gmax

        bmin = 30
        auridesi.bmin = bmin

        min_r = 1
        max_r = 100

        if len(argv) == 7:
            knot_override = parse_knots(argv[4:])
    elif kind == "iron":
        load_fnc = iron.load_BHB
        load_params = ()

        print("\033[1;33m**** RUNNING IRON ****\033[0m")

        figs_path = "results/BHB/iron/"
        true_path = None

        Gmin = 16.0
        Gmax = 19.5
        iron.Gmin = Gmin
        iron.Gmax = Gmax

        min_r = 1
        max_r = 100

        if len(argv) == 5:
            knot_override = parse_knots(argv[2:])

    if knot_override is not None:
        min_knot = knot_override[0]
        max_knot = knot_override[1]
        num_knots = knot_override[2]

    timestamp = datetime.now().strftime('%m%d%y_%H%M')
    figs_path += "_".join(map(str, (min_knot, max_knot, num_knots)))+"/"
    figs_path += timestamp + "/"

    print("Output to", figs_path)
    return kind, load_fnc, load_params, figs_path, true_path


if __name__ == "__main__":
    kind, load_fnc, load_params, figs_path, true_path = parse_args(sys.argv)

    # TODO: pull in distance, some apparent mag
    l, b, true_dens_radii, Gapp, pml, pmb, vlos, PMerr, vloserr, true_sigmar, true_sigmat, \
        lsr_info = load_fnc(*load_params, SUBSAMPLE, VERBOSE)

    external_rho = medina24rrl_rho
    external_dlnrho = medina24rrl_dlnrho

    if VERBOSE:
        print('Number of particles in survey volume:', len(l))
        print('Gmax:', Gmax)
        print('Gmin:', Gmin)
        print('bmin:', bmin)
        print('decmin:', decmin)
        print('min_r:', min_r)
        print('max_r:', max_r)

    if SUBSAMPLE:
        figs_path += "sub/"

    if not os.path.exists(figs_path):
        os.makedirs(figs_path)
        if VERBOSE:
            print("created output directory for figures at " + figs_path)
    print()

    blow, bupp, lmin, lsym = getSurveyFootprintBoundary(decmin)

    # diagnostic plot showing the stars in l,b and the selection region boundary
    if VERBOSE:
        # TODO: replace Gapp with new apparent mag from Amanda's catalog
        plt.scatter(l, b, s=2, c=Gapp, cmap='hell', vmin=Gmin, vmax=Gmax+1, edgecolors='none', rasterized=True)
        plt.colorbar(label='Gapp')
        if blow <= -bmin*d2r:  # selection region in the southern Galactic hemisphere
            bb=np.linspace(blow, -bmin*d2r, 100)
            l1=lmin(bb)
            l2=2*lsym-l1
            plt.plot(np.hstack((l1, l2[::-1], l1[0])) / d2r, np.hstack((bb, bb[::-1], bb[0])) / d2r, 'g')
        if bupp >= bmin*d2r:  # selection region in the northern Galactic hemisphere
            bb=np.linspace(bupp, bmin*d2r, 100)
            l1=lmin(bb)
            l2=2*lsym-l1
            plt.plot(np.hstack((l1, l2[::-1], l1[0])) / d2r, np.hstack((bb, bb[::-1], bb[0])) / d2r, 'g')
        plt.title(figs_path)
        plt.xlabel('galactic longitude l (degrees)')
        plt.ylabel('galactic latitude b (degrees)')
        plt.tight_layout()
        plt.savefig(figs_path+"sel_bounds.pdf", dpi=200, bbox_inches='tight')
        plt.cla()

    def modelDensity(params, truedens=False):
        # params is the array of logarithms of density at radial knots, which must monotonically decrease
        # note that since the result is invariant w.r.t. the overall amplitude of the density
        # (it is always renormalized to unity), the first element of this array may be fixed to 0
        knots_logdens = np.hstack((0, params))
        if any(knots_logdens[1:] >= knots_logdens[:-1]):
            raise RuntimeError('Density is non-monotonic')
        # represent the spherically symmetric 1d profile log(rho) as a cubic spline in log(r)
        dens_knots = np.linspace(np.log(1), np.log(200), len(knots_logdens))
        logrho = agama.CubicSpline(dens_knots, knots_logdens, reg=True)  # ensure monotonic spline (reg)
        # check that the density profile has a finite total mass
        # (this is not needed for the fit, because the normalization is computed over the accessible volume,
        # but it generally makes sense to have a physically valid model for the entire space).
        slope_in  = logrho(dens_knots[0], der=1)  # d[log(rho)]/d[log(r)], log-slope at the lower radius
        slope_out = logrho(dens_knots[-1], der=1)
        if slope_in <= -3.0 or slope_out >= -3.0:
            raise RuntimeError('Density has invalid asymptotic slope: inner=%.2f, outer=%.2f' % (slope_in, slope_out))

        return logrho

    def modelSigma(params):
        # params is the array of log(sigma(r)) at radial knots (applicable to both velocity components)
        return agama.CubicSpline(knots_logr, params)

    def likelihood(params):
        # function to be maximized in the MCMC and deterministic optimization
        params_sigmar = params[0:len(knots_logr)]
        params_sigmat = params[len(knots_logr):]
        try:
            # construct intrinsic velocity dispersion profiles of the model
            sigmar2   = np.exp(2*modelSigma(params_sigmar)(logr_samp))  # squared radial velocity dispersion
            sigmat2   = np.exp(2*modelSigma(params_sigmat)(logr_samp))  # squared tangential --"--
            sigmaboth = np.vstack((sigmar2, sigmat2))  # shape: (2, nbody*nsamples)
            # convert these profiles to the values of line-of-sight velocity dispersion at each data sample
            cov_vlos  = np.einsum('kp,kp->p', mat_vlos, sigmaboth)
            # same for the PM dispersion profiles - diagonal and off-diagonal elements of the PM covariance matrix
            cov_pmll, cov_pmbb, cov_pmlb = np.einsum('ikp,kp->ip', mat_pm, sigmaboth)
            # add individual observational errors for each data sample
            cov_vlos += vloserr2_samp
            cov_pmll += pmlerr2_samp  # here add to diagonal elements of PM cov matrix only,
            cov_pmbb += pmberr2_samp  # but with the actual Gaia data should also use the off-diagonal term
            det_pm    = cov_pmll * cov_pmbb - cov_pmlb**2  # determinant of the PM cov matrix
            # compute likelihoods of Vlos and PM values of each data sample, taking into account obs.errors
            like_vlos = cov_vlos**-0.5 * np.exp(-0.5 * vlos_samp**2 / cov_vlos)
            like_pm   = det_pm**-0.5 * np.exp(
                -0.5 / det_pm * (pml_samp**2 * cov_pmbb + pmb_samp**2 * cov_pmll - 2 * pml_samp * pmb_samp * cov_pmlb)
            )
            # the overall log-likelihood of the model:
            # first average the likelihoods of all sample points for each star -
            # this corresponds to marginalization over distance uncs, also propagated to PM space;
            # then sum up marginalized log-likelihoods of all stars.
            # at this stage may also add a prior if necessary
            loglikelihood = np.sum(np.log(
                np.mean((like_pm * like_vlos).reshape(npoints, nsamples), axis=1)
            ))
            # print("%s => %.2f" % (params, loglikelihood))
            if not np.isfinite(loglikelihood):
                loglikelihood = -np.inf
            return loglikelihood
        except Exception as ex:
            if VERBOSE:
                print("%s => %s" % (params, ex))
            return -np.inf

    def plotprofiles(chain, plotname=''):
        # Density plots
        fig = plt.figure(figsize=(7, 7))
        axs = fig.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw=dict(hspace=0, height_ratios=[3, 1]))

        r  = np.logspace(np.log10(min_r), np.log10(max_r), 200)
        lr = np.log(r)
        rhist = np.logspace(np.log10(min_r), np.log10(max_r), 81)
        lrhist = np.log(rhist)
        if true_path is not None:
            num_true_knots = 12
            if kind == "auridesi":
                num_true_knots = 4
            knots_logr_true = np.linspace(np.log(1), np.log(200), num_true_knots)
            S = agama.splineLogDensity(knots_logr_true, x=np.log(true_dens_radii), w=np.ones(len(true_dens_radii)), infLeft=True, infRight=True)
            trueparams_dens = np.log((np.exp(S(knots_logr_true))) / (4.0 * np.pi * (np.exp(knots_logr_true)**3)))
            trueparams_dens = trueparams_dens[1:] - trueparams_dens[0]  # set the first element of array to zero and exclude it
            truedens = np.exp(modelDensity(trueparams_dens)(lr))
        else:
            truedens = None

        # main plot
        if (plotname == 'converged') or VERBOSE:
            if true_path is not None:
                axs[0].plot(r, truedens, 'k-', label='True')
                axs[0].set_ylim(min(truedens)*0.2, max(truedens)*2)
            axs[0].plot(r, external_rho(r), c='k', linestyle='dotted', label='assumed density profile')
            # TODO: add plotting of assumed mass profile here
            # # retrieve density profiles of each model in the chain, and compute median and 16/84 percentiles
            # results = np.zeros((len(chain), len(r)))
            # for i in range(len(chain)):
            #     results[i] = np.exp(modelDensity(chain[i, 0:len(knots_logr)-1])(lr))
            # dens_low, dens_med, dens_upp = np.percentile(results, [16,50,84], axis=0)
            # # plot the model profiles with 1sigma confidence intervals
            # axs[0].fill_between(r, dens_low, dens_upp, alpha=0.3, lw=0, color='r')
            # axs[0].plot(r, dens_med, color='r', label='MCMC Fit')
            count_obs = np.histogram(logr_obs, bins=lrhist)[0]
            rho_obs = count_obs / (4 * np.pi * (lrhist[1:]-lrhist[:-1]) * (rhist[1:]*rhist[:-1])**1.5 * len(logr_obs))
            axs[0].plot(np.repeat(rhist,2)[1:-1], np.repeat(rho_obs, 2), 'r--', label='with SFs and Errors')

            axs[0].set_title(figs_path)
            axs[0].set_ylabel('3d density of tracers')
            axs[0].set_yscale('log')
            axs[0].set_xlim(min_r, max_r)
            axs[0].legend(loc='upper right', frameon=False)

            # percent error
            if true_path is not None:
                percerr = 100*((dens_med - truedens) / truedens)
                axs[1].plot(r, percerr, 'r')
            axs[1].axhline(0, c='gray', linestyle='dashed')
            axs[1].set_ylim(-40,40)

            axs[1].set_xlabel('Galactocentric radius (kpc)')
            axs[1].set_ylabel('percent error')
            axs[1].set_xscale('log')

            plt.tight_layout()
            plt.savefig(figs_path+plotname+'_dens.pdf', dpi=200, bbox_inches='tight')
            plt.cla()

            # Sigma plots
            fig = plt.figure(figsize=(12,7))
            axs = fig.subplots(nrows=2, ncols=2, sharex=True, gridspec_kw=dict(hspace=0, wspace=0, height_ratios=[3, 1], width_ratios=[1,1]))
            axs[0,0].text(x=45, y=470, s='Radial', size=18)
            axs[0,1].text(x=45, y=470, s='Tangential', size=18)

            # again collect the model profiles and plot median and 16/84 percentile confidence intervals
            results_r, results_t = np.zeros((2, len(chain), len(r)))
            for i in range(len(chain)):
                results_r[i] = np.exp(modelSigma(chain[i, :len(knots_logr)])(lr))
                results_t[i] = np.exp(modelSigma(chain[i, len(knots_logr):])(lr))
            low_r, med_r, upp_r = np.percentile(results_r, [16,50,84], axis=0)
            axs[0,0].fill_between(r, low_r, upp_r, alpha=0.3, lw=0, color='g')
            axs[0,0].plot(r, med_r, color='g', label='MCMC Fit $\sigma_\mathrm{rad}$')
            low_t, med_t, upp_t = np.percentile(results_t, [16,50,84], axis=0)
            axs[0,1].fill_between(r, low_t, upp_t, alpha=0.3, lw=0, color='b')
            axs[0,1].plot(r, med_t, color='b', label='MCMC Fit $\sigma_\mathrm{tan}$')

            if true_path is not None:
                truesigr = true_sigmar(lr)**0.5
                truesigt = true_sigmat(lr)**0.5
                axs[0,0].plot(r, truesigr, 'k-', label='True $\sigma_\mathrm{rad}$')
                axs[0,1].plot(r, truesigt, 'k-', label='True $\sigma_\mathrm{tan}$')

                percerr_r = 100*((med_r - truesigr) / truesigr)
                lowerr_r = 100*((low_r - truesigr) / truesigr)
                upperr_r = 100*((upp_r - truesigr) / truesigr)
                axs[1,0].plot(r, percerr_r, c='g')
                axs[1,0].axhline(0, c='gray', linestyle='dashed')
                axs[1,0].fill_between(r, lowerr_r, upperr_r, alpha=0.3, lw=0, color='g')
                percerr_t = 100*((med_t - truesigt) / truesigt)
                lowerr_t = 100*((low_t - truesigt) / truesigt)
                upperr_t = 100*((upp_t - truesigt) / truesigt)
                axs[1,1].plot(r, percerr_t, c='b')
                axs[1,1].axhline(0, c='gray', linestyle='dashed')
                axs[1,1].fill_between(r, lowerr_t, upperr_t, alpha=0.3, lw=0, color='b')

            # and plot the observed radial/tangential dispersions, which are affected by distance errors
            # and broadened by PM errors (especially the tangential dispersion)
            sigmar_obs = (np.histogram(logr_obs, bins=lrhist, weights=vr_obs**2)[0] / count_obs)**0.5
            sigmat_obs = (np.histogram(logr_obs, bins=lrhist, weights=vt_obs**2)[0] / count_obs)**0.5
            axs[0,0].plot(np.repeat(rhist,2)[1:-1], np.repeat(sigmar_obs, 2), 'g--', label='with SFs and Errors $\sigma_\mathrm{rad}$', alpha=0.3)
            axs[0,1].plot(np.repeat(rhist,2)[1:-1], np.repeat(sigmat_obs, 2), 'b--', label='with SFs and Errors $\sigma_\mathrm{tan}$', alpha=0.3)

            #upper_bound = max(np.hstack((upp_r, upp_t, sigmar_obs, sigmat_obs, truesigr, truesigt)))*1.1
            axs[0,0].set_ylim(-20,500)
            axs[0,1].set_ylim(-20,500)
            axs[0,1].set_yticklabels([])
            axs[1,0].set_ylim(-40,40)
            axs[1,1].set_ylim(-40,40)
            axs[1,1].set_yticklabels([])

            axs[0,0].set_xlim(min_r, max_r*1.1)
            axs[0,0].set_title(figs_path)
            axs[1,0].set_xlabel('Galactocentric radius (kpc)')
            axs[1,1].set_xlabel('Galactocentric radius (kpc)')
            axs[0,0].set_ylabel('velocity dispersion of tracers')
            axs[1,0].set_ylabel('percent error (%)')
            axs[0,0].legend(loc='upper left', frameon=False)
            axs[0,1].legend(loc='upper left', frameon=False)

            plt.tight_layout()
            plt.savefig(figs_path+plotname+'_sigs.pdf', dpi=200, bbox_inches='tight')
            plt.cla()

        if plotname == 'converged':
            # True and MCMC velocity dispersion and density profiles
            if true_path is not None:
                profiles = np.stack([r, truedens, low_r, med_r, upp_r, truesigr, low_t, med_t, upp_t, truesigt], axis=1)
                write_csv(profiles, "veldisp_dens_profiles.csv", "r, truedens, low_r, med_r, upp_r, truesigr, low_t, med_t, upp_t, truesigt")
            else:
                profiles = np.stack([r, low_r, med_r, upp_r, low_t, med_t, upp_t], axis=1)
                write_csv(profiles, "veldisp_dens_profiles.csv", "r, low_r, med_r, upp_r, low_t, med_t, upp_t")

            # histograms of velocity dispersion and density after obs effects
            histograms = np.stack([np.repeat(rhist,2)[1:-1], np.repeat(rho_obs, 2), np.repeat(sigmar_obs, 2), np.repeat(sigmat_obs, 2)], axis=1)
            write_csv(histograms, "veldisp_dens_hists.csv", "rgrid, dens, sigmar, sigmat")

    npoints = len(l)

    # convert l,b,dist.mod. of all stars into logarithm of Galactocentric radius (observed, not true)
    # unit conversion: degrees to radians for l,b,  mas/yr to km/s/kpc for PM
    dist_obs = 10**(0.2*(Gapp-Grrl)-2)  # TODO: pull distance from amanda's sample
    x, y, z, vx, vy, vz = agama.getGalactocentricFromGalactic(
        l*d2r, b*d2r, dist_obs, pml*4.74, pmb*4.74, vlos, *lsr_info)
    logr_obs = 0.5 * np.log(x**2 + y**2 + z**2)
    vr_obs = (x*vx+y*vy+z*vz) / np.exp(logr_obs)
    vt_obs = (0.5 * (vx**2+vy**2+vz**2 - vr_obs**2))**0.5

    # create random samples from distance modulus unc for each star and convert to Galactocentric r
    nsamples = 20  # number of random samples per star
    Gsamp = (np.random.normal(size=(npoints, nsamples)) * DMerr + Gapp[:, None]).reshape(-1)
    dist_samp = 10**(0.2*(Gsamp-Grrl)-2)  # TODO: instead, impose error directly in distance - based on spread for GC / dwarf?
    x, y, z = agama.getGalactocentricFromGalactic(
        np.repeat(l * d2r, nsamples), np.repeat(b * d2r, nsamples),
        dist_samp, galcen_distance=lsr_info[0], galcen_v_sun=lsr_info[1], z_sun=lsr_info[2]
    )
    R = (x**2 + y**2)**0.5
    r = (x**2 + y**2 + z**2)**0.5  # array of samples for Galactocentric radius
    logr_samp = np.log(r)

    # a rather clumsy way of constructing the matrices describing how the intrinsic 3d velocity dispersions
    # are translated to the Vlos and PM dispersions at each data sample:
    # first compute the expected mean values (pml, pmb, vlos) for a star at rest at a given distance,
    # then repeat the exercise 3 times, setting one of velocity components (v_r, v_theta, v_phi)
    # to 1 km/s, and subtract from the zero-velocity projection.
    vel0 = np.array(agama.getGalacticFromGalactocentric(x, y, z,
                                                        x*0, y*0, z*0,
                                                        *lsr_info)[3:6])
    velr = np.array(agama.getGalacticFromGalactocentric(x, y, z,
                                                        x/r, y/r, z/r,
                                                        *lsr_info)[3:6]) - vel0
    velt = np.array(agama.getGalacticFromGalactocentric(x, y, z,
                                                        z/r*x/R, z/r*y/R, -R/r,
                                                        *lsr_info)[3:6]) - vel0
    velp = np.array(agama.getGalacticFromGalactocentric(x, y, z,
                                                        -y/R, x/R, 0*r,
                                                        *lsr_info)[3:6]) - vel0

    # matrix of shape (2, npoints*nsamples) describing how the two intrinsic velocity dispersions
    # in 3d Galactocentric coords translate to the line-of-sight velocity dispersion at each sample point
    mat_vlos = np.array([ velr[2]**2, velt[2]**2 + velp[2]**2 ])
    # same for the PM dispersions: this is a 2x2 symmetric matrix for each datapoint,
    # characterized by two diagonal and one off-diagonal elements,
    # and each element, in turn, is computed from the two Galactocentric intrinsic velocity dispersions
    mat_pm   = np.array([
        [velr[0]*velr[0], velt[0]*velt[0] + velp[0]*velp[0]],
        [velr[1]*velr[1], velt[1]*velt[1] + velp[1]*velp[1]],
        [velr[0]*velr[1], velt[0]*velt[1] + velp[0]*velp[1]]
    ]) / 4.74**2

    # difference between the measured PM and Vlos values and expected mean values at each sample
    # (the latter correspond to a zero 3d velocity, translated to the Heliocentric frame)
    pml_samp = np.repeat(pml,  nsamples) - vel0[0] / 4.74
    pmb_samp = np.repeat(pmb,  nsamples) - vel0[1] / 4.74
    vlos_samp = np.repeat(vlos, nsamples) - vel0[2]
    # vectors of PM and Vlos errors for each sample, to be added to the model covariance matrices
    pmlerr2_samp = np.repeat(PMerr, nsamples)**2
    # here is identical to pml, but in general may be different
    pmberr2_samp = np.repeat(PMerr, nsamples)**2
    vloserr2_samp = np.repeat(vloserr, nsamples)**2

    # knots in Galactocentric radius (minimum is imposed by our cut |b|>30, maximum - by the extent of data)
    knots_logr = np.linspace(np.log(min_knot), np.log(max_knot), num_knots)

    # initial values of parameters
    # log of radial velocity dispersion values at the radial knots
    params_sigmar = np.zeros(len(knots_logr)) + 5.0
    params_sigmat = np.zeros(len(knots_logr)) + 5.0   # same for tangential dispersion
    params = np.hstack((params_sigmar, params_sigmat))
    paramnames = ['sigmar(r=%4.1f)' % r for r in np.exp(knots_logr)] + \
        ['sigmat(r=%4.1f)' % r for r in np.exp(knots_logr)]
    prevmaxloglike = -np.inf
    prevavgloglike = -np.inf
    # first find the best-fit model by deterministic optimization algorithm,
    # restarting it several times until it seems to arrive at the global minimum
    num_tries = 0
    while True:
        if VERBOSE:
            print('\033[1;37mStarting deterministic search\033[0m')
        # minimization algorithm - so provide a negative likelihood to it
        params = scipy.optimize.minimize(
            lambda x: -likelihood(x), params, method='Nelder-Mead',
            options=dict(maxfev=500)
        ).x
        maxloglike = likelihood(params)
        if maxloglike - prevmaxloglike < 1.0:
            if VERBOSE:
                for i in range(len(params)):
                    print('%s = %8.4g' % (paramnames[i], params[i]))
                print('Converged')
            break
        elif VERBOSE:
            print('Improved log-likelihood by %f' % (maxloglike - prevmaxloglike))
        prevmaxloglike = maxloglike

        num_tries += 1

        if num_tries >= 100:
            print("Too many tries in deterministic search")
            exit()

    # show profiles and wait for the user to marvel at them
    plotprofiles(params.reshape(1,-1), "preMCMC")

    # then start a MCMC around the best-fit params
    paramdisp= np.ones(len(params))*0.1  # spread of initial walkers around best-fit params
    nwalkers = 2*len(params)   # minimum possible number of walkers in emcee
    nsteps   = 500  # 1000
    walkers  = np.empty((nwalkers, len(params)))
    numtries = 0
    for i in range(nwalkers):
        while numtries<10000:   # ensure that we initialize walkers with feasible values
            walker = params + np.random.randn(len(params))*paramdisp
            if np.isfinite(likelihood(walker)):
                walkers[i] = walker
                break
            numtries+=1
    if numtries>=10000:
        raise RuntimeError('cannot initialize MCMC')
    with Pool() as pool:
        # numthreads = nwalkers//2   # parallel threads in emcee - make sure you don't clog up your machine!
        sampler  = emcee.EnsembleSampler(nwalkers, len(params), likelihood, pool=pool)
        if VERBOSE:
            print('\033[1;37mStarting MCMC search\033[0m')
        converged = False
        iter = 0
        while not converged:  # run several passes until log-likelihood stabilizes (convergence is reached)
            sampler.run_mcmc(walkers, nsteps, progress=True)
            walkers = sampler.chain[:,-1]
            chain   = sampler.chain[:,-nsteps:].reshape(-1, len(params))
            maxloglike = np.max (sampler.lnprobability[:,-nsteps:])
            avgloglike = np.mean(sampler.lnprobability[:,-nsteps:])
            walkll = sampler.lnprobability[:,-1]
            if VERBOSE:
                for i in range(len(params)):
                    print('%s = %8.4g +- %7.4g' % (paramnames[i],
                                                   np.mean(chain[:, i]),
                                                   np.std(chain[:, i])))
                print('max loglikelihood: %.2f, average: %.2f' % (maxloglike, avgloglike))
            converged = abs(maxloglike-prevmaxloglike) < 1.0 and \
                abs(avgloglike-prevavgloglike) < 2.0
            prevmaxloglike = maxloglike
            prevavgloglike = avgloglike
            if converged:
                if VERBOSE:
                    print('\033[1;37mConverged\033[0m')
                plotprofiles(chain[::20], "converged")

            # produce diagnostic plots after each MCMC episode:
            # 1. evolution of parameters along the chain for each walker
            if VERBOSE:
                axes = plt.subplots(len(params)+1, 1, sharex=True, figsize=(10, 10))[1]
                for i in range(len(params)):
                    axes[i].plot(sampler.chain[:, :, i].T, color='k', alpha=0.3)
                    axes[i].set_xticklabels([])
                    axes[i].set_ylabel(paramnames[i])
                axes[0].set_title(figs_path)
                axes[-1].plot(sampler.lnprobability.T, color='k', alpha=0.3)
                axes[-1].set_ylabel('likelihood')   # bottom panel is the evolution of likelihood
                axes[-1].set_ylim(maxloglike-3*len(params), maxloglike)
                plt.tight_layout(h_pad=0)
                plt.subplots_adjust(hspace=0, wspace=0)
                plt.savefig(figs_path+"param_evol_iter"+str(iter)+".png",
                            dpi=200, bbox_inches='tight')
                plt.cla()
                # 2. corner plot - covariances of all parameters
                corner.corner(chain, quantiles=[0.16, 0.5, 0.84],
                              labels=paramnames, show_titles=True)
                plt.title(figs_path)
                plt.savefig(figs_path+"corner_iter"+str(iter)+".png", dpi=200, bbox_inches='tight')
                plt.cla()
            # 3. density and velocity dispersion profiles - same as before
            if not converged:
                plotprofiles(chain[::20], "MCMC_iter"+str(iter))
            iter += 1
    plt.cla()

    # now, plug resulting density and sigma profiles into the jeans equation
    # one mass profile for each trial of MCMC

    r  = np.logspace(np.log10(min_r), np.log10(max_r), 201)
    lr = np.log(r)
    G = 4.3e-6  # (kpc km2) / (s2 Msun)

    def frac_error(r_est, r_true, M_est, M_true):
        frac_error = np.zeros(len(r_est))
        for i in range(len(r_est)):
            match_idx = (np.abs(r_true - r_est[i])).argmin()
            frac_error[i] = (M_est[i]-M_true[match_idx])/M_true[match_idx]
        return frac_error

    # Read the _true.csv file written by read_latte.ipynb
    # files with '_true.csv' have the following properties:
    # radius in kpc, mass in Msun
    # columns arranged as [r, Menc] where Menc is the true enclosed mass
    # rows are sorted by increasing radius
    if true_path:
        r_true, M_true = np.loadtxt(fname=true_path, delimiter=',',
                                    skiprows=1, unpack=True)

    # Thin the chain
    chain_smpl = chain[::20]

    dlnrho, dlnsigr, sigr, sigt = np.zeros((4, len(chain_smpl), len(r)))
    for i in range(len(chain_smpl)):
        dlnrho[i] = external_dlnrho(lr)
        dlnsigr[i] = modelSigma(chain_smpl[i, :len(knots_logr)])(lr, der=1)
        sigr   [i] = np.exp(modelSigma(chain_smpl[i, :len(knots_logr)])(lr))
        sigt   [i] = np.exp(modelSigma(chain_smpl[i, len(knots_logr):])(lr))

    Mencs, betas = np.zeros((2, len(chain_smpl), len(r)))
    for i in range(len(chain_smpl)):
        betas[i] = 1 - (sigt[i]**2 / sigr[i]**2)
        Mencs[i] = -(sigr[i]**2 * r / G)*(dlnrho[i] + 2*dlnsigr[i] + 2*betas[i])
    Menc_low, Menc_med, Menc_upp = np.percentile(Mencs, [16,50,84], axis=0)
    beta_low, beta_med, beta_upp = np.percentile(betas, [16,50,84], axis=0)

    # Anisotropy plot
    if VERBOSE:
        fig = plt.figure(figsize=(7,7))
        plt.plot(r, beta_med, label=r"$\beta$")
        plt.fill_between(r, beta_low, beta_upp, alpha=0.3, label=r'$\pm1\sigma$ interval')
        plt.axhline(0, c='k', label=r"$\beta=0$")
        plt.title(figs_path)
        plt.xlabel('Galactocentric radius (kpc)')
        plt.ylabel(r"Anisotropy ($\beta$)", fontsize=18)
        plt.ylim([-1.0, 1.0])
        plt.legend()
        plt.tight_layout()
        plt.savefig(figs_path+'anisotropy.pdf', dpi=200, bbox_inches='tight')
        plt.cla()

    # Mass enclosed plot
    plt.rcParams.update({'font.size': 18})
    plt.rcParams['agg.path.chunksize'] = 10000  # overflow error on line 835 without this
    fig = plt.figure(figsize=(15, 10))
    axs = fig.subplots(nrows=2, ncols=1, sharex=True,
                       gridspec_kw=dict(hspace=0, height_ratios=[3, 1]))

    description = kind+" ".join(load_params)

    if "m12f" in description:
        color = '#4a0078'
    elif "m12i" in description:
        color = '#157F1F'
    elif "m12m" in description:
        color = '#931621'
    elif "06" in description:
        color = 'g'
    elif "21" in description:
        color = 'r'
    elif "24" in description:
        color = 'b'
    elif "iron" in description:
        color = 'mediumblue'
    else:
        color = 'gold'

    axs[0].plot(r, Menc_med, c=color, linewidth=2.5, label='Jeans estimate')
    axs[0].fill_between(r, Menc_low, Menc_upp, color=color, alpha=0.3,
                        lw=0, label=r'$\pm1\sigma$ interval')

    if true_path is not None:
        axs[0].plot(r_true, M_true, c='k', linewidth=1.5, linestyle='dashed', label='True')

        axs[1].plot(r, frac_error(r, r_true, Menc_med, M_true), c=color, linewidth=2.0)
        axs[1].fill_between(r, frac_error(r, r_true, Menc_low, M_true),
                            frac_error(r, r_true, Menc_upp, M_true), color=color,
                            alpha=0.3, lw=0)

    axs[0].set_title(figs_path)
    axs[0].set_ylim([0.9*min(Menc_med), 1.1*Menc_med[-1]])
    axs[1].axhline(0, c='k', linewidth=1)
    axs[1].axhline(0.2, c='k', linewidth=0.5, linestyle='dotted')
    axs[1].axhline(-0.2, c='k', linewidth=0.5, linestyle='dotted')
    axs[1].set_xlim([min_r, max_r])
    axs[0].set_ylabel(r"$M(<r) (M_{\odot})$", size=24)
    axs[1].set_xlabel('Galactocentric Radius (kpc)', size=24)
    axs[1].set_ylabel('Fractional Error', size=20)
    axs[0].legend()
    axs[1].set_ylim([-0.40, 0.40])
    axs[1].set_yticks([-0.2, 0, 0.2])

    for ax in axs:
        ax.label_outer()

    plt.tight_layout()
    plt.savefig(figs_path+'mass_enc.pdf', dpi=200, bbox_inches='tight')
    plt.cla()

    finals_data = np.stack([r, Menc_low, Menc_med, Menc_upp, beta_low, beta_med, beta_upp], axis=1)
    write_csv(finals_data, "Menc_beta_final.csv", "r, Menc_low, Menc_med, Menc_upp, beta_low, beta_med, beta_upp")

    print(f"FINISHED AT {figs_path}\n")
