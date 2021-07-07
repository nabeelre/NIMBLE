import os, numpy, scipy.special, scipy.optimize, emcee, corner, agama, time, matplotlib.pyplot as plt
from pygaia.errors.astrometric import proper_motion_uncertainty, total_proper_motion_uncertainty
from multiprocessing import Pool
numpy.set_printoptions(linewidth=200, precision=6, suppress=True)
numpy.random.seed(42)

Gmax  = 20.7   # max apparent G magnitude for a RRL star to enter the selection
Gmin  = 16.0   # min apparent G magnitude (exclude nearby stars)
Grrl  = 0.58   # abs.magnitude of RRL (mean value; the actual value for each star is scattered around it)
DMerr = 0.24   # scatter in abs.magnitude
# assume that the error in distance modulus is purely due to intrinsic scatter in abs.mag
# (neglect photometric measurement errors, which are likely much smaller even at G=20.7)
bmin  = 30.0   # min galactic latitude for the selection box (in degrees)
decmin=-35.0   # min declination for the selection box (degrees)

d2r   = numpy.pi/180  # conversion from degrees to radians

# TO DO:
# true sigma and density profiles for Latte datasets -> combine both tangential dispersions
# LSR information for latte datasets and coordinate rotations where necessary
# test on indu.astro

# Notes:
# Check to see what total_proper_motion_uncertainty is ... sum or average or something else?
    # total_proper_motion_uncertainty is a simple average of the PM RAcosdec and PM Dec uncertainties
    # So, no need to divide its output by two when using it as PMerr: a single unc for either direction
# emcee parallelization will probably only work with newer emcee versions
# combine makeMock & readMock to reduce repeated code
# DESI lower mag limit is 16? https://arxiv.org/pdf/2010.11284.pdf - says 16 in r mag


# def cartesian_to_spherical(xpos, ypos, zpos, xVel, yVel, zVel):
#     numParticles = len(xpos)
#     sinTheta = numpy.zeros(numParticles)
#     cosTheta = numpy.zeros(numParticles)

#     sinPhi = numpy.zeros(numParticles)
#     cosPhi = numpy.zeros(numParticles)

#     for i in range(0, numParticles):
#         sinTheta[i] = numpy.sqrt(xpos[i]**2 + ypos[i]**2) / numpy.sqrt(xpos[i]**2 + ypos[i]**2 + zpos[i]**2)
#         cosTheta[i] = zpos[i] / numpy.sqrt(xpos[i]**2 + ypos[i]**2 + zpos[i]**2)
#         sinPhi[i] = ypos[i] / numpy.sqrt(xpos[i]**2 + ypos[i]**2)
#         cosPhi[i] = xpos[i] / numpy.sqrt(xpos[i]**2 + ypos[i]**2)

#     rVel = numpy.zeros(numParticles)
#     tVel = numpy.zeros(numParticles)
#     pVel = numpy.zeros(numParticles)

#     for i in range(0, numParticles):
#         conversionMatrix = [[sinTheta[i] * cosPhi[i], sinTheta[i] * sinPhi[i],  cosTheta[i]],
#                             [cosTheta[i] * cosPhi[i], cosTheta[i] * sinPhi[i], -sinTheta[i]],
#                             [       -sinPhi[i]      ,        cosPhi[i]       ,        0    ]]

#         velMatrix = [ [xVel[i]], [yVel[i]], [zVel[i]] ]

#         sphereVels = numpy.matmul(conversionMatrix, velMatrix)

#         rVel[i] = sphereVels[0]
#         tVel[i] = sphereVels[1]
#         pVel[i] = sphereVels[2]

#     return rVel, tVel, pVel


def loadMock(datasetType, gaiaRelease, density=None, potential=None, beta0=None, r_a=None, nbody=None, lattesim=None):
    rr = numpy.logspace(0, 2, 15)
    xyz= numpy.column_stack((rr, rr*0, rr*0))
    if datasetType == 'agama':
         # create a spherical anisotropic DF and compute its true velocity dispersions
        df = agama.DistributionFunction(type='quasispherical',
            density=density, potential=potential, beta0=beta0, r_a=r_a)
        gm = agama.GalaxyModel(potential, df)
        
        sig= gm.moments(xyz, dens=False, vel=False, vel2=True)
        # represent sigma profiles as cubic splines for log(sigma) as a function of log(r)
        true_sigmar = agama.CubicSpline(numpy.log(rr), numpy.log(sig[:,0]**0.5))
        true_sigmat = agama.CubicSpline(numpy.log(rr), numpy.log(sig[:,1]**0.5))

        # sample 6d points from the model (over the entire space)
        xv = gm.sample(nbody)[0]
        radii = numpy.sqrt(xv[:,0]**2 + xv[:,1]**2 + xv[:,2]**2)

        # rvel, tvel, pvel = cartesian_to_spherical(xv[:,0], xv[:,1], xv[:,2], xv[:,3], xv[:,4], xv[:,5])

        # rvelsq = numpy.asarray([i**2 for i in rvel])
        # tvelsq = numpy.asarray([i**2 for i in tvel])
        # pvelsq = numpy.asarray([i**2 for i in pvel])
        
        # true_sigmar    = agama.splineApprox(numpy.log(rr), numpy.log(radii), numpy.log(rvelsq)*0.5)
        # true_sigmatheta2 = lambda lr: numpy.sqrt(agama.splineApprox(numpy.log(rr), numpy.log(radii), numpy.log(tvelsq))(lr))
        # true_sigmaphi2   = lambda lr: numpy.sqrt(agama.splineApprox(numpy.log(rr), numpy.log(radii), numpy.log(pvelsq))(lr))
        # true_sigmat     = lambda lr: (true_sigmaphi2(lr) + true_sigmatheta2(lr)) / 2

        # plt.plot(rr, numpy.exp(true_sigmar2(numpy.log(rr))), c='r', linestyle='dashed')
        # plt.plot(numpy.log(rr), true_sigmaphi2(numpy.log(rr)))
        # plt.plot(numpy.log(rr), true_sigmatheta2(numpy.log(rr)))
        # plt.plot(rr, numpy.exp(true_sigmat2(numpy.log(rr))), c='b', linestyle='dashed')

        # plt.plot(rr, numpy.exp(true_sigmar(numpy.log(rr))), c='r', linestyle='solid')
        # plt.plot(rr, numpy.exp(true_sigmat(numpy.log(rr))), c='b', linestyle='solid')
        
        # plt.savefig("testing.png")
        # plt.show()
        # exit(1)
        
    if datasetType == 'latte':
        # dataset for m12f available at https://drive.google.com/file/d/1Z8lQEdPeX1995WDJsc1qxeh07ZDLBZXr/view?usp=sharing
        # formatted such that rows represent particles and columns represent different quantites
        # col 0-5: cartesian positions and velocities, col 6: particle mass, col 7: galactocentric spherical radius
        # col 8-10: square of spherical velocity components - r (radial), theta (polar), phi (azimuthal)
        x, y, z, vx, vy, vz, mass, radii, rvelsq, tvelsq, pvelsq = numpy.loadtxt(f"latte/{lattesim}/{lattesim}_chem-1.5_full.csv", unpack=True, skiprows=1, delimiter=',')
        xv = numpy.column_stack((x, y, z, vx, vy, vz))
        nbody = len(x)

        # represent sigma profiles as cubic splines for log(sigma) as a function of log(r)
        sorter = numpy.argsort(radii)
        radii = radii[sorter]
        rvelsq = rvelsq[sorter]
        tvelsq = tvelsq[sorter]
        pvelsq = pvelsq[sorter]

        true_sigmar     = agama.splineApprox(numpy.log(rr), numpy.log(radii), numpy.log(rvelsq)*0.5)
        true_sigmatheta = lambda lr: numpy.sqrt(agama.splineApprox(numpy.log(rr), numpy.log(radii), numpy.log(tvelsq))(lr))
        true_sigmaphi   = lambda lr: numpy.sqrt(agama.splineApprox(numpy.log(rr), numpy.log(radii), numpy.log(pvelsq))(lr))
        true_sigmat     = lambda lr: (true_sigmaphi(lr) + true_sigmatheta(lr)) / 2

    l,b,dist,pml,pmb,vlos = agama.getGalacticFromGalactocentric(*xv.T)
    ra, dec, pmra, pmdec  = agama.transformCelestialCoords(agama.fromGalactictoICRS, l, b, pml, pmb)
    l   /=d2r;  b   /=d2r   # convert from radians to degrees
    ra  /=d2r;  dec /=d2r
    pml /=4.74; pmb /=4.74  # convert from km/s/kpc to mas/yr
    pmra/=4.74; pmdec/=4.74

    # impose spatial selection based on the survey footprint
    filt = (abs(b) >= bmin) * (dec >= decmin)

    # compute apparent G-band magnitude and impose a cut Gmin<G<Gmax
    Gabs = Grrl + numpy.random.normal(size=nbody) * DMerr  # abs.mag with scatter
    Gapp = Gabs + 5*numpy.log10(dist) + 10  # apparent magnitude
    filt *= (Gapp > Gmin) * (Gapp < Gmax)

    # Doesnt work with current cov matrix set up
    # pmracosdec_err, pmdec_err = proper_motion_uncertainty(Gapp, release=gaiaRelease)  # uas/yr
    # pmra_err *= 0.001; pmdec_err *= 0.001  # uas/yr -> mas/yr

    PMerr = total_proper_motion_uncertainty(Gapp, release=gaiaRelease)  # uas/yr
    PMerr *= 0.001  # uas/yr -> mas/yr

    pmra  += numpy.random.normal(size=nbody) * PMerr
    pmdec += numpy.random.normal(size=nbody) * PMerr

    # RA, Dec back to radians
    # go back to galactic coords
    ra *= d2r
    dec *= d2r
    l, b, pml, pmb = agama.transformCelestialCoords(agama.fromICRStoGalactic, ra, dec, pmra, pmdec)
    l /= d2r
    b /= d2r

    # add Vlos errors
    vloserr = numpy.ones(nbody) * 2.0
    vlos += numpy.random.normal(size=nbody) * vloserr

    return (l[filt], b[filt], radii, Gapp[filt], pml[filt], pmb[filt], vlos[filt], PMerr[filt], vloserr[filt],
        true_sigmar, true_sigmat)


def getSurveyFootprintBoundary(decmin):
    # determine the range of l as a function of b, resulting from a cut in declination dec>=decmin
    # (note: the approach is tuned specifically to a constant boundary in declination,
    # but should work for any value of decmin from -90 to +90)
    if decmin == -90:  # no restrictions
        return -numpy.pi/2, numpy.pi/2, lambda b: b*0, numpy.pi

    # 1. obtain the l,b coords of the two poles (south and north) in ICRS
    lp, bp = agama.transformCelestialCoords(agama.fromICRStoGalactic, [0,0], [-numpy.pi/2, numpy.pi/2])
    # if decmin < decp[0], there is only one loop around the south ICRS pole,
    # likewise if decmin > decp[1], there is only one loop around the north ICRS pole,
    # otherwise there is a boundary spanning the entire range of l

    # 2. find the latitude at which the boundary crosses the meridional lines in l,b containing the poles
    def rootfinder(b, l0):
        return agama.transformCelestialCoords(agama.fromGalactictoICRS, l0, b)[1] - decmin*d2r
    if decmin*d2r < bp[0]:
        b1 = scipy.optimize.brentq(rootfinder, bp[0], -numpy.pi/2, args=(lp[0],))
    else:
        b1 = scipy.optimize.brentq(rootfinder, bp[1], -numpy.pi/2, args=(lp[1],))
    if decmin*d2r > bp[1]:
        b2 = scipy.optimize.brentq(rootfinder, bp[1], +numpy.pi/2, args=(lp[1],))
    else:
        b2 = scipy.optimize.brentq(rootfinder, bp[0], +numpy.pi/2, args=(lp[0],))

    # 3. construct a boundary - minimum l for each b between b1 and b2
    npoints = 201
    bb = numpy.linspace(0, 1, npoints)
    bb = bb*bb*(3-2*bb) * (b2-b1) + b1
    ll = numpy.zeros(npoints)
    ll[0] = lp[0] if decmin*d2r < bp[0] else lp[1]
    ll[-1]= lp[1] if decmin*d2r > bp[1] else lp[0]
    for i in range(1,npoints-1):
        ll[i] = scipy.optimize.brentq(
            lambda l: agama.transformCelestialCoords(agama.fromGalactictoICRS, l, bb[i])[1] - decmin*d2r,
            lp[0], lp[1])
    curve = agama.CubicSpline(bb, ll)

    # 4. return a tuple of four elements:
    #   lower and upper limit on b,
    #   a function evaluating lmin for the given b,
    #   and lsym such that lmax(b) = 2*lsym - lmin(b).
    # note that all coords here are in radians, and the range of l is from lsym-pi to lsym+pi (or smaller),
    # not from 0 to 2pi or -pi to pi; thus the selection boundary doesn't enclose the actual coordinates
    # of points unless these are shifted to the same angular range, but this doesn't matter for the purpose
    # of computing the normalization factor (integral of density over the selection region)
    if decmin*d2r < bp[0]:
        # excluded a region inside a closed loop aroung South pole; b can be anywhere between -pi/2 and pi/2
        return -numpy.pi/2, numpy.pi/2, lambda b: curve(b, ext=lp[0]), lp[1]
    elif decmin*d2r <= bp[1]:
        # excluded a region below a curve spanning the entire range of l; b must be >= b1
        return b1, numpy.pi/2, lambda b: curve(b, ext=lp[0]), lp[1]
    else:  # excluded a region outside a closed loop around North pole; b must be between b1 and b2
        return b1, b2, lambda b: curve(b, ext=lp[1]), lp[1]


gaiaRelease = 'dr3'
datasetType = 'latte'  # 'latte' or 'agama'

if datasetType == 'agama':
    agama.setUnits(length=1, mass=1, velocity=1)
    pot = agama.Potential(type='nfw', scaleradius=18, mass=1e12)    # a typical Milky Way-sized NFW halo
    den = agama.Density(type='spheroid', gamma=1, beta=5, scaleradius=20)   # some fiducial stellar halo profile
    l, b, radii, Gapp, pml, pmb, vlos, PMerr, vloserr, true_sigmar, true_sigmat = loadMock(datasetType=datasetType, gaiaRelease=gaiaRelease, density=den, potential=pot, beta0=-0.5, r_a=60.0, nbody=30000)
    figs_path = f"agama_{gaiaRelease}_figs/"
elif datasetType == 'latte':
    lattesim = "m12f"
    l, b, radii, Gapp, pml, pmb, vlos, PMerr, vloserr, true_sigmar, true_sigmat = loadMock(datasetType=datasetType, lattesim=lattesim, gaiaRelease=gaiaRelease)
    figs_path = f"latte_{lattesim}_{gaiaRelease}_figs/"
else:
    print("datasetType not understood")
    exit(1)

print('%i stars in the survey volume' % len(l))
blow, bupp, lmin, lsym = getSurveyFootprintBoundary(decmin)

if not os.path.exists(figs_path):
    os.makedirs(figs_path)
    print("created output directory for figures at " + figs_path)

# diagnostic plot showing the stars in l,b and the selection region boundary
if True:
    plt.scatter(l, b, s=2, c=Gapp, cmap='hell', vmin=Gmin, vmax=Gmax+1, edgecolors='none')
    if blow <= -bmin*d2r:  # selection region in the southern Galactic hemisphere
        bb=numpy.linspace(blow, -bmin*d2r, 100)
        l1=lmin(bb)
        l2=2*lsym-l1
        plt.plot(numpy.hstack((l1, l2[::-1], l1[0])) / d2r, numpy.hstack((bb, bb[::-1], bb[0])) / d2r, 'g')
    if bupp >= bmin*d2r:  # selection region in the northern Galactic hemisphere
        bb=numpy.linspace(bupp, bmin*d2r, 100)
        l1=lmin(bb)
        l2=2*lsym-l1
        plt.plot(numpy.hstack((l1, l2[::-1], l1[0])) / d2r, numpy.hstack((bb, bb[::-1], bb[0])) / d2r, 'g')
    plt.xlabel('l')
    plt.ylabel('b')
    plt.savefig(figs_path+"sel_bounds.png", dpi=250)
    plt.cla()
    plt.show()

#fitDensityProfile(l, b, Gapp, pml, pmb, vlos, PMerr, vloserr)
#def fitDensityProfile(l, b, Gapp, pml, pmb, vlos, PMerr, vloserr):

# this used to be a function, but the MCMC parallelization doesn't work unless the likelihood fnc
# is in the global scope (possibly fixed in the latest EMCEE version - haven't checked)
if True:
    npoints = len(l)

    # convert l,b,dist.mod. of all stars into logarithm of Galactocentric radius (observed, not true)
    # unit conversion: degrees to radians for l,b,  mas/yr to km/s/kpc for PM
    dist_obs = 10**(0.2*(Gapp-Grrl)-2)
    x,y,z,vx,vy,vz = agama.getGalactocentricFromGalactic(
        l * d2r, b * d2r, dist_obs, pml * 4.74, pmb * 4.74, vlos)    # coord rotation here
    logr_obs = 0.5 * numpy.log(x**2 + y**2 + z**2)
    vr_obs = (x*vx+y*vy+z*vz) / numpy.exp(logr_obs)
    vt_obs = (0.5 * (vx**2+vy**2+vz**2 - vr_obs**2))**0.5

    # create random samples from the distance modulus uncertainty for each star and convert to Galactocentric r
    nsamples = 20  # number of random samples per star
    Gsamp = (numpy.random.normal(size=(npoints, nsamples)) * DMerr + Gapp[:,None]).reshape(-1)
    dist_samp = 10**(0.2*(Gsamp-Grrl)-2)
    x,y,z = agama.getGalactocentricFromGalactic(
        numpy.repeat(l * d2r, nsamples), numpy.repeat(b * d2r, nsamples), dist_samp)
    R = (x**2 + y**2)**0.5
    r = (x**2 + y**2 + z**2)**0.5  # array of samples for Galactocentric radius
    logr_samp = numpy.log(r)

    # a rather clumsy way of constructing the matrices describing how the intrinsic 3d velocity dispersions
    # are translated to the Vlos and PM dispersions at each data sample:
    # first compute the expected mean values (pml, pmb, vlos) for a star at rest at a given distance,
    # then repeat the exercise 3 times, setting one of velocity components (v_r, v_theta, v_phi)
    # to 1 km/s, and subtract from the zero-velocity projection.
    vel0 = numpy.array(agama.getGalacticFromGalactocentric(x, y, z, x*0, y*0, z*0)[3:6])
    velr = numpy.array(agama.getGalacticFromGalactocentric(x, y, z, x/r, y/r, z/r)[3:6]) - vel0
    velt = numpy.array(agama.getGalacticFromGalactocentric(x, y, z, z/r*x/R, z/r*y/R, -R/r)[3:6]) - vel0
    velp = numpy.array(agama.getGalacticFromGalactocentric(x, y, z, -y/R, x/R, 0*r)[3:6]) - vel0

    # matrix of shape (2, npoints*nsamples) describing how the two intrinsic velocity dispersions
    # in 3d Galactocentric coords translate to the line-of-sight velocity dispersion at each sample point
    mat_vlos = numpy.array([ velr[2]**2, velt[2]**2 + velp[2]**2 ])
    # same for the PM dispersions: this is a 2x2 symmetric matrix for each datapoint,
    # characterized by two diagonal and one off-diagonal elements,
    # and each element, in turn, is computed from the two Galactocentric intrinsic velocity dispersions
    mat_pm   = numpy.array([
        [velr[0]*velr[0], velt[0]*velt[0] + velp[0]*velp[0]],
        [velr[1]*velr[1], velt[1]*velt[1] + velp[1]*velp[1]],
        [velr[0]*velr[1], velt[0]*velt[1] + velp[0]*velp[1]] ]) / 4.74**2

    # difference between the measured PM and Vlos values and the expected mean values at each data sample
    # (the latter correspond to a zero 3d velocity, translated to the Heliocentric frame)
    pml_samp  = numpy.repeat(pml,  nsamples) - vel0[0] / 4.74
    pmb_samp  = numpy.repeat(pmb,  nsamples) - vel0[1] / 4.74
    vlos_samp = numpy.repeat(vlos, nsamples) - vel0[2]
    # vectors of PM and Vlos errors for each data sample, to be added to the model covariance matrices
    pmlerr2_samp  = numpy.repeat(PMerr, nsamples)**2
    pmberr2_samp  = numpy.repeat(PMerr, nsamples)**2   # here is identical to pml, but in general may be different
    vloserr2_samp = numpy.repeat(vloserr, nsamples)**2

    # knots in Galactocentric radius (minimum is imposed by our cut |b|>30, maximum - by the extent of data)
    knots_logr = numpy.linspace(numpy.log(5.0), numpy.log(80.0), 6)

    def modelDensity(params):
        # params is the array of logarithms of density at radial knots, which must monotonically decrease
        # note that since the result is invariant w.r.t. the overall amplitude of the density
        # (it is always renormalized to unity), the first element of this array may be fixed to 0
        knots_logdens = numpy.hstack((0, params))
        if any(knots_logdens[1:] >= knots_logdens[:-1]):
            raise RuntimeError('Density is non-monotonic')
        # represent the spherically symmetric 1d profile log(rho) as a cubic spline in log(r)
        logrho = agama.CubicSpline(knots_logr, knots_logdens, reg=True)  # ensure monotonic spline (reg)
        # check that the density profile has a finite total mass
        # (this is not needed for the fit, because the normalization is computed over the accessible volume,
        # but it generally makes sense to have a physically valid model for the entire space).
        slope_in  = logrho(knots_logr[ 0], der=1)  # d[log(rho)]/d[log(r)], log-slope at the lower radius
        slope_out = logrho(knots_logr[-1], der=1)
        if slope_in <= -3.0 or slope_out >= -3.0:
            raise RuntimeError('Density has invalid asymptotic slope: inner=%.2f, outer=%.2f' % (slope_in, slope_out))

        # now the difficult part: normalize the 3d density over the volume of the survey,
        # taking into account fuzzy outer boundary in distance due to obs.errors in abs.magnitude.
        # integration is performed in scaled l, sin(b), dm=distance modulus  (l,b expressed in radians)
        # over the region lmin(b)<=l<=lmax(b), bmin<=b<=bmax, Gmin-4*DMerr <= dm+Grrl <= Gmax+4*DMerr.
        def integrand(coords):
            # return the density times selection function times jacobian of coord transformation
            scaledl, sinb, dm = coords.T
            b = numpy.arcsin(sinb)
            # unscale the coordinates inside the curved selection region: 0<=scaledl<=1 => lmin(b)<=l<=lmax(b)
            lminb = lmin(b)
            lmaxb = 2*lsym - lminb
            l     = lminb + (lmaxb-lminb) * scaledl
            dist  = 10**(0.2*dm-2)
            x,y,z = agama.getGalactocentricFromGalactic(l, b, dist)  # coord rotation here only positions
            logr  = numpy.log(x**2 + y**2 + z**2) * 0.5
            jac   = dist**3 * numpy.log(10)/5 * (lmaxb-lminb)
            # multiplicative factor <= 1 accounting for a gradual decline in selection probability
            # as stars become fainter that the limiting magnitude of the survey
            if DMerr==0:
                mult = ((dm <= Gmax-Grrl) * (dm >= Gmin-Grrl)).astype(float)
            else: mult = 0.5 * (
                scipy.special.erf( (Gmax-Grrl-dm) / 2**0.5 / DMerr ) -
                scipy.special.erf( (Gmin-Grrl-dm) / 2**0.5 / DMerr ) )
            return numpy.exp(logrho(logr)) * mult * jac

        # first compute the integral over the selection region in the northern Galactic hemisphere
        norm = agama.integrateNdim(integrand,
            lower=[0, numpy.sin(max(bmin*d2r, blow)), Gmin-4*DMerr],
            upper=[1, numpy.sin(bupp),                Gmax+4*DMerr], toler=1e-5)[0]
        # then add the contribution from the region in the southern Galactic hemisphere
        if blow <= -bmin*d2r:
            norm += agama.integrateNdim(integrand,
            lower=[0, numpy.sin(blow),      Gmin-4*DMerr],
            upper=[1, numpy.sin(-bmin*d2r), Gmax+4*DMerr], toler=1e-5)[0]

        # now renormalize the density profile to have unit integral over the selection volume
        logrho = agama.CubicSpline(knots_logr, knots_logdens - numpy.log(norm), reg=True)
        return logrho

    def modelSigma(params):
        # params is the array of log(sigma(r)) at radial knots (applicable to both velocity components)
        return agama.CubicSpline(knots_logr, params)

    def likelihood(params):
        # function to be maximized in the MCMC and deterministic optimization
        params_dens   = params[0 : len(knots_logr)-1]
        params_sigmar = params[len(knots_logr)-1 : 2*len(knots_logr)-1]
        params_sigmat = params[2*len(knots_logr)-1 :]
        try:
            # compute the predicted density from the model at each data sample
            logrho    = modelDensity(params_dens)(logr_samp)
            # likelihood of finding each data sample in the given density profile (multiplied by r^3)
            like_dens = numpy.exp(logrho + 3*logr_samp)
            # construct intrinsic velocity dispersion profiles of the model
            sigmar2   = numpy.exp(2*modelSigma(params_sigmar)(logr_samp))  # squared radial velocity dispersion
            sigmat2   = numpy.exp(2*modelSigma(params_sigmat)(logr_samp))  # squared tangential --"--
            sigmaboth = numpy.vstack((sigmar2, sigmat2))  # shape: (2, nbody*nsamples)
            # convert these profiles to the values of line-of-sight velocity dispersion at each data sample
            cov_vlos  = numpy.einsum('kp,kp->p', mat_vlos, sigmaboth)
            # same for the PM dispersion profiles - diagonal and off-diagonal elements of the PM covariance matrix
            cov_pmll, cov_pmbb, cov_pmlb = numpy.einsum('ikp,kp->ip', mat_pm, sigmaboth)
            # add individual observational errors for each data sample
            cov_vlos += vloserr2_samp
            cov_pmll += pmlerr2_samp  # here add to diagonal elements of PM cov matrix only,
            cov_pmbb += pmberr2_samp  # but with the actual Gaia data should also use the off-diagonal term
            det_pm    = cov_pmll * cov_pmbb - cov_pmlb**2  # determinant of the PM cov matrix
            # compute likelihoods of Vlos and PM values of each data sample, taking into account obs.errors
            like_vlos = cov_vlos**-0.5 * numpy.exp(-0.5 * vlos_samp**2 / cov_vlos)
            like_pm   = det_pm**-0.5   * numpy.exp(-0.5 / det_pm *
                (pml_samp**2 * cov_pmbb + pmb_samp**2 * cov_pmll - 2 * pml_samp * pmb_samp * cov_pmlb) )
            # the overall log-likelihood of the model:
            # first average the likelihoods of all sample points for each star -
            # this corresponds to marginalization over distance uncertainties, also propagated to PM space;
            # then sum up marginalized log-likelihoods of all stars.
            # at this stage may also add a prior if necessary
            loglikelihood = numpy.sum(numpy.log(
                numpy.mean((like_dens * like_pm * like_vlos).reshape(npoints, nsamples), axis=1)))
            #print("%s => %.2f" % (params, loglikelihood))
            if not numpy.isfinite(loglikelihood): loglikelihood = -numpy.inf
            return loglikelihood
        except Exception as ex:
            print("%s => %s" % (params, ex))
            return -numpy.inf

    def plotprofiles(chain, plotname="a"):
        ax = plt.subplots(1, 2, figsize=(10,5))[1]
        # left panel: density profiles
        r  = numpy.logspace(0, 2, 41)
        lr = numpy.log(r)
        if datasetType == 'agama':
            trueparams_dens = numpy.log(den.density(numpy.column_stack((numpy.exp(knots_logr), knots_logr*0, knots_logr*0))))
        if datasetType == 'latte':
            S = agama.splineLogDensity(knots_logr, x=numpy.log(radii), w=numpy.ones(len(radii)))
            trueparams_dens = numpy.log((numpy.exp(S(knots_logr))) / (4.0 * numpy.pi * (numpy.exp(knots_logr)**3)))
        trueparams_dens = trueparams_dens[1:] - trueparams_dens[0]  # set the first element of array to zero and exclude it
        truedens = numpy.exp(modelDensity(trueparams_dens)(lr))
        ax[0].plot(r, truedens, 'k--', label='true density')
        # retrieve density profiles of each model in the chain, and compute median and 16/84 percentiles
        results = numpy.zeros((len(chain), len(r)))
        for i in range(len(chain)):
            results[i] = numpy.exp(modelDensity(chain[i, 0:len(knots_logr)-1])(lr))
        low, med, upp = numpy.percentile(results, [16,50,84], axis=0)
        # plot the model profiles with 1sigma confidence intervals
        ax[0].fill_between(r, low, upp, alpha=0.3, lw=0, color='r')
        ax[0].plot(r, med, color='r', label='fit density')
        # construct the histogram of observed galactocentric distances of all stars in the sample
        # (blurred by distance errors and affected by spatial selection function)
        # dN/d(ln r) = 4pi r^3 rho
        count_obs = numpy.histogram(logr_obs, bins=lr)[0]
        rho_obs = count_obs / (4 * numpy.pi * (lr[1:]-lr[:-1]) * (r[1:]*r[:-1])**1.5 * len(Gapp))
        ax[0].plot(numpy.repeat(r,2)[1:-1], numpy.repeat(rho_obs, 2), 'b', label='actual dataset')
        ax[0].set_xlabel('Galactocentric radius')
        ax[0].set_ylabel('3d density of tracers')
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[0].set_xlim(min(r), max(r))
        ax[0].set_ylim(min(truedens)*0.2, max(truedens)*2)
        ax[0].legend(loc='upper right', frameon=False)

        # right panel: velocity dispersion profiles
        ax[1].plot(r, numpy.exp(true_sigmar(lr)), 'r--', label='true $\sigma_\mathrm{rad}$')
        ax[1].plot(r, numpy.exp(true_sigmat(lr)), 'b--', label='true $\sigma_\mathrm{tan}$')
        # again collect the model profiles and plot median and 16/84 percentile confidence intervals
        results_r, results_t = numpy.zeros((2, len(chain), len(r)))
        for i in range(len(chain)):
            results_r[i] = numpy.exp(modelSigma(chain[i, len(knots_logr)-1 : 2*len(knots_logr)-1])(lr))
            results_t[i] = numpy.exp(modelSigma(chain[i, 2*len(knots_logr)-1 :])(lr))
        low, med, upp = numpy.percentile(results_r, [16,50,84], axis=0)
        ax[1].fill_between(r, low, upp, alpha=0.3, lw=0, color='m')
        ax[1].plot(r, med, color='m', label='fit $\sigma_\mathrm{rad}$')
        low, med, upp = numpy.percentile(results_t, [16,50,84], axis=0)
        ax[1].fill_between(r, low, upp, alpha=0.3, lw=0, color='c')
        ax[1].plot(r, med, color='c', label='fit $\sigma_\mathrm{tan}$')
        # and plot the observed radial/tangential dispersions, which are affected by distance errors
        # and broadened by PM errors (especially the tangential dispersion)
        sigmar_obs = (numpy.histogram(logr_obs, bins=lr, weights=vr_obs**2)[0] / count_obs)**0.5
        sigmat_obs = (numpy.histogram(logr_obs, bins=lr, weights=vt_obs**2)[0] / count_obs)**0.5
        ax[1].plot(numpy.repeat(r,2)[1:-1], numpy.repeat(sigmar_obs, 2), 'm', label='actual dataset $\sigma_\mathrm{rad}$')
        ax[1].plot(numpy.repeat(r,2)[1:-1], numpy.repeat(sigmat_obs, 2), 'c', label='actual dataset $\sigma_\mathrm{tan}$')
        ax[1].set_xlabel('Galactocentric radius')
        ax[1].set_ylabel('velocity dispersion of tracers')
        ax[1].set_xscale('log')
        ax[1].legend(loc='upper left', frameon=False)

        plt.tight_layout()
        plt.savefig(figs_path+plotname+".png", dpi=250)
        plt.show()

    # initial values of parameters
    params_dens    = -numpy.linspace(1, 3, len(knots_logr)-1)**2  # log of (un-normalized) density values at radial knots
    params_sigmar  = numpy.zeros(len(knots_logr)) + 5.0   # log of radial velocity dispersion values at the radial knots
    params_sigmat  = numpy.zeros(len(knots_logr)) + 5.0   # same for tangential dispersion
    params         = numpy.hstack((params_dens, params_sigmar, params_sigmat))
    paramnames     = [ 'logrho(r=%4.1f)' % r for r in numpy.exp(knots_logr[1:]) ] + \
        [ 'sigmar(r=%4.1f)' % r for r in numpy.exp(knots_logr) ] + \
        [ 'sigmat(r=%4.1f)' % r for r in numpy.exp(knots_logr) ]
    prevmaxloglike = -numpy.inf
    prevavgloglike = -numpy.inf
    # first find the best-fit model by deterministic optimization algorithm,
    # restarting it several times until it seems to arrive at the global minimum
    while True:
        print('Starting deterministic search')
        # minimization algorithm - so provide a negative likelihood to it
        params = scipy.optimize.minimize(lambda x: -likelihood(x), params, method='Nelder-Mead',
            options=dict(maxfev=500)).x
        maxloglike = likelihood(params)
        if maxloglike - prevmaxloglike < 1.0:
            for i in range(len(params)):
                print('%s = %8.4g' % (paramnames[i], params[i]))
            print('Converged')
            break
        else:
            print('Improved log-likelihood by %f' % (maxloglike - prevmaxloglike))
        prevmaxloglike = maxloglike

    # show profiles and wait for the user to marvel at them
    plotprofiles(params.reshape(1,-1), "preMCMC")
    
    # then start a MCMC around the best-fit params
    paramdisp= numpy.ones(len(params))*0.1  # spread of initial walkers around best-fit params
    nwalkers = 2*len(params)   # minimum possible number of walkers in emcee
    nsteps   = 500
    walkers  = numpy.empty((nwalkers, len(params)))
    numtries = 0
    for i in range(nwalkers):
        while numtries<10000:   # ensure that we initialize walkers with feasible values
            walker = params + numpy.random.randn(len(params))*paramdisp
            if numpy.isfinite(likelihood(walker)):
                walkers[i] = walker
                break
            numtries+=1
    if numtries>=10000:
        raise RuntimeError('cannot initialize MCMC')
    with Pool() as pool:
        # numthreads = nwalkers//2   # parallel threads in emcee - make sure you don't clog up your machine!
        start = time.time()
        sampler  = emcee.EnsembleSampler(nwalkers, len(params), likelihood, pool=pool)
        print('Starting MCMC search')
        converged = False
        iter = 0
        while not converged:  # run several passes until log-likelihood stabilizes (convergence is reached)
            sampler.run_mcmc(walkers, nsteps, progress=True)
            walkers = sampler.chain[:,-1]
            chain   = sampler.chain[:,-nsteps:].reshape(-1, len(params))
            maxloglike = numpy.max (sampler.lnprobability[:,-nsteps:])
            avgloglike = numpy.mean(sampler.lnprobability[:,-nsteps:])
            walkll = sampler.lnprobability[:,-1]
            for i in range(len(params)):
                print('%s = %8.4g +- %7.4g' % (paramnames[i], numpy.mean(chain[:,i]), numpy.std(chain[:,i])))
            print('max loglikelihood: %.2f, average: %.2f' % (maxloglike, avgloglike))
            converged = abs(maxloglike-prevmaxloglike) < 1.0 and abs(avgloglike-prevavgloglike) < 2.0
            prevmaxloglike = maxloglike
            prevavgloglike = avgloglike
            if converged: print('Converged'); plotprofiles(chain[::20], "converged")

            # produce diagnostic plots after each MCMC episode:
            # 1. evolution of parameters along the chain for each walker
            axes = plt.subplots(len(params)+1, 1, sharex=True, figsize=(10,10))[1]
            for i in range(len(params)):
                axes[i].plot(sampler.chain[:,:,i].T, color='k', alpha=0.5)
                axes[i].set_xticklabels([])
                axes[i].set_ylabel(paramnames[i])
            axes[-1].plot(sampler.lnprobability.T, color='k', alpha=0.5)
            axes[-1].set_ylabel('likelihood')   # bottom panel is the evolution of likelihood
            axes[-1].set_ylim(maxloglike-3*len(params), maxloglike)
            plt.tight_layout(h_pad=0)
            plt.subplots_adjust(hspace=0,wspace=0)
            plt.savefig(figs_path+"param_evol_iter"+str(iter)+".png", dpi=350)
            plt.show()
            # 2. corner plot - covariances of all parameters
            corner.corner(chain, quantiles=[0.16, 0.5, 0.84], labels=paramnames, show_titles=True)
            plt.savefig(figs_path+"corner_iter"+str(iter)+".png", dpi=350)
            plt.show()
            # 3. density and velocity dispersion profiles - same as before
            if not converged: plotprofiles(chain[::20], "MCMC_iter"+str(iter))
            iter += 1
    print("MCMC run time:", time.time()-start)
    plt.show()
