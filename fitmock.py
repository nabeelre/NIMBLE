import os, sys, numpy, scipy.special, scipy.optimize, emcee, corner, agama, time, matplotlib.pyplot as plt
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
# test on indu.astro
# code true profiles for agama

# Notes:
# Check to see what total_proper_motion_uncertainty is ... sum or average or something else?
    # total_proper_motion_uncertainty is a simple average of the PM RAcosdec and PM Dec uncertainties
    # So, no need to divide its output by two when using it as PMerr: a single unc for either direction
# emcee parallelization will probably only work with newer emcee versions
# combine makeMock & readMock to reduce repeated code
# DESI lower mag limit is 16? https://arxiv.org/pdf/2010.11284.pdf - says 16 in r mag
# true sigma and density profiles for Latte datasets -> combine both tangential dispersions
# LSR information for latte datasets and coordinate rotations where necessary
# improve plots
# write jeans into this script
# doing true profiles wrong? 
    # I'm using only the metal poor star particles to make them, should i use all star particles


def rotate_x(x_old, y_old, theta):
    return x_old*numpy.cos(theta) + y_old*numpy.sin(theta)


def rotate_y(x_old, y_old, theta):
    return -1*x_old*numpy.sin(theta) + y_old*numpy.cos(theta)


def get_lsr_cartesian(sim, lsr):
    if lsr == "LSR0":
        x_sun = 0.000000
        y_sun = 8.200000
        z_sun = 0.000000
        if sim == "m12f":
            vx_sun = 226.184921
            vy_sun = 14.377288
            vz_sun = -4.890565
        elif sim == "m12i":
            vx_sun = 224.709198
            vy_sun = -20.380102
            vz_sun = 3.895417
        elif sim == "m12m":
            vx_sun = 254.918686
            vy_sun = 16.790098
            vz_sun = 1.964817
        else:
            print("Could not find simulation")
            exit()
    elif lsr == "LSR1":
        x_sun = -7.101408
        y_sun = -4.100000
        z_sun = 0.000000
        if sim == "m12f":
            vx_sun = -114.035072
            vy_sun = 208.726669
            vz_sun = 5.063526
        elif sim == "m12i":
            vx_sun = -80.426880
            vy_sun = 191.723969
            vz_sun = 1.503948
        elif sim == "m12m":
            vx_sun = -128.247955
            vy_sun = 221.148926
            vz_sun = 5.850575
        else:
            print("Could not find simulation")
            exit()
    elif lsr == "LSR2":
        x_sun = 7.101408
        y_sun = -4.100000
        z_sun = 0.000000
        if sim == "m12f":
            vx_sun = -118.143044
            vy_sun = -187.763062
            vz_sun = -3.890517
        elif sim == "m12i":
            vx_sun = -87.273514
            vy_sun = -186.856659
            vz_sun = -9.460751
        elif sim == "m12m":
            vx_sun = -106.620308
            vy_sun = -232.205551
            vz_sun = -6.418519
        else:
            print("Could not find simulation")
            exit()
    else:
        print("Couldn't find lsr")
        exit()
    return x_sun, y_sun, z_sun, vx_sun, vy_sun, vz_sun


def rotate_coords(sim, lsr, positions, velocities):
    # positions and velocities are an Nx3 matricies with columns x, y, z and vx, vy, vz respectively
    # all coordinates here are galactocentric cartesian
    x_sun_orig, y_sun_orig, z_sun_orig, \
        vx_sun_orig, vy_sun_orig , vz_sun_orig = get_lsr_cartesian(sim, lsr)

    theta  = numpy.arctan2(y_sun_orig, x_sun_orig) + numpy.pi  # angle in radians to rotate to -x axis
    x_sun  = rotate_x(x_sun_orig, y_sun_orig, theta)    # kpc
    y_sun  = rotate_y(x_sun_orig, y_sun_orig, theta)    # kpc
    vx_sun = rotate_x(vx_sun_orig, vy_sun_orig, theta)  # km/s
    vy_sun = rotate_y(vx_sun_orig, vy_sun_orig, theta)  # km/s

    galcen_v_sun_sim    = (vx_sun, vy_sun, vz_sun_orig)
    galcen_distance_sim = numpy.sqrt(x_sun**2 + y_sun**2)

    x_new  = rotate_x(positions[:,0], positions[:,1], theta)
    y_new  = rotate_y(positions[:,0], positions[:,1], theta)
    vx_new = rotate_x(velocities[:,0], velocities[:,1], theta)
    vy_new = rotate_y(velocities[:,0], velocities[:,1], theta)

    return (galcen_distance_sim, galcen_v_sun_sim, z_sun_orig), \
        x_new, y_new, vx_new, vy_new


def loadMock(datasetType, gaiaRelease, density=None, potential=None, beta0=None, r_a=None, nbody=None, lattesim=None, lsr=None):
    rr = numpy.logspace(0, 2, 15)
    xyz= numpy.column_stack((rr, rr*0, rr*0))
    if datasetType == 'agama':
         # create a spherical anisotropic DF and compute its true velocity dispersions
        df = agama.DistributionFunction(type='quasispherical',
            density=density, potential=potential, beta0=beta0, r_a=r_a)
        gm = agama.GalaxyModel(potential, df)

        # sample 6d points from the model (over the entire space)
        xv = gm.sample(nbody)[0]
        radii = numpy.sqrt(xv[:,0]**2 + xv[:,1]**2 + xv[:,2]**2)

        sig= gm.moments(xyz, dens=False, vel=False, vel2=True)
        # represent sigma profiles as cubic splines for log(sigma) as a function of log(r)
        true_sigmar = agama.CubicSpline(numpy.log(rr), numpy.log(sig[:,0]**0.5))
        true_sigmat = agama.CubicSpline(numpy.log(rr), numpy.log(sig[:,1]**0.5))

        # agama default values
        lsr_info = (8.122, (12.9, 245.6, 7.78), 0.0208)
    elif datasetType == 'latte':
        # dataset for m12f available at https://drive.google.com/file/d/1Z8lQEdPeX1995WDJsc1qxeh07ZDLBZXr/view?usp=sharing
        # formatted such that rows represent particles and columns represent different quantites
        # col 0-5: cartesian positions and velocities, col 6: particle mass, col 7: galactocentric spherical radius
        # col 8-10: square of spherical velocity components - r (radial), theta (polar), phi (azimuthal)
        x, y, z, vx, vy, vz, \
            mass, radii, rvelsq, tvelsq, pvelsq = numpy.loadtxt(f"latte/{lattesim}/{lattesim}_chem-1.5_full.csv",
                                                                unpack=True, skiprows=1, delimiter=',')
        lsr_info, x_new, y_new, vx_new, vy_new = rotate_coords(lattesim, lsr,
                                                               numpy.column_stack((x, y, z)),
                                                               numpy.column_stack((vx, vy, vz)))

        xv = numpy.column_stack((x_new, y_new, z, vx_new, vy_new, vz))
        nbody = len(x_new)

        true_sigmar = agama.splineApprox(numpy.log(rr), numpy.log(radii), rvelsq)
        true_sigmat = agama.splineApprox(numpy.log(rr), numpy.log(radii), (tvelsq + pvelsq)/2)

    l, b, dist, pml, pmb, vlos = agama.getGalacticFromGalactocentric(*xv.T, *lsr_info)
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
    l, b, pml, pmb = agama.transformCelestialCoords(agama.fromICRStoGalactic,
                                                    ra, dec, pmra, pmdec)
    l /= d2r
    b /= d2r

    # add Vlos errors
    vloserr = numpy.ones(nbody) * (10.0 if LARGEVLOS else 2.0)
    vlos += numpy.random.normal(size=nbody) * vloserr

    return (l[filt], b[filt], radii, Gapp[filt], pml[filt], pmb[filt],
            vlos[filt], PMerr[filt], vloserr[filt], true_sigmar, true_sigmat,
            lsr_info)


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

LARGEVLOS = False
if len(sys.argv) < 3 or len(sys.argv) > 6:
    print("Too few or too many command line arguments")
    exit()
else:
    datasetType = sys.argv[1]
    assert(datasetType in ['latte', 'agama'])

    if datasetType == 'agama':
        assert(len(sys.argv) == 3 or len(sys.argv) == 4)
        gaiaRelease = sys.argv[2]
        lattesim = None
        lsr = None
        if len(sys.argv) == 4:
            if sys.argv[3] == 'largevlos':
                LARGEVLOS = True
        print(f"RUNNING AGAMA {gaiaRelease} {'with large Vlos errors' if LARGEVLOS else ''}")
    elif datasetType == 'latte':
        assert(len(sys.argv) == 5 or len(sys.argv) == 6)
        lattesim = sys.argv[2].lower()
        lsr = sys.argv[3].upper()
        gaiaRelease = sys.argv[4]
        if len(sys.argv) == 6:
            if sys.argv[3] == 'largevlos':
                LARGEVLOS = True
        print(f"RUNNING LATTE {lattesim} {lsr} {gaiaRelease} {'with large Vlos errors' if LARGEVLOS else ''}")
    else:
        print("Unrecognized datasetType")
        exit()

    assert(gaiaRelease in ['dr3', 'dr4', 'dr5'])
    assert(lattesim in ['m12f', 'm12i', 'm12m', None])
    assert(lsr in ['LSR0', 'LSR1', 'LSR2', None])
        
if datasetType == 'agama':
    agama.setUnits(length=1, mass=1, velocity=1)
    pot = agama.Potential(type='nfw', scaleradius=18, mass=1e12)    # a typical Milky Way-sized NFW halo
    den = agama.Density(type='spheroid', gamma=1, beta=5, scaleradius=20)   # some fiducial stellar halo profile
    l, b, radii, Gapp, pml, pmb, vlos, PMerr, vloserr, true_sigmar, true_sigmat, \
        lsr_info = loadMock(datasetType, gaiaRelease, density=den, potential=pot,
                            beta0=-0.5, r_a=60.0, nbody=30000)
    figs_path = f"agama_{gaiaRelease}_figs/"
elif datasetType == 'latte':
    l, b, radii, Gapp, pml, pmb, vlos, PMerr, vloserr, true_sigmar, true_sigmat, \
        lsr_info = loadMock(datasetType, lattesim=lattesim, gaiaRelease=gaiaRelease, lsr=lsr)
    figs_path = f"latte_{lattesim}_{lsr}_{gaiaRelease}_figs/"

print('%i stars in the survey volume' % len(l))
blow, bupp, lmin, lsym = getSurveyFootprintBoundary(decmin)

# vlos temp
if LARGEVLOS:
    figs_path += "vlos/"

if not os.path.exists(figs_path):
    os.makedirs(figs_path)
    print("created output directory for figures at " + figs_path)

# diagnostic plot showing the stars in l,b and the selection region boundary
if True:
    plt.scatter(l, b, s=2, c=Gapp, cmap='hell', vmin=Gmin, vmax=Gmax+1, edgecolors='none')
    plt.colorbar(label='Gapp')
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
    plt.title(figs_path)
    plt.xlabel('galactic longitude l (degrees)')
    plt.ylabel('galactic latitude b (degrees)')
    plt.tight_layout()
    plt.savefig(figs_path+"sel_bounds.png", dpi=250)
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
        l*d2r, b*d2r, dist_obs, pml*4.74, pmb*4.74, vlos, *lsr_info)
    logr_obs = 0.5 * numpy.log(x**2 + y**2 + z**2)
    vr_obs = (x*vx+y*vy+z*vz) / numpy.exp(logr_obs)
    vt_obs = (0.5 * (vx**2+vy**2+vz**2 - vr_obs**2))**0.5

    # create random samples from the distance modulus uncertainty for each star and convert to Galactocentric r
    nsamples = 20  # number of random samples per star
    Gsamp = (numpy.random.normal(size=(npoints, nsamples)) * DMerr + Gapp[:,None]).reshape(-1)
    dist_samp = 10**(0.2*(Gsamp-Grrl)-2)
    x,y,z = agama.getGalactocentricFromGalactic(
        numpy.repeat(l * d2r, nsamples), numpy.repeat(b * d2r, nsamples), dist_samp, galcen_distance=lsr_info[0], galcen_v_sun=lsr_info[1], z_sun=lsr_info[2])
    R = (x**2 + y**2)**0.5
    r = (x**2 + y**2 + z**2)**0.5  # array of samples for Galactocentric radius
    logr_samp = numpy.log(r)

    # a rather clumsy way of constructing the matrices describing how the intrinsic 3d velocity dispersions
    # are translated to the Vlos and PM dispersions at each data sample:
    # first compute the expected mean values (pml, pmb, vlos) for a star at rest at a given distance,
    # then repeat the exercise 3 times, setting one of velocity components (v_r, v_theta, v_phi)
    # to 1 km/s, and subtract from the zero-velocity projection.
    vel0 = numpy.array(agama.getGalacticFromGalactocentric(x, y, z, x*0, y*0, z*0, *lsr_info)[3:6])
    velr = numpy.array(agama.getGalacticFromGalactocentric(x, y, z, x/r, y/r, z/r, *lsr_info)[3:6]) - vel0
    velt = numpy.array(agama.getGalacticFromGalactocentric(x, y, z, z/r*x/R, z/r*y/R, -R/r, *lsr_info)[3:6]) - vel0
    velp = numpy.array(agama.getGalacticFromGalactocentric(x, y, z, -y/R, x/R, 0*r, *lsr_info)[3:6]) - vel0

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
    knots_logr = numpy.linspace(numpy.log(7.0), numpy.log(80.0), 6)

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
            x,y,z = agama.getGalactocentricFromGalactic(l, b, dist, galcen_distance=lsr_info[0], galcen_v_sun=lsr_info[1], z_sun=lsr_info[2])
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

    def plotprofiles(chain, plotname=''):
        # Density plots
        fig = plt.figure(figsize=(7,7))
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[3, 1])
        axs = gs.subplots(sharex=True)
        
        r  = numpy.logspace(0, 2, 41)
        lr = numpy.log(r)
        if datasetType == 'agama':
            trueparams_dens = numpy.log(den.density(numpy.column_stack((numpy.exp(knots_logr), knots_logr*0, knots_logr*0))))
        if datasetType == 'latte':
            S = agama.splineLogDensity(knots_logr, x=numpy.log(radii), w=numpy.ones(len(radii)), infLeft=True, infRight=True)
            trueparams_dens = numpy.log((numpy.exp(S(knots_logr))) / (4.0 * numpy.pi * (numpy.exp(knots_logr)**3)))
        trueparams_dens = trueparams_dens[1:] - trueparams_dens[0]  # set the first element of array to zero and exclude it
        truedens = numpy.exp(modelDensity(trueparams_dens)(lr))
        
        # main plot
        axs[0].plot(r, truedens, 'k--', label='true density')
        # retrieve density profiles of each model in the chain, and compute median and 16/84 percentiles
        results = numpy.zeros((len(chain), len(r)))
        for i in range(len(chain)):
            results[i] = numpy.exp(modelDensity(chain[i, 0:len(knots_logr)-1])(lr))
        low, med, upp = numpy.percentile(results, [16,50,84], axis=0)
        # plot the model profiles with 1sigma confidence intervals
        axs[0].fill_between(r, low, upp, alpha=0.3, lw=0, color='r')
        axs[0].plot(r, med, color='r', label='fit density')
        count_obs = numpy.histogram(logr_obs, bins=lr)[0]
        rho_obs = count_obs / (4 * numpy.pi * (lr[1:]-lr[:-1]) * (r[1:]*r[:-1])**1.5 * len(Gapp))
        axs[0].plot(numpy.repeat(r,2)[1:-1], numpy.repeat(rho_obs, 2), 'b', label='actual dataset')
        
        axs[0].set_title(figs_path)
        axs[0].set_ylabel('3d density of tracers')
        axs[0].set_yscale('log')
        axs[0].set_xlim(min(r), max(r))
        axs[0].set_ylim(min(truedens)*0.2, max(truedens)*2)
        axs[0].legend(loc='upper right', frameon=False)
        
        # percent error
        percerr = 100*((med - truedens) / truedens)
        lowerr = 100*((low - truedens) / truedens)
        upperr = 100*((upp - truedens) / truedens)
        axs[1].plot(r, percerr, 'g')
        axs[1].fill_between(r, lowerr, upperr, alpha=0.3, lw=0, color='g')
        axs[1].axhline(0, c='gray', linestyle='dashed')
        axs[1].set_ylim(-55,55)
        
        axs[1].set_xlabel('Galactocentric radius (kpc)')
        axs[1].set_ylabel('percent error (%)')
        axs[1].set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(figs_path+plotname+'_dens.jpg', dpi=300)
        plt.show()
        
        # Sigma plots
        fig = plt.figure(figsize=(12,7))
        gs = fig.add_gridspec(2,2, hspace=0, wspace=0, height_ratios=[3, 1], width_ratios=[1,1])
        axs = gs.subplots(sharex=True)
        axs[0,0].text(x=45, y=470, s='Radial', size=18)
        axs[0,1].text(x=45, y=470, s='Tangential', size=18)

        # again collect the model profiles and plot median and 16/84 percentile confidence intervals
        results_r, results_t = numpy.zeros((2, len(chain), len(r)))
        for i in range(len(chain)):
            results_r[i] = numpy.exp(modelSigma(chain[i, len(knots_logr)-1 : 2*len(knots_logr)-1])(lr))
            results_t[i] = numpy.exp(modelSigma(chain[i, 2*len(knots_logr)-1 :])(lr))
        low_r, med_r, upp_r = numpy.percentile(results_r, [16,50,84], axis=0)
        axs[0,0].fill_between(r, low_r, upp_r, alpha=0.3, lw=0, color='r')
        axs[0,0].plot(r, med_r, color='r', label='fit $\sigma_\mathrm{rad}$')
        low_t, med_t, upp_t = numpy.percentile(results_t, [16,50,84], axis=0)
        axs[0,1].fill_between(r, low_t, upp_t, alpha=0.3, lw=0, color='r')
        axs[0,1].plot(r, med_t, color='r', label='fit $\sigma_\mathrm{tan}$')
        
        if datasetType == 'agama':
            truesigr = numpy.exp(true_sigmar(lr))
            truesigt = numpy.exp(true_sigmat(lr))
        elif datasetType == 'latte':
            truesigr = true_sigmar(lr)**0.5
            truesigt = true_sigmat(lr)**0.5
        axs[0,0].plot(r, truesigr, 'k--', label='true $\sigma_\mathrm{rad}$')
        axs[0,1].plot(r, truesigt, 'k--', label='true $\sigma_\mathrm{tan}$')
        
        percerr_r = 100*((med_r - truesigr) / truesigr)
        lowerr_r = 100*((low_r - truesigr) / truesigr)
        upperr_r = 100*((upp_r - truesigr) / truesigr)
        axs[1,0].plot(r, percerr_r, c='g')
        axs[1,0].axhline(0, c='gray', linestyle='dashed')
        axs[1,0].fill_between(r, lowerr_r, upperr_r, alpha=0.3, lw=0, color='g')
        percerr_t = 100*((med_t - truesigt) / truesigt)
        lowerr_t = 100*((low_t - truesigt) / truesigt)
        upperr_t = 100*((upp_t - truesigt) / truesigt)
        axs[1,1].plot(r, percerr_t, c='g')
        axs[1,1].axhline(0, c='gray', linestyle='dashed')
        axs[1,1].fill_between(r, lowerr_t, upperr_t, alpha=0.3, lw=0, color='g')
        
        # and plot the observed radial/tangential dispersions, which are affected by distance errors
        # and broadened by PM errors (especially the tangential dispersion)
        sigmar_obs = (numpy.histogram(logr_obs, bins=lr, weights=vr_obs**2)[0] / count_obs)**0.5
        sigmat_obs = (numpy.histogram(logr_obs, bins=lr, weights=vt_obs**2)[0] / count_obs)**0.5
        axs[0,0].plot(numpy.repeat(r,2)[1:-1], numpy.repeat(sigmar_obs, 2), 'b', label='actual dataset $\sigma_\mathrm{rad}$', alpha=0.3)
        axs[0,1].plot(numpy.repeat(r,2)[1:-1], numpy.repeat(sigmat_obs, 2), 'b', label='actual dataset $\sigma_\mathrm{tan}$', alpha=0.3)

        #upper_bound = max(numpy.hstack((upp_r, upp_t, sigmar_obs, sigmat_obs, truesigr, truesigt)))*1.1
        axs[0,0].set_ylim(-20,500)
        axs[0,1].set_ylim(-20,500)
        axs[0,1].set_yticklabels([])
        axs[1,0].set_ylim(-55,55)
        axs[1,1].set_ylim(-55,55)
        axs[1,1].set_yticklabels([])
        
        axs[0,0].set_xlim(min(r), max(r))
        axs[0,0].set_title(figs_path)
        axs[1,0].set_xlabel('Galactocentric radius (kpc)')
        axs[1,1].set_xlabel('Galactocentric radius (kpc)')
        axs[0,0].set_ylabel('velocity dispersion of tracers')
        axs[1,0].set_ylabel('percent error (%)')
        axs[0,0].legend(loc='upper left', frameon=False)
        axs[0,1].legend(loc='upper left', frameon=False)
        
        plt.tight_layout()
        plt.savefig(figs_path+plotname+'_sigs.jpg', dpi=300)
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
                axes[i].plot(sampler.chain[:,:,i].T, color='k', alpha=0.3)
                axes[i].set_xticklabels([])
                axes[i].set_ylabel(paramnames[i])
            axes[0].set_title(figs_path)
            axes[-1].plot(sampler.lnprobability.T, color='k', alpha=0.3)
            axes[-1].set_ylabel('likelihood')   # bottom panel is the evolution of likelihood
            axes[-1].set_ylim(maxloglike-3*len(params), maxloglike)
            plt.tight_layout(h_pad=0)
            plt.subplots_adjust(hspace=0,wspace=0)
            plt.savefig(figs_path+"param_evol_iter"+str(iter)+".png", dpi=350)
            plt.show()
            # 2. corner plot - covariances of all parameters
            corner.corner(chain, quantiles=[0.16, 0.5, 0.84], labels=paramnames, show_titles=True)
            plt.title(figs_path)
            plt.savefig(figs_path+"corner_iter"+str(iter)+".png", dpi=350)
            plt.show()
            # 3. density and velocity dispersion profiles - same as before
            if not converged: plotprofiles(chain[::20], "MCMC_iter"+str(iter))
            iter += 1
    print("MCMC run time:", time.time()-start)
    plt.show()

    # now, plug resulting density and sigma profiles into the jeans equation
    # one mass profile for each trial of MCMC - do percentiles to get band on mass profile

    # M = (-sigr^2 * r / G) * (a + b + c)
    # a = dlnrho/dlnr
    # b = dlnsigr^2/dlnr
    # c = 2beta
    # beta = 1-(sigt/sigr)^2

    r  = numpy.logspace(0, 2, 201)
    lr = numpy.log(r)
    G = 4.3e-6  # (kpc km2) / (s2 Msun)

    def frac_error(r_est, r_true, M_est, M_true):
        frac_error = numpy.zeros(len(r_est))
        for i in range(len(r_est)):
            match_idx = (numpy.abs(r_true - r_est[i])).argmin()
            frac_error[i] = (M_est[i]-M_true[match_idx])/M_true[match_idx]
        return frac_error

    if datasetType == 'latte':
        r_true, M_true = numpy.loadtxt(fname=f"latte/{lattesim}/{lattesim}_smpl.csv", 
                                       delimiter=',', skiprows=1, unpack=True)
    elif datasetType == 'agama':
        r_true = r
        M_true = numpy.zeros(len(r))
        for i in range(len(r)):
            M_true[i] = -pot.force(r_true[i],0,0)[0]*r_true[i]**2/agama.G

    chain_smpl = chain[::20]

    dlnrho, dlnsigr, sigr, sigt = numpy.zeros((4, len(chain_smpl), len(r)))
    for i in range(len(chain_smpl)):
        dlnrho [i] = modelDensity(chain_smpl[i, 0:len(knots_logr)-1])(lr, der=1)
        dlnsigr[i] = modelSigma(chain_smpl[i, len(knots_logr)-1 : 2*len(knots_logr)-1])(lr, der=1)
        sigr   [i] = numpy.exp(modelSigma(chain_smpl[i, len(knots_logr)-1 : 2*len(knots_logr)-1])(lr))
        sigt   [i] = numpy.exp(modelSigma(chain_smpl[i, 2*len(knots_logr)-1 :])(lr))

    # dlnrho_low, dlnrho_med, dlnrho_upp = numpy.percentile(dlnrho, [16,50,84], axis=0)
    # dlnsigr_low, dlnsigr_med, dlnsigr_upp = numpy.percentile(dlnsigr, [16,50,84], axis=0)
    # sigr_low, sigr_med, sigr_upp = numpy.percentile(sigr, [16,50,84], axis=0)
    # sigt_low, sigt_med, sigt_upp = numpy.percentile(sigt, [16,50,84], axis=0)
    # beta_low = 1 - (sigt_low**2 / sigr_low**2)
    # beta_med = 1 - (sigt_med**2 / sigr_med**2)
    # beta_upp = 1 - (sigt_upp**2 / sigr_upp**2)
    # M_enc = -(sigr_med**2 * r / G)*(dlnrho_med + dlnsigr_med*2 + 2*beta_med)

    Mencs, betas = numpy.zeros((2, len(chain_smpl), len(r)))
    for i in range(len(chain_smpl)):
        betas[i] = 1 - (sigt[i]**2 / sigr[i]**2)
        Mencs[i] = -(sigr[i]**2 * r / G)*(dlnrho[i] + 2*dlnsigr[i] + 2*betas[i])
    #     plt.plot(r, M_encs[i])
    Menc_low, Menc_med, Menc_upp = numpy.percentile(Mencs, [16,50,84], axis=0)
    beta_low, beta_med, beta_upp = numpy.percentile(betas, [16,50,84], axis=0)

    # Anisotropy plot
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
    plt.savefig(figs_path+'anisotropy.jpg', dpi=250)
    plt.show()
    plt.cla()

    # Mass enclosed plot
    plt.rcParams.update({'font.size': 18})
    plt.rcParams['agg.path.chunksize'] = 10000  # overflow error on line 835 without this
    fig = plt.figure(figsize=(15,10), dpi=250)
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[3, 1])
    axs = gs.subplots(sharex=True)
    colors = {'m12f': '#4a0078', 'm12i': '#157F1F', 'm12m': '#931621', None: 'red'}
    color = colors[lattesim]

    axs[0].plot(r, Menc_med, c=color, linewidth=2.5, label='Jeans estimate')
    axs[0].fill_between(r, Menc_low, Menc_upp, color=color, alpha=0.3, lw=0, label=r'$\pm1\sigma$ interval')
    
    axs[0].plot(r_true, M_true, c='k', linewidth=1.5, linestyle='dashed', label='True')
    
    axs[1].plot(r, frac_error(r, r_true, Menc_med, M_true), c=color, linewidth=2.0)
    axs[1].fill_between(r, frac_error(r, r_true, Menc_low, M_true), 
                        frac_error(r, r_true, Menc_upp, M_true), color=color, 
                        alpha=0.3, lw=0)

    axs[0].set_title(figs_path)
    axs[1].axhline(0, c='k', linewidth=1)
    axs[1].axhline(0.2, c='k', linewidth=0.5, linestyle='dotted')
    axs[1].axhline(-0.2, c='k', linewidth=0.5, linestyle='dotted')
    axs[1].set_xlim([0,110])
    axs[0].set_ylabel(r"M(<r) ($M_{\odot}$)", size=24)
    axs[1].set_xlabel('Galactocentric Radius (kpc)', size=24)
    axs[1].set_ylabel('Fractional Error', size=20)
    axs[0].legend()
    axs[1].set_ylim([-0.40, 0.40])
    axs[1].set_yticks([-0.2, 0, 0.2])

    for ax in axs:
        ax.label_outer()

    plt.tight_layout()
    plt.savefig(figs_path+'jeans_result.jpg', dpi=250)
    plt.show()

    print(f"FINISHED AT {figs_path}")