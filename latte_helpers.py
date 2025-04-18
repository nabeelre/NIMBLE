"""
Read Latte simulation snapshots and prepare them for use with jeans_bspline.py

The Latte suite of FIRE-2 cosmological zoom-in baryonic simulations of Milky
Way-mass galaxies (Wetzel et al 2016), part of the Feedback In Realistic
Environments (FIRE) simulation project, were run using the Gizmo gravity plus
hydrodynamics code in meshless finite-mass (MFM) mode (Hopkins 2015) and the
FIRE-2 physics model (Hopkins et al 2018).

See run_latte_errorfree.sh for instructions on downloading the Latte data from
https://girder.hub.yt/#collection/5b0427b2e9914800018237da/folder/5b211e5a323d120001c7a826
"""

import sys, os, matplotlib.pyplot as plt, numpy as np, gizmo_read, agama
from pygaia.errors.astrometric import total_proper_motion_uncertainty

Gmax  = 20.7
Gmin  = 16.0
Grrl  = 0.58
DMerr = 0.24
bmin  = 30.0
decmin=-35.0
d2r   = np.pi/180


def rotate_x(x_old, y_old, theta):
    return x_old*np.cos(theta) + y_old*np.sin(theta)


def rotate_y(x_old, y_old, theta):
    return -1*x_old*np.sin(theta) + y_old*np.cos(theta)


def get_lsr_cartesian(sim, lsr):
    # Solar positions and velocities of local standards of rest used by the Ananke catalogue (Sanderson et al. 2020)
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


def write_mock_and_true(sim):
    # Preparing _prejeans.csv file
    part = gizmo_read.read.Read.read_snapshot(species=['star'], directory=f'data/{sim}',
                                              properties=['position', 'velocity', 'mass', 'massfraction', 'form.scalefactor'])

    x = part['star']['position'][:,0]
    y = part['star']['position'][:,1]
    z = part['star']['position'][:,2]
    r = (x**2+y**2+z**2)**0.5

    radlim = (r < 600)
    r = r[radlim]

    Y_part = part['star']['massfraction'][:,1][radlim]  # mass fraction of He
    Z_part = part['star']['massfraction'][:,0][radlim]  # mass fraction of metals (everything except H, He)
    X_part = 1 - (Y_part + Z_part)                      # mass fraction of H
    print(f"X_part_avg={np.mean(X_part):.4f}, Y_part_avg={np.mean(Y_part):.4f}, Z_part_avg={np.mean(Z_part):.4f}")

    X_sun = gizmo_read.constant.sun_composition['hydrogen']['massfraction']
    Y_sun = gizmo_read.constant.sun_composition['helium']['massfraction']
    Z_sun = gizmo_read.constant.sun_composition['metals']['massfraction']
    print(f"X_sun={X_sun:.4f}, Y_sun={Y_sun:.4f}, Z_sun={Z_sun:.4f}")

    metallicity = np.log10(Z_part/X_part) - np.log10(Z_sun/X_sun) # [M/H]
    age = part['star']['age'][radlim]

    # Scatter of star particle [M/H] vs galactocentric radius
    # plt.figure()
    # plt.ylabel('[M/H]')
    # plt.xlabel('Radius (kpc)')
    # plt.ylim([-5.5,2])
    # plt.scatter(r[::100], metallicity[::100], marker='.', alpha=0.1)
    # plt.axhline(-1.5, c='r', label='[M/H] = -1.5 cutoff')
    # plt.legend()
    # plt.show()

    # Scatter of star particle age vs galactocentric radius
    # plt.figure()
    # plt.ylabel('Age (Gyr)')
    # plt.xlabel('Radius (kpc)')
    # plt.scatter(r[::100], age[::100], marker='.', alpha=0.1)
    # plt.axhline(8, c='r', label='Age = 8 Gyr cutoff')
    # plt.legend()
    # plt.show()

    # [M/H] and age threshold to select halo stars
    halo = (metallicity < -1.5) * (age > 8)

    # apply halo selection to all quantities
    x = x[radlim][halo]; y = y[radlim][halo]; z = z[radlim][halo]; r = r[halo]
    vx = part['star']['velocity'][:,0][radlim][halo]
    vy = part['star']['velocity'][:,1][radlim][halo]
    vz = part['star']['velocity'][:,2][radlim][halo]
    m  = part['star']['mass'][radlim][halo]

    # precompute spherical velocities for use in jeans
    sphvels = gizmo_read.coordinate.get_velocities_in_coordinate_system(part['star']['velocity'][radlim][halo],
                                                                        part['star']['position'][radlim][halo],
                                                                        system_from='cartesian',
                                                                        system_to='spherical')
    vr, vtheta, vphi = np.transpose(sphvels)

    # Write data to disk
    np.savetxt(
        fname=f"data/{sim}/{sim}_prejeans.csv",
        X=np.stack([x, y, z, vx, vy, vz, m, r, vr**2, vtheta**2, vphi**2], axis=1), delimiter=',',
        header="x, y, z [kpc], vx, vy, vz [km/s], mass [Msun], gc_radius, vr_sq, vtheta_sq, vphi_sq [km2/s2]"
    )

    # Cleaning up before moving on to next file
    del part, x, y, z, r, vx, vy, vz, sphvels, vr, vtheta, vphi, X_part, Y_part, Z_part, metallicity, age, halo

    # Preparing _true.csv file
    # Need radii and masses of all particle types for true enclosed mass profile
    part = gizmo_read.read.Read.read_snapshot(species=['star', 'dark', 'gas'],
                                              properties=['position', 'mass'],
                                              directory=f'data/{sim}')

    # calculate radii of each particle type
    r_star = (part['star']['position'][:,0]**2+part['star']['position'][:,1]**2+part['star']['position'][:,2]**2)**0.5
    r_dark = (part['dark']['position'][:,0]**2+part['dark']['position'][:,1]**2+part['dark']['position'][:,2]**2)**0.5
    r_gas  = (part['gas'] ['position'][:,0]**2+part['gas'] ['position'][:,1]**2+part['gas'] ['position'][:,2]**2)**0.5

    radlim_star = (r_star < 600); radlim_dark = (r_dark < 600); radlim_gas  = (r_gas < 600)

    r_star = r_star[radlim_star]; r_dark = r_dark[radlim_dark]; r_gas = r_gas[radlim_gas]

    # collect radii and masses of all particles and sort by radius
    all_radii = np.concatenate((r_star, r_dark, r_gas))
    all_masses = np.concatenate((part['star']['mass'][radlim_star],
                                 part['dark']['mass'][radlim_dark],
                                 part['gas']['mass'][radlim_gas]))

    sorter = np.argsort(all_radii)
    all_radii  = all_radii [sorter]
    all_masses = all_masses[sorter]

    # cumulative sum of sorted masses is mass enclosed
    Menc = np.cumsum(all_masses, dtype=np.float64)

    # Thin profile to save memory
    all_radii = all_radii[::500]; Menc = Menc[::500]

    # Check for a smooth profile after thinning
    # plt.plot(all_radii, Menc)
    # plt.xlim([0,100])
    # plt.xlabel('gc radius [kpc]')
    # plt.ylabel('True M(<r) [Msun]')
    # plt.show()

    # Write data to disk
    np.savetxt(
        fname=f"data/{sim}/{sim}_true.csv",
        X=np.stack([all_radii, Menc], axis=1), delimiter=',',
        header="r [kpc], M(<r)_true [Msun]"
    )

    print(f"Finished preparing {sim} for use with jeans_bspline.py\n")


def rotate_coords(sim, lsr, positions, velocities):
    # Rotate coordinates of latte simulation (sim = "m12f" or "m12i" or "m12m")
    # to put the solar position of the local standard of rest numbered lsr
    # on the -x axis as required by agama
    # positions and velocities are an Nx3 matricies with columns x, y, z and vx, vy, vz respectively
    # all coordinates here are galactocentric cartesian
    x_sun_orig, y_sun_orig, z_sun_orig, \
        vx_sun_orig, vy_sun_orig , vz_sun_orig = get_lsr_cartesian(sim, lsr)

    theta  = np.arctan2(y_sun_orig, x_sun_orig) + np.pi  # angle in radians to rotate to -x axis
    x_sun  = rotate_x(x_sun_orig,  y_sun_orig, theta)    # kpc
    y_sun  = rotate_y(x_sun_orig,  y_sun_orig, theta)    # kpc
    vx_sun = rotate_x(vx_sun_orig, vy_sun_orig, theta)  # km/s
    vy_sun = rotate_y(vx_sun_orig, vy_sun_orig, theta)  # km/s

    galcen_v_sun_sim    = (vx_sun, vy_sun, vz_sun_orig)
    galcen_distance_sim = np.sqrt(x_sun**2 + y_sun**2)

    x_new  = rotate_x(positions[:,0],  positions[:,1], theta)
    y_new  = rotate_y(positions[:,0],  positions[:,1], theta)
    vx_new = rotate_x(velocities[:,0], velocities[:,1], theta)
    vy_new = rotate_y(velocities[:,0], velocities[:,1], theta)

    return (galcen_distance_sim, galcen_v_sun_sim, z_sun_orig), \
        x_new, y_new, vx_new, vy_new


def load(lattesim, lsr, gaia_release, SUBSAMPLE, VERBOSE):
    # Load {lattesim}_prejeans.csv file written by read_latte.ipynb notebook
    # files with '_prejeans.csv' have the following properties:
    # positions in kpc, velocities in km/s, mass in Msun
    # arranged as [x, y, z, vx, vy, vz, m, gc_radius, vr_sq, vtheta_sq, vphi_sq]
    x, y, z, vx, vy, vz, \
        mass, radii, vr_sq, vtheta_sq, vphi_sq = np.loadtxt(f"data/{lattesim}/{lattesim}_prejeans.csv",
                                                                unpack=True, skiprows=1, delimiter=',')

    if SUBSAMPLE:
        # optionally subsample the dataset down to 1/10th the original size
        if VERBOSE:
            print("Number of particles before subsample:", len(x))
        sample_size = int(len(x)/4)

        sample_idxs = np.random.choice(np.arange(len(x)), sample_size, replace=False)

        x = x[sample_idxs]; y = y[sample_idxs]; z = z[sample_idxs]
        vx = vx[sample_idxs]; vy = vy[sample_idxs]; vz = vz[sample_idxs]
        mass = mass[sample_idxs]; radii = radii[sample_idxs]
        vr_sq = vr_sq[sample_idxs]; vtheta_sq = vtheta_sq[sample_idxs]; vphi_sq = vphi_sq[sample_idxs]
        if VERBOSE:
            print("Number of particles after subsample:", len(x))
    elif VERBOSE:
        print("Number of particles:", len(x))

    # Align solar position of LSR{lsr} onto the -x axis for transformations with agama
    lsr_info, x_new, y_new, vx_new, vy_new = rotate_coords(lattesim, lsr,
                                                            np.column_stack((x, y, z)),
                                                            np.column_stack((vx, vy, vz)))

    xv = np.column_stack((x_new, y_new, z, vx_new, vy_new, vz))
    nbody = len(x_new)

    # True velocity dispersion profiles used later for plotting comparisons
    truesig_knots = np.logspace(0, np.log10(200), 15)
    true_sigmar = agama.splineApprox(np.log(truesig_knots), np.log(radii), vr_sq)
    true_sigmat = agama.splineApprox(np.log(truesig_knots), np.log(radii), (vtheta_sq + vphi_sq)/2)

    # Use agama to get positions and velocities in galactic and equatorial coordinates
    l, b, dist, pml, pmb, vlos = agama.getGalacticFromGalactocentric(*xv.T, *lsr_info)
    ra, dec, pmra, pmdec  = agama.transformCelestialCoords(agama.fromGalactictoICRS, l, b, pml, pmb)
    l   /=d2r;  b   /=d2r   # convert from radians to degrees
    ra  /=d2r;  dec /=d2r
    pml /=4.74; pmb /=4.74  # convert from km/s/kpc to mas/yr
    pmra/=4.74; pmdec/=4.74

    # impose spatial selection based on the survey footprint
    filt = (abs(b) >= bmin) * (dec >= decmin)

    # compute apparent G-band magnitude and impose a cut Gmin<G<Gmax
    Gabs = Grrl + np.random.normal(size=nbody) * DMerr  # abs.mag with scatter
    Gapp = Gabs + 5*np.log10(dist) + 10  # apparent magnitude
    filt *= (Gapp > Gmin) * (Gapp < Gmax)

    # Doesnt work with current cov matrix set up
    # pmracosdec_err, pmdec_err = proper_motion_uncertainty(Gapp, release=gaia_release)  # uas/yr
    # pmra_err *= 0.001; pmdec_err *= 0.001  # uas/yr -> mas/yr

    # pull magnitude of proper motion errors from pygaia
    PMerr = total_proper_motion_uncertainty(Gapp, release=gaia_release.lower())  # uas/yr
    PMerr *= 0.001  # uas/yr -> mas/yr

    # add proper motion errors
    pmra  += np.random.normal(size=nbody) * PMerr
    pmdec += np.random.normal(size=nbody) * PMerr

    # RA, Dec back to radians
    # go back to galactic coords
    ra *= d2r
    dec *= d2r
    l, b, pml, pmb = agama.transformCelestialCoords(agama.fromICRStoGalactic,
                                                    ra, dec, pmra, pmdec)
    l /= d2r
    b /= d2r

    # add Vlos errors
    vloserr = np.ones(nbody) * 10.0  # km/s
    vlos += np.random.normal(size=nbody) * vloserr

    return (l[filt], b[filt], radii, Gapp[filt], pml[filt], pmb[filt],
            vlos[filt], PMerr[filt], vloserr[filt], true_sigmar, true_sigmat,
            lsr_info)


if __name__ == "__main__":
    write_mock_and_true(sys.argv[1])
