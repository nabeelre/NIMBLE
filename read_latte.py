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

import sys, os, matplotlib.pyplot as plt, numpy as np, gizmo_read

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Cannot understand command line args")
        exit()

    sim = sys.argv[1]
    assert(sim in ['m12f', 'm12i', 'm12m'])

    if os.path.exists(f"data/{sim}/{sim}_prejeans.csv") and os.path.exists(f"data/{sim}/{sim}_true.csv"):
        print(f"Prepped Latte {sim} files already exist")
        print("  Delete the files if you'd like to regenerate them")
        exit()

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
