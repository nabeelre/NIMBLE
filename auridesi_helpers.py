from astropy.table import Table
from astropy.io import fits
import astropy.coordinates as coord
import astropy.units as u

import numpy as np, pandas as pd, h5py, agama
import matplotlib.pyplot as plt

Gmax  = 20.0
Gmin  = 16.0
Grrl  = 0.58
DMerr = 0.24
bmin  = 20.0
decmin=-35.0
d2r   = np.pi/180

# adapted from Alex Riley's AuriDESI tutorial and Namitha's how_to_use_mocks.ipynb
# https://github.com/desimilkyway/tutorials/blob/main/ahriley/auridesi-demo-hawaii.ipynb
nersc_path = "/global/cfs/cdirs/desi/users/namitha/Aurigaia/AuriDESI_Mocks_Spectroscopic_Catalog"


def get_lsr_frame(halonum):
    galcen_distance = 8.000*u.kpc
    z_sun = 2e-2*u.kpc

    usun = 11.1
    vsun = 12.24
    wsun =  7.25
    
    if halonum == "06":
        vlsr = 229.2225816451948
    elif halonum == "16":
        vlsr = 213.5378191707707
    elif halonum == "21":
        vlsr = 226.5683036672675
    elif halonum == "23":
        vlsr = 234.6171739867179
    elif halonum == "24":
        vlsr = 218.7874144767017
    elif halonum == "27":
        vlsr = 250.8076194638379
    else:
        print("Could not find halo")
        exit()

    vsun = vsun + vlsr
    galcen_v_sun = [usun, vsun, wsun]*u.km/u.s

    return galcen_distance.value, galcen_v_sun.value, z_sun.to(u.kpc).value


def write_mockRRL(halonum, lsrdeg, source_path = None, write_path = None):
    """
    Read AuriDESI fits files and output csv of mock RRL stars ready for input
    into deconv.py

    Parameters
    ----------
    halonum: str
        Auriga halo number of mock to write ("06", "16", "21", "23", "24", "27")

    lsrdeg: str
        LSR position for AuriDESI mock ("030", "120", "210", "300")

    source_path: str
        Path to load AuriDESI mocks from

    write_path: str
        Path to write RRL mocks to
    """

    print(f"Writing mock RRL sample for H{halonum} at {lsrdeg}deg")

    if source_path is None:
        source_path = f"data/AuriDESI_Spec/{lsrdeg}_deg/H{halonum}_{lsrdeg}deg_mock.fits"
    header = fits.open(source_path)[0].header

    rvtab =    Table.read(source_path, hdu='rvtab')
    fibermap = Table.read(source_path, hdu='fibermap')
    gaia =     Table.read(source_path, hdu='gaia')
    true =     Table.read(source_path, hdu='true_values')
    # prog =     Table.read(source_path, hdu='progenitors')

    print("Initial particle count:", len(rvtab))

    dist = coord.Distance(parallax=true['PARALLAX']*u.mas)
    true['GMAG'] = true['APP_GMAG'] - dist.distmod.value

    select_RRL = (true['AGE'] > 10) & (true['MASS'] > 0.7) & (true['MASS'] < 0.9) & \
                 (true['FEH'] < -0.5) & (true['TEFF'] > 6000) & (true['TEFF'] < 7000) & \
                 (true['GMAG'] > 0.45) & (true['GMAG'] < 0.65)

    def is_RRL(arr):
        return arr[select_RRL]
    
    rrls = Table(
        list(map(is_RRL, [gaia['RA'], gaia['DEC'], gaia['PMRA'], 
                          gaia['PMRA_ERROR'], gaia['PMDEC'], gaia['PMDEC_ERROR'],
                          rvtab['VRAD'], rvtab['VRAD_ERR'], 
                          fibermap['GAIA_PHOT_G_MEAN_MAG'], true['RA'], 
                          true['DEC'], true['PMRA'], true['PMDEC'], 
                          true['PARALLAX'], true['VRAD']])), 
        names=['RA', 'DEC', 'PMRA', 'PMRA_ERROR', 'PMDEC', 'PMDEC_ERROR', 
               'VRAD', 'VRAD_ERR', 'GAIA_PHOT_G_MEAN_MAG', 'TRUE_RA', 
               'TRUE_DEC', 'TRUE_PMRA', 'TRUE_PMDEC', 'TRUE_PARALLAX', 
               'TRUE_VRAD']
    )
    print("RR Lyrae count:", len(rrls), "\n")

    if write_path is None:
        write_path = f"data/AuriDESI_Spec/{lsrdeg}_deg/H{halonum}_{lsrdeg}deg_mockRRL.csv"
    rrls.write(write_path, delimiter=',', format='ascii', overwrite=True)
    

def write_true(halonum):
    """
    Read Auriga snapshot and write true cumulative mass profile

    Parameters:
    ----------
    halonum: str
        Auriga halo number of mock to write ("06", "16", "21", "23", "24", "27")
    
    Auriga halos downloaded from:
    https://wwwmpa.mpa-garching.mpg.de/auriga/data.html
    """
    filename = f"data/auriga/H{halonum}/snapshot_reduced_temprho_halo_{halonum}_063.hdf5"
    f = h5py.File(filename, 'r')

    # (Z, Y, X) converted from Mpc to kpc
    star_wind_coordinates   = f['PartType4']['Coordinates'][:]*1000.
    # converted from 10^10 Msun to 1 Msun
    star_wind_masses        = f['PartType4']['Masses'][:]*1e10

    dm_coordinates = f['PartType1']['Coordinates'][:]*1000.
    dm_masses      = f['PartType1']['Masses'][:]*1e10

    gas_coordinates = f['PartType0']['Coordinates'][:]*1000.
    gas_masses      = f['PartType0']['Masses'][:]*1e10

    bh_coordinates = f['PartType5']['Coordinates'][:]*1000.
    bh_masses      = f['PartType5']['Masses'][:]*1e10

    f.close()

    total_mass = np.sum(np.hstack((dm_masses, star_wind_masses, 
                                   gas_masses, bh_masses)))
    print("total mass (1e12 Msun):", total_mass/1e12)

    # r_sph = sqrt(x^2 + y^2 + z^2)
    star_wind_radii = np.sqrt((star_wind_coordinates[:,2] ** 2) + 
                              (star_wind_coordinates[:,1] ** 2) + 
                              (star_wind_coordinates[:,0] ** 2))

    gas_radii = np.sqrt((gas_coordinates[:,2] ** 2) + 
                        (gas_coordinates[:,1] ** 2) + 
                        (gas_coordinates[:,0] ** 2))

    dm_radii = np.sqrt((dm_coordinates[:,2] ** 2) + 
                       (dm_coordinates[:,1] ** 2) + 
                       (dm_coordinates[:,0] ** 2))
    
    bh_radii = np.sqrt((bh_coordinates[:,2] ** 2) + 
                       (bh_coordinates[:,1] ** 2) + 
                       (bh_coordinates[:,0] ** 2))

    # rows are particles
    # column 0 is spherical galactocentric radius
    # column 1 is particle mass
    star_wind = np.column_stack((star_wind_radii, star_wind_masses))[star_wind_radii.argsort()]
    gas = np.column_stack((gas_radii, gas_masses))[gas_radii.argsort()]
    dm = np.column_stack((dm_radii, dm_masses))[dm_radii.argsort()]
    bh = np.column_stack((bh_radii, bh_masses))[bh_radii.argsort()]

    rgrid = np.linspace(0,120,1000)

    M_cumul_star_wind = np.interp(rgrid, star_wind[:,0], np.cumsum(star_wind[:,1]))
    M_cumul_gas       = np.interp(rgrid, gas[:,0],       np.cumsum(gas[:,1]))
    M_cumul_dm        = np.interp(rgrid, dm[:,0],        np.cumsum(dm[:,1]))
    M_cumul_bh        = np.interp(rgrid, bh[:,0],        np.cumsum(bh[:,1]))
    
    M_cumul = M_cumul_star_wind + M_cumul_gas + M_cumul_dm + M_cumul_bh

    np.savetxt(
        fname=f"data/auriga/H{halonum}/Au{halonum}_true.csv",
        X=np.stack([rgrid, M_cumul], axis=1), delimiter=',',
        header="r [kpc], M(<r)_true [Msun]"
    )

    plt.figure()
    plt.plot(rgrid, M_cumul, label='total')
    plt.plot(rgrid, M_cumul_star_wind, label='star_wind')
    plt.plot(rgrid, M_cumul_gas, label='gas')
    plt.plot(rgrid, M_cumul_dm, label='dm')
    plt.plot(rgrid, M_cumul_bh, label='bh')
    plt.legend()
    plt.xlabel("Galactocentric radius [kpc]")
    plt.ylabel("Mass enclosed [Msun]")
    plt.title(f"Auriga H{halonum}")
    plt.savefig(f"data/auriga/H{halonum}/Au{halonum}_true.pdf", bbox_inches='tight')
    plt.cla()

    print(f"Finished writing true mass profile of Auriga halo{halonum}\n")


def halo_velocity_density_profiles(halonum):
    filename = f"data/auriga/H{halonum}/snapshot_reduced_temprho_halo_{halonum}_063.hdf5"

    f = h5py.File(filename, 'r')

    # Extract data from Auriga hdf5 file
    star_wind_coordinates = f['PartType4']['Coordinates']             [:]*1000  # (Z, Y, X) in kpc
    star_wind_metallicity = f['PartType4']['GFM_Metallicity']         [:]
    star_wind_form_time   = f['PartType4']['GFM_StellarFormationTime'][:]       # Gyr
    star_wind_velocities  = f['PartType4']['Velocities']              [:]       # (Vz, Vy, Vx) in km/s

    # Set floor for halo metallicity, some particles have 
    star_wind_metal_mask = star_wind_metallicity < 1e-7
    star_wind_metallicity[star_wind_metal_mask] = 1e-7

    log_star_wind_metallicity = np.log10(star_wind_metallicity / 0.0127)

    # Old, low metallicity to select halo - also removes wind particles
    halo = (log_star_wind_metallicity < -1.5) & (star_wind_form_time > 8)

    halo_star_coordinates = star_wind_coordinates    [halo]
    halo_star_velocities  = star_wind_velocities     [halo]

    halo_star_x = halo_star_coordinates[:,2]
    halo_star_y = halo_star_coordinates[:,1]
    halo_star_z = halo_star_coordinates[:,0]

    halo_star_radii = np.sqrt((halo_star_x ** 2) + 
                              (halo_star_y ** 2) + 
                              (halo_star_z ** 2))
    
    halo_star_vx = halo_star_velocities[:,2]
    halo_star_vy = halo_star_velocities[:,1]
    halo_star_vz = halo_star_velocities[:,0]

    rvel, tvel, pvel = cartesian_to_spherical(halo_star_x, halo_star_y, 
                                              halo_star_z, halo_star_vx, 
                                              halo_star_vy, halo_star_vz)
    
    truesig_knots = np.logspace(0, np.log10(100), 10)

    # velocity squared as function of log r
    true_sigmar = agama.splineApprox(np.log(truesig_knots), 
                                     np.log(halo_star_radii), 
                                     rvel**2)

    true_sigmat = agama.splineApprox(np.log(truesig_knots), 
                                     np.log(halo_star_radii), 
                                     (tvel**2 + pvel**2)/2)
    
    return true_sigmar, true_sigmat


def load(halonum, lsrdeg, SUBSAMPLE, VERBOSE):
    rrls = pd.read_csv(f"data/AuriDESI_Spec/{lsrdeg}_deg/H{halonum}_{lsrdeg}deg_mockRRL.csv")
    print("Number of initial particles:", len(rrls))

    lsr_info = get_lsr_frame(halonum)

    ra = rrls['RA'].to_numpy()  # deg
    dec = rrls['DEC'].to_numpy()  # deg
    ra  *= d2r  # ra and dec to rad
    dec *= d2r

    pmra = rrls['PMRA'].to_numpy()  # mas/yr
    pmdec = rrls['PMDEC'].to_numpy()  # mas/yr
    pmra_error = rrls['PMRA_ERROR'].to_numpy()  # mas/yr
    pmdec_error = rrls['PMDEC_ERROR'].to_numpy()  # mas/yr

    # TODO: fix
    PMerr = (pmra_error + pmdec_error) / 2  # mas/yr

    Gapp = rrls['GAIA_PHOT_G_MEAN_MAG'].to_numpy()  # mag
    dist = 10**((Grrl - Gapp - 5)/-5)  # pc

    vlos = rrls['VRAD'].to_numpy()  # km/s
    vloserr = rrls['VRAD_ERR'].to_numpy()  # km/s

    l, b, pml, pmb = agama.transformCelestialCoords(agama.fromICRStoGalactic,
                                                    ra, dec, pmra, pmdec)
    x, y, z = agama.getGalactocentricFromGalactic(l, b, dist)
    radii = np.sqrt(x**2 + y**2 + z**2)  # kpc

    # back to degrees
    l   /= d2r  
    b   /= d2r
    dec /= d2r

    filt = (abs(b) >= bmin) * (dec >= decmin) * (Gapp > Gmin) * (Gapp < Gmax)

    true_sigmar, true_sigmat = halo_velocity_density_profiles(halonum)

    return (l[filt], b[filt], radii, Gapp[filt], pml[filt], pmb[filt],
            vlos[filt], PMerr[filt], vloserr[filt], true_sigmar, true_sigmat, lsr_info)


def cartesian_to_spherical(xpos, ypos, zpos, xVel, yVel, zVel):
    numParticles = len(xpos)

    sinTheta = np.sqrt(xpos**2 + ypos**2) / np.sqrt(xpos**2 + ypos**2 + zpos**2)
    cosTheta = zpos / np.sqrt(xpos**2 + ypos**2 + zpos**2)
    sinPhi = ypos / np.sqrt(xpos**2 + ypos**2)
    cosPhi = xpos / np.sqrt(xpos**2 + ypos**2)

    for i in range(0, numParticles):
        conversionMatrix = [[sinTheta[i] * cosPhi[i], sinTheta[i] * sinPhi[i],  cosTheta[i]],
                            [cosTheta[i] * cosPhi[i], cosTheta[i] * sinPhi[i], -sinTheta[i]],
                            [       -sinPhi[i]      ,        cosPhi[i]       ,        0    ]]

    velMatrix = [ xVel, yVel, zVel ]

    sphereVels = np.matmul(conversionMatrix, velMatrix)

    rVel = sphereVels[0]
    tVel = sphereVels[1]
    pVel = sphereVels[2]
    
    return rVel, tVel, pVel


if __name__ == "__main__":
    halonums = ["06", "16", "21", "23", "24", "27"]
    lsrdegs = ["030", "120", "210", "300"]

    for lsrdeg in lsrdegs:
        for halonum in halonums:
            source_path = nersc_path+f"/{lsrdeg}_deg/H{halonum}_{lsrdeg}deg_mock.fits"
            write_mockRRL(halonum, lsrdeg, source_path)

    # for halonum in halonums:
    #     write_true(halonum)