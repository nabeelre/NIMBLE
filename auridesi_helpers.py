from astropy.table import Table
from astropy.io import fits
import astropy.coordinates as coord
import astropy.units as u

import numpy as np, h5py

Gmax  = 20.7
Gmin  = 16.0
Grrl  = 0.58
DMerr = 0.24
bmin  = 30.0
decmin=-35.0
d2r   = np.pi/180

# adapted from Alex Riley's AuriDESI tutorial and Namitha's how_to_use_mocks.ipynb
# https://github.com/desimilkyway/tutorials/blob/main/ahriley/auridesi-demo-hawaii.ipynb
nersc_path = "/global/cfs/cdirs/desi/users/namitha/Aurigaia/AuriDESI_Mocks_Spectroscopic_Catalog"


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
    
    rrls = Table(list(map(is_RRL, [
        gaia['RA'], gaia['DEC'], gaia['PMRA'], gaia['PMRA_ERROR'], 
        gaia['PMDEC'], gaia['PMDEC_ERROR'], rvtab['VRAD'], rvtab['VRAD_ERR'], 
        fibermap['GAIA_PHOT_G_MEAN_MAG']
    ])))
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
    """
    filename = f"data/AuriDESI_Spec/auriga_halos/snapshot_reduced_temprho_halo_{halonum}_063.hdf5"
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
        fname=f"data/AuriDESI_Spec/auriga_halos/Au{halonum}_true.csv",
        X=np.stack([rgrid, M_cumul], axis=1), delimiter=',',
        header="r [kpc], M(<r)_true [Msun]"
    )

    print(f"Finished writing true mass profile of Auriga halo{halonum}")


def load():
    return


if __name__ == "__main__":
    halonums = ["06", "16", "21", "23", "24", "27"]
    lsrdegs = ["030", "120", "210", "300"]

    for lsrdeg in lsrdegs:
        for halonum in halonums:
            source_path = nersc_path+f"/{lsrdeg}_deg/H{halonum}_{lsrdeg}deg_mock.fits"
            write_mockRRL(lsrdeg, halonum, source_path)

    # for halonum in halonums:
    #     write_true(halonum)