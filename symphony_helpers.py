# from operator import index
# from astropy.table import Table
# from astropy.io import fits
# from astropy.coordinates import SkyCoord, Galactocentric
# import astropy.units as u

import numpy as np
import pandas as pd
# import h5py
import agama
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon

Gmax  = 19.0
Gmin  = 16.0

dmax  = 60.6
dmin  = 5.0

bmin  = 30.0
decmin=-35.0
d2r   = np.pi/180


def get_lsr_frame(halonum):
    # only for one halo and one solar position
    return 8, [12.9, 245.6, 7.78], 2.08e-2


def load_BHB(halonum, lsrdeg, SUBSAMPLE):
    fname = f"data/symphony/survey.mwest_scaled_Halo{halonum}_sunrot{lsrdeg}_0kpc300kpc.0_bhb_comb_processed_qualcut.csv"

    bhbs = pd.read_csv(fname, index_col=None)

    print("Number of initial particles:", len(bhbs))

    lsr_info = get_lsr_frame(halonum)

    ra = np.asarray(bhbs['RA'])  # deg
    dec = np.asarray(bhbs['DEC'])  # deg
    ra  *= d2r  # ra and dec to rad
    dec *= d2r

    pmra = np.asarray(bhbs['PMRA'])  # mas/yr
    pmdec = np.asarray(bhbs['PMDEC'])  # mas/yr
    pmra_error = np.asarray(bhbs['PMRA_ERROR'])  # mas/yr
    pmdec_error = np.asarray(bhbs['PMDEC_ERROR'])  # mas/yr

    PMerr = (pmra_error + pmdec_error) / 2  # mas/yr

    vlos = bhbs['RV_HELIO'].to_numpy()  # km/s
    vloserr = bhbs['RV_ERR'].to_numpy()  # km/s

    dist = bhbs['DISTANCE_HELIO'].to_numpy()  # kpc
    disterr = dist * 0.10  # TODO improve dist err

    Gapp = bhbs['GAIA_PHOT_G_MAG'].to_numpy()  # mag

    l, b, pml, pmb = agama.transformCelestialCoords(agama.fromICRStoGalactic,
                                                    ra, dec, pmra, pmdec)

    # back to degrees
    l   /= d2r
    b   /= d2r
    dec /= d2r

    filt = (dist > dmin) * (dist < dmax)
    # (abs(b) >= bmin) * (dec >= decmin) *

    # true_sigmar, true_sigmat, true_dens_radii = halo_velocity_density_profiles(halonum)

    return (l[filt], b[filt], None, dist[filt], Gapp[filt], pml[filt], pmb[filt],
            vlos[filt], PMerr[filt], disterr[filt], vloserr[filt], None, None, lsr_info)


if __name__ == "__main__":
    halonums = ["06", "16", "21", "23", "24", "27"]
    lsrdegs = ["030", "120", "210", "300"]

    # for lsrdeg in lsrdegs:
    #     for halonum in halonums:
    #         source_path = nersc_path+f"/{lsrdeg}_deg/H{halonum}_{lsrdeg}deg_mock.fits"
    #         write_mockRRL(halonum, lsrdeg, source_path)

    # for halonum in halonums:
    #     write_true(halonum)
