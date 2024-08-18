import numpy as np, pandas as pd
import agama

Gmax  = 20.7
Gmin  = 16.0
bmin  = 30.0
decmin=-35.0
d2r   = np.pi/180

def load_RRL(SUBSAMPLE, VERBOSE):
    rrls = pd.read_csv("data/gmedinat/DESI-iron_RRLs_v0.2_qualcut.csv")
    if VERBOSE:
        print('Total number of particles in dataset:', len(rrls))
    
    ra = rrls['ra'].to_numpy()  # deg
    dec = rrls['dec'].to_numpy()  # deg
    ra  *= d2r  # ra and dec to rad
    dec *= d2r

    pmra = rrls['pmra'].to_numpy()  # mas/yr
    pmdec = rrls['pmdec'].to_numpy()  # mas/yr
    pmra_error = rrls['pmra_error'].to_numpy()  # mas/yr
    pmdec_error = rrls['pmdec_error'].to_numpy()  # mas/yr

    # TODO: fix
    PMerr = (pmra_error + pmdec_error) / 2  # mas/yr

    # TODO, restore to phot_g_mean_mag?
    Gapp = rrls['int_average_g'].to_numpy()  # mag

    vlos = rrls['v0_mean'].to_numpy()  # km/s
    vloserr = rrls['v0_std'].to_numpy()  # km/s

    # dist = rrls['dist_from_feh_k_mean'].to_numpy()  # kpc
    # disterr = rrls['dist_from_feh_k_err'].to_numpy()  # kpc TODO fix

    l, b, pml, pmb = agama.transformCelestialCoords(agama.fromICRStoGalactic,
                                                    ra, dec, pmra, pmdec)

    l /= d2r  # back to degrees
    b /= d2r
    dec /= d2r

    filt = (abs(b) >= bmin) * (dec >= decmin) * (Gapp > Gmin) * (Gapp < Gmax)

    lsr_info = (8.122, (12.9, 245.6, 7.78), 0.0208)

    return (l[filt], b[filt], None, Gapp[filt], pml[filt], pmb[filt],
            vlos[filt], PMerr[filt], vloserr[filt], None, None, lsr_info)
    
    
def load_BHB(SUBSAMPLE, VERBOSE):
    # TODO: fetch fixed abs mags from Amanda
    
    rrls = pd.read_csv("data/gmedinat/DESI-iron_RRLs_v0.2_qualcut.csv")
    if VERBOSE:
        print('Total number of particles in dataset:', len(rrls))
    
    ra = rrls['ra'].to_numpy()  # deg
    dec = rrls['dec'].to_numpy()  # deg
    ra  *= d2r  # ra and dec to rad
    dec *= d2r

    pmra = rrls['pmra'].to_numpy()  # mas/yr
    pmdec = rrls['pmdec'].to_numpy()  # mas/yr
    pmra_error = rrls['pmra_error'].to_numpy()  # mas/yr
    pmdec_error = rrls['pmdec_error'].to_numpy()  # mas/yr

    # TODO: fix
    PMerr = (pmra_error + pmdec_error) / 2  # mas/yr

    # TODO, restore to phot_g_mean_mag?
    Gapp = rrls['int_average_g'].to_numpy()  # mag

    vlos = rrls['v0_mean'].to_numpy()  # km/s
    vloserr = rrls['v0_std'].to_numpy()  # km/s

    # dist = rrls['dist_from_feh_k_mean'].to_numpy()  # kpc
    # disterr = rrls['dist_from_feh_k_err'].to_numpy()  # kpc TODO fix

    l, b, pml, pmb = agama.transformCelestialCoords(agama.fromICRStoGalactic,
                                                    ra, dec, pmra, pmdec)

    l /= d2r  # back to degrees
    b /= d2r
    dec /= d2r

    filt = (abs(b) >= bmin) * (dec >= decmin) * (Gapp > Gmin) * (Gapp < Gmax)

    lsr_info = (8.122, (12.9, 245.6, 7.78), 0.0208)

    return (l[filt], b[filt], None, Gapp[filt], pml[filt], pmb[filt],
            vlos[filt], PMerr[filt], vloserr[filt], None, None, lsr_info)