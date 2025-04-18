import numpy as np
import pandas as pd
import agama

Gmax  = 20.0
Gmin  = 16.0

dmax  = 60.0
dmin  = 30.0

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

    filt = (Gapp > Gmin) * (Gapp < Gmax)
    # (abs(b) >= bmin) * (dec >= decmin)

    lsr_info = (8.122, (12.9, 245.6, 7.78), 0.0208)

    return (l[filt], b[filt], None, Gapp[filt], pml[filt], pmb[filt],
            vlos[filt], PMerr[filt], vloserr[filt], None, None, lsr_info)


def load_BHB(SUBSAMPLE):
    bhbs = pd.read_csv("data/abystrom/iron_bhb_240830_qualcut.csv", index_col=None)
    print('Total number of particles in dataset:', len(bhbs))

    ra = bhbs['RA'].to_numpy()  # deg
    dec = bhbs['DEC'].to_numpy()  # deg
    ra  *= d2r  # ra and dec to rad
    dec *= d2r

    pmra = bhbs['PMRA'].to_numpy()  # mas/yr
    pmdec = bhbs['PMDEC'].to_numpy()  # mas/yr
    pmra_error = bhbs['PMRA_ERROR'].to_numpy()  # mas/yr
    pmdec_error = bhbs['PMDEC_ERROR'].to_numpy()  # mas/yr

    PMerr = (pmra_error + pmdec_error) / 2  # mas/yr

    vlos = bhbs['VRAD'].to_numpy()  # km/s
    vloserr = bhbs['VRAD_ERR'].to_numpy()  # km/s

    dist = bhbs['dist'].to_numpy()  # kpc
    disterr = bhbs['dist_err'].to_numpy()  # kpc

    Gapp = bhbs['PHOT_G_MEAN_MAG'].to_numpy()  # mag

    l, b, pml, pmb = agama.transformCelestialCoords(agama.fromICRStoGalactic,
                                                    ra, dec, pmra, pmdec)

    l /= d2r  # back to degrees
    b /= d2r
    dec /= d2r

    filt = (dist > dmin) * (dist < dmax)
    # (abs(b) >= bmin) * (dec >= decmin) *

    lsr_info = (8.122, (12.9, 245.6, 7.78), 0.0208)

    return (l[filt], b[filt], None, dist[filt], Gapp[filt], pml[filt], pmb[filt],
            vlos[filt], PMerr[filt], disterr[filt], vloserr[filt], None, None, lsr_info)


def load_RRL_as_BHBs(SUBSAMPLE):
    rrls = pd.read_csv("data/gmedinat/DESI-iron_RRLs_v0.2_qualcut.csv")
    print('Total number of particles in dataset:', len(rrls))

    ra = rrls['ra'].to_numpy()  # deg
    dec = rrls['dec'].to_numpy()  # deg
    ra  *= d2r  # ra and dec to rad
    dec *= d2r

    pmra = rrls['pmra'].to_numpy()  # mas/yr
    pmdec = rrls['pmdec'].to_numpy()  # mas/yr
    pmra_error = rrls['pmra_error'].to_numpy()  # mas/yr
    pmdec_error = rrls['pmdec_error'].to_numpy()  # mas/yr

    PMerr = (pmra_error + pmdec_error) / 2  # mas/yr

    vlos = rrls['v0_mean'].to_numpy()  # km/s
    vloserr = rrls['v0_std'].to_numpy()  # km/s

    Gapp = rrls['int_average_g'].to_numpy()  # mag
    dist = 10**(0.2*(Gapp-0.58)-2)
    disterr = dist * 0.15

    l, b, pml, pmb = agama.transformCelestialCoords(agama.fromICRStoGalactic,
                                                    ra, dec, pmra, pmdec)

    l /= d2r  # back to degrees
    b /= d2r
    dec /= d2r

    filt = (Gapp > Gmin) * (Gapp < Gmax)
    # (abs(b) >= bmin) * (dec >= decmin) *

    lsr_info = (8.122, (12.9, 245.6, 7.78), 0.0208)

    return (l[filt], b[filt], None, dist[filt], Gapp[filt], pml[filt], pmb[filt],
            vlos[filt], PMerr[filt], disterr[filt], vloserr[filt], None, None, lsr_info)
