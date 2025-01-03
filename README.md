# NIMBLE (**N**on-parametr**I**c jeans **M**odeling with **B**-sp**L**in**E**s)

NIMBLE is a tool for inferring the cumulative mass profile of a gravitating system (the Milky Way) from full 6D phase space coordinates of its tracers (field halo RR Lyrae stars) via spherical Jeans modeling. NIMBLE uses B-splines to non-parametrically fit the velocity and density profiles used in evaluating the Jeans equation. Spherical Jeans modeling assumes the system is spherically symmetric and in dynamical equilibrium, but [Rehemtulla et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022MNRAS.511.5536R/abstract) demonstrate that these conditions are not necessary for an accurate mass profile estimate when applied mock Milky Way-like galaxies.



### Installation

Clone the NIMBLE repository

```bash
https://github.com/nabeelre/NIMBLE.git
```

Create an environment with the necessary dependencies. This may take ~5 minutes and you will be prompted to install additional components during the process.

```bash
conda env create --file environment.yaml
```

Activate the environment

```bash
conda activate nimble
```


Mock datasets can be downloaded from their respective repositories - or contact nabeelr at u.northwestern.edu for NIMBLE-ready catalogs.



### Running `deconv.py`

The `deconv.py` script should be used for RR Lyrae datasets which are spatially incomplete and have observational errors imposed on the measurements, i.e. real data or realistic mock data. `deconv.py` supports running with three different types of datasets: mock data from the Latte simulations ([Wetzel et al. 2016](https://ui.adsabs.harvard.edu/abs/2016ApJ...827L..23W/abstract)), mock data from the AuriDESI simulations ([Kizhuprakkat et al. 2024](https://ui.adsabs.harvard.edu/abs/2024MNRAS.531.4108K/abstract)), and real DESI data.



The command line arguments tell `deconv.py` which dataset and parameters to use while running. For mock data, you need to specify which mock to run on, and you can always optionally overwrite the default knot configuration. Ordering of the parameters must match that shown here.

#### Latte mocks

Running `deconv.py` on the Latte m12i galaxy at LSR0 with Gaia DR3-like uncertainties with 6 knots ranging from 5 kpc to 50 kpc

```
python deconv.py latte m12i lsr0 dr3 5 50 6
```

In general, `python deconv.py latte {halo_name} {LSR_index} {gaia_DR} {min_knot} {max_knot} {num_knots}`




#### AuriDESI mocks

Running `deconv.py` on the AuriDESI 06 halo at the 030 degrees LSR with 5 knots ranging from 10 kpc to 60 kpc

```
python deconv.py auridesi 06 030 10 60 5
```

In general, `python deconv.py auridesi {halo_name} {LSR_degrees} {min_knot} {max_knot} {num_knots}`




#### DESI data

Running `deconv.py` on the DESI internal iron release with 4 knots ranging from 5 kpc to 50 kpc

```
python deconv.py iron 5 50 4
```

In general, `python deconv.py latte {min_knot} {max_knot} {num_knots}`



### `deconv_BHB.py`

`deconv_BHB.py` is a new script in development to make NIMBLE support BHB stars



### Citing NIMBLE

```
@ARTICLE{Rehemtulla+2022,
       author = {{Rehemtulla}, Nabeel and {Valluri}, Monica and {Vasiliev}, Eugene},
        title = "{Non-parametric spherical Jeans mass estimation with B-splines}",
      journal = {\mnras},
     keywords = {galaxies: haloes, galaxies: kinematics and dynamics, galaxies: structure, Astrophysics - Astrophysics of Galaxies},
         year = 2022,
        month = apr,
       volume = {511},
       number = {4},
        pages = {5536-5549},
          doi = {10.1093/mnras/stac400},
archivePrefix = {arXiv},
       eprint = {2202.05440},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022MNRAS.511.5536R},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

