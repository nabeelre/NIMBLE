# README

NIMBLE (**N**on-parametr**I**c jeans **M**odeling with **B**-sp**L**in**E**s) is a tool for inferring the cumulative mass distribution of a gravitating system from full 6D phase space coordinates of its tracers via spherical Jeans modeling. Spherical Jeans modeling inherently assumes the system is spherically symmetric and in dynamical equilibrium, however, Rehemtulla+2022 show that these conditions are not completely necessary for an accurate mass profile estimate when applied mock Milky Way-like galaxies. Rehemtulla+2022 gives much more detail on the routines included here and extensive tests using them.  

NIMBLE also includes codes for performing related tasks:

- Creating a variety of equilibrium mock galaxies using the Agama library (https://github.com/GalacticDynamics-Oxford/Agama)
- Creating mock Gaia & DESI observational data of the Latte cosmological hydrodynamic zoom in simulations (Wetzel+2016, https://ui.adsabs.harvard.edu/abs/2016ApJ...827L..23W)

### Testing with `halo alone` datasets

NIMBLE includes a bash script titled `run_halo_alone.sh` which will generate, run Jeans modeling on, and plot a figure for `halo alone` type datasets of up to 3 provided axis ratios.

The `halo alone` datasets are generated using Agama (https://github.com/GalacticDynamics-Oxford/Agama) through `halo_alone.py` located in `equilibrium_models/`. For example, to create a `halo alone` dataset with q=0.8 run the following:

```bash
python3 equilibrium_models/halo_alone.py 0.8
```

This will create `halo_alone_0.8_prejeans.csv` and `halo_alone_0.8_true.csv` in `data/halo_alone`, which contain kinematic information of an N-body representation of this model. These files are used in the following step where the NIMBLE inverse modeling Jeans routine is executed on the dataset using `jeans_bspline.py`. To run it, provide file paths to the `_prejeans.csv`  and `_true.csv` files created in the previous step.

```bash
python3 jeans_bspline.py data/halo_alone/halo_alone_0.8_prejeans.csv data/halo_alone/halo_alone_0.8_true.csv
```

This will create an assortment of files in `results/halo_alone_0.8/` including plots of the velocity, density, velocity anisotropy, and mass enclosed profiles. To create a figure comparing the results of multiple `halo alone` runs, similar to Fig. 3 in Rehemtulla+2022, run `fig3-5.py` located in `figures/` with the argument `halo_alone`.

### Testing with `halo_disk_bulge` datasets

Running `halo_disk_bulge` is very similar to running `halo_alone`. The `run_halo_disk_bulge.sh` script will generate, run Jeans modeling on, and plot a figure for the original dataset and its two variants (described in Sec. 3.1 of Rehemtulla+2022). 

To generate these datasets manually, use `halo_disk_bulge.py` in `equilibrium_models/` as follows:

```bash
python3 equilibrium_models/halo_disk_bulge.py OM
```

In this example I've optionally added `OM` which creates the variant with a Cuddeford-Osipkov-Meritt velocity anisotropy profile. Omit `OM` and you'll generate the original HDB dataset alongside its disk contamination variant, again described in Sec. 3.1 of Rehemtulla+2022.

The process of running NIMBLE's inverse modeling Jeans routine on these and plotting a comparison figure is done identically for these and the `halo_alone` datasets. Do note that the disk contamination variants share a `_true.csv` file with their non-disk contamination parents so there will be no `HDB_DC_true.csv` file.

### Testing with Latte FIRE-2 galaxies (without observational effects)

Running with the inverse modeling routine on error-free Latte data comes with the additional complication of having to download the Latte data. `run_latte_errorfree.sh` explains what data is needed and where it should be downloaded from ([yt Hub](https://girder.hub.yt/#collection/5b0427b2e9914800018237da/folder/5b211e42323d120001c7a813)).  Once the data is downloaded for each Latte galaxy of interest, `run_latte_errorfree.sh` will prepare the mocks, run the inverse modeling Jeans routine on them, and plot a figure of the resulting mass profiles. 

As usual, all these steps can be performed manually with the individual scripts if desired.

```bash
python3 read_latte.py m12f
```

Running the above command will create the `_prejeans.csv` and `_true.csv` files for m12f used for Jeans modeling. This will also require the `gizmo_read` package, also available at [yt Hub](https://girder.hub.yt/#collection/5b0427b2e9914800018237da/folder/5b211e42323d120001c7a813). You can then run the inverse modeling Jeans routine in `jeans_bspline.py` similarly to `halo_alone` datasets by providing the paths to these two `.csv` files. The Jeans routine will automatically select the knot configurations shown in Rehemtulla+2022 but can still be configured to use custom knots.  

### Testing with Latte FIRE-2 galaxies (with observational effects and the deconvolution subroutine)

Documentation WIP
