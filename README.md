# README

NIMBLE (**N**on-parametr**I**c jeans **M**odeling with **B**-sp**L**in**E**s) is a tool for inferring the cumulative mass distribution of a gravitating system from full 6D phase space coordinates of its tracers via spherical Jeans modeling. Spherical Jeans modeling inherently assumes the system is spherically symmetric and in dynamical equilibrium, however, Rehemtulla+2022 show that these conditions are not completely necessary for an accurate mass profile estimate when applied mock Milky Way-like galaxies. Rehemtulla+2022 gives much more detail on the routines included here and extensive tests using them.  

NIMBLE also includes codes for performing related tasks:

- Creating a variety of equilibrium mock galaxies using the Agama library (https://github.com/GalacticDynamics-Oxford/Agama)
- Creating mock Gaia & DESI observational data of the Latte cosmological hydrodynamic zoom in simulations (Wetzel+2016, https://ui.adsabs.harvard.edu/abs/2016ApJ...827L..23W)

### Testing with ```halo alone``` datasets

NIMBLE includes a bash script titled ```run_halo_alone.sh``` which will generate, run Jeans modeling on, and plot a figure for ```halo alone``` type datasets of up to 3 provided axis ratios.

The ```halo alone``` datasets are generated using Agama (https://github.com/GalacticDynamics-Oxford/Agama) through ```halo_alone.py``` located in ```equilibrium_models/```. For example, to create a `halo alone` dataset with q=0.8 run the following:

```bash
python3 equilibrium_models/halo_alone.py 0.8
```

This will create ```halo_alone_0.8_prejeans.csv``` and ```halo_alone_0.8_true.csv``` in ```data/halo_alone```, which contain kinematic information of an N-body representation of this model. These files are used in the following step where the NIMBLE inverse modeling Jeans routine is executed on the dataset using ```jeans_bspline.py```. To run it, provide file paths to the ```_prejeans.csv```  and ```_true.csv``` files created in the previous step.

```bash
python3 jeans_bspline.py data/halo_alone/halo_alone_0.8_prejeans.csv data/halo_alone/halo_alone_0.8_true.csv
```

This will create an assortment of files in ```results/halo_alone_0.8/``` including plots of the velocity, density, velocity anisotropy, and mass enclosed profiles. To create a figure comparing the results of multiple ```halo alone``` runs, similar to Fig. 3 in Rehemtulla+2022, run ```fig3.py``` located in ```figures/```.

### Testing with ```halo_disk_bulge``` datasets

Documentation WIP

### Testing with Latte FIRE-2 galaxies (without observational effects)

Documentation WIP

### Testing with Latte FIRE-2 galaxies (with observational effects and the deconvolution subroutine)

Documentation WIP
