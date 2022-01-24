#!/usr/bin/env bash
# This script creates mock datasets from the Latte suite of FIRE-2 cosmological simulations (https://ui.adsabs.harvard.edu/abs/2016ApJ...827L..23W/abstract)
# then runs NIMBLE's *forward* modeling Jeans routine on them.
# The resulting figure is stored in figures/ and with default settings matching that
# presented in Rehemtulla+2022.

./download_latte.sh

# The forward modeling routine with deconvolution takes much more time than the
# inverse modeling routine. It can take upwards of 10 minutes even on fast machines
# It makes use of multiprocessing, so it will benefit from being run on machines
# with more CPU cores.
# I reccomend to install tqdm to monitor progress of the emcee run

# Make sure to make the list of sims and lsrs here match those in fig7.py (sims_to_plot, lsrs_to_plot)
sims=('m12f') #('m12f' 'm12i' 'm12m')
lsrs=('lsr0') #('lsr0' 'lsr1' 'lsr2')
drs=('dr3') #('dr3' 'dr5')

for sim in "${sims[@]}"; do
  # prepare the raw latte snapshot for use with the Jeans modeling routine
  python3 read_latte.py $sim
done

# Forward modeling routine with deconvolution will be run on all combinations
# of provided Latte galaxies (sims), LSRs (lsrs), and Gaia DRs (drs)
for sim in "${sims[@]}"; do
  for lsr in "${lsrs[@]}"; do
    for dr in "${drs[@]}"; do
      # Run the NIMBLE forward modeling jeans routine
      python3 deconv.py ${sim} ${lsr} ${dr}
    done
  done
done

for dr in "${drs[@]}"; do
  python3 figures/fig7.py ${dr}
done
