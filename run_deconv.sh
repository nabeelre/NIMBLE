#!/usr/bin/env bash
# This script creates mock datasets from Latte cosmological simulation snapshots (https://ui.adsabs.harvard.edu/abs/2016ApJ...827L..23W/abstract)
# then runs NIMBLE's *forward* modeling Jeans routine on them.
# The resulting figure is stored in figures/ and with default settings matches that
# presented in Rehemtulla+2022.

./download_latte.sh

# The forward modeling routine with deconvolution takes much more time than the
# inverse modeling routine. It can take upwards of 10 minutes even on fast machines
# It makes use of multiprocessing, so it will benefit from being run on machines
# with more CPU cores.
# Therefore, it is reccomended to use option 1 below for automating the forward
# modeling routine at least when getting familiar with NIMBLE.

gals=('m12f' 'm12i' 'm12m')

for gal in "${gals[@]}"; do
  # prepare the raw latte snapshot for use with the Jeans modeling routine
  python3 read_latte.py $gal
done

# Option 1: enter run configurations into this array, all will be run sequentially
# each element in this list should have the latte galaxy name, the desired local
# standard of rest number, and the gaia data release in that order, separated
# by spaces
configs=('m12f lsr0 dr3' 'm12i lsr2 dr5')
for config in "${configs[@]}"; do
  python3 deconv.py $config
done

# Option 2: Forward modeling routine with deconvolution will be run on all
# combinations of provided Latte galaxies (gals), LSRs (lsrs), and Gaia DRs (drs)
# lsrs=('lsr0') # 'lsr1' 'lsr2')
# drs=('dr3') # 'dr5')
#
# for gal in "${gals[@]}"; do
#   for lsr in "${lsrs[@]}"; do
#     for dr in "${drs[@]}"; do
#       # Run the NIMBLE forward modeling jeans routine
#       python3 deconv.py ${gal} ${lsr} ${dr}
#     done
#   done
# done

# PLOT
