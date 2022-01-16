#!/usr/bin/env bash
# This script creates mock datasets from Latte cosmological simulation snapshots (https://ui.adsabs.harvard.edu/abs/2016ApJ...827L..23W/abstract)
# then runs NIMBLE's inverse modeling Jeans routine on them.
# The resulting figure is stored in figures/ and with default settings matches that
# presented in Rehemtulla+2022.

./download_latte

# Latte galaxies to model
# if you change these makes the corresponding changes in the latte branch of fig3-5.py
gals=('m12f' 'm12i' 'm12m')

for gal in "${gals[@]}"; do
  # prepare the raw latte snapshot for use with the Jeans modeling routine
  python3 read_latte.py $gal
  # Run the NIMBLE inverse modeling jeans routine
  python3 jeans_bspline.py data/${gal}/${gal}_prejeans.csv data/${gal}/${gal}_true.csv
done

# Create figure in the style of Fig. 3 in Rehemtulla+2022
python3 figures/fig3-5.py latte
