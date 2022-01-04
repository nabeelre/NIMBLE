#!/usr/bin/env bash
# This script creates mock datasets from Latte cosmological simulation snapshots (https://ui.adsabs.harvard.edu/abs/2016ApJ...827L..23W/abstract)
# then runs NIMBLE's inverse modeling Jeans routine on them.
# The resulting figure is stored in figures/ and with default settings matches that
# presented in Rehemtulla+2022.

# *** PREREQUISITES ***
# Download the Latte z=0 simulation snapshots for each galaxy you'd like to run and gizmo_read
# https://girder.hub.yt/#collection/5b0427b2e9914800018237da/folder/5b211e42323d120001c7a813
# snapdir_600/ and _res7100_center.txt are required and other files are not needed
# Place the folder for each galaxy in data/
  # Taking m12f as an example, you would have:
  # data/m12f/snapdir_600/ and data/m12f/m12f_res7100_center.txt
# And place gizmo_read in NIMBLE's home directory
# *********************

# Latte galaxies to model
# if you change these makes the corresponding changes in the latte branch of fig3-5.py
gals=('m12m' 'm12i' 'm12f')

for gal in "${gals[@]}"; do
  # prepare the raw latte snapshot for use with the Jeans modeling routine
  python3 read_latte.py $gal
  # Run the NIMBLE inverse modeling jeans routine
  python3 jeans_bspline.py data/${gal}/${gal}_prejeans.csv data/${gal}/${gal}_true.csv
done

# Create figure in the style of Fig. 3 in Rehemtulla+2022
python3 figures/fig3-5.py latte
