#!/usr/bin/env bash
# This script generates halo disk bulge datasets then runs NIMBLE's inverse
# modeling Jeans routine on them.
# The resulting figure is stored in figures/ and with default settings matches that
# presented in Rehemtulla+2022.

# If you change these, make corresponding changes in the halo_disk_bulge branch of fig3-5.py
cmd_line_args=('' 'OM')
dataset_names=('' '_DC' '_OM')

for arg in "${cmd_line_args[@]}"; do
  # Generate the halo disk bulge datasets with agama
  python3 equilibrium_models/halo_disk_bulge.py $arg
done

for name in "${dataset_names[@]}"; do
  # Special handling for the _DC models because they share a _true.csv file with their non _DC counterparts
  truename=${name%"_DC"}
  # Run the NIMBLE inverse modeling jeans routine
  python3 jeans_bspline.py data/halo_disk_bulge/HDB${name}_prejeans.csv data/halo_disk_bulge/HDB${truename}_true.csv
done

# Create figure in the style of Fig. 4 from Rehemtulla+2022
python3 figures/fig3-5.py halo_disk_bulge
