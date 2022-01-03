#!/usr/bin/env bash
# This script generates halo alone datasets with axis ratio q=1.0, 0.8, and 0.6
# then runs NIMBLE's inverse modeling Jeans routine on them.
# The resulting figure is stored in figures/ and with default settings matches that
# presented in Rehemtulla+2022.

# This script can also be modified to make and model halo alone datasets with
# other axis ratios not shown in Rehemtulla+2022 including q=0.9, and q=0.7

# Axis ratios to model
qs=('0.6' '0.8' '1.0')

for q in "${qs[@]}"; do
  # Generate the halo alone datasets with agama
  python3 equilibrium_models/halo_alone.py $q
  # Run the NIMBLE inverse modeling jeans routine
  python3 jeans_bspline.py data/halo_alone/halo_alone_${q}_prejeans.csv data/halo_alone/halo_alone_${q}_true.csv
done

# Create figure in the style of Fig. 3 in Rehemtulla+2022
python3 figures/fig3.py
