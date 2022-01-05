#!/usr/bin/env bash
# This script creates mock datasets from Latte cosmological simulation snapshots (https://ui.adsabs.harvard.edu/abs/2016ApJ...827L..23W/abstract)
# then runs NIMBLE's inverse modeling Jeans routine on them.
# The resulting figure is stored in figures/ and with default settings matches that
# presented in Rehemtulla+2022.

# The Latte data is quite large (upwards of 10 GB each), downloading will take some time
# Comment out these four blocks if you already have the Latte data and gizmo_read downloaded
# You can manually download them from https://girder.hub.yt/#collection/5b0427b2e9914800018237da/folder/5b211e42323d120001c7a813
# Make sure to organize the files as NIMBLE expects. Taking m12f as an example:
  # data/m12f/snapdir_600/ contains the raw .hd5f files
  # data/m12f/ contains m12f_res7100_center.txt
  # other files are not needed
# Download m12f z=0 snapshot
mkdir data/m12f # create director for m12f
curl --output data/m12f/snapdir_600.zip -X GET --header "Accept: application/zip" "https://girder.hub.yt/api/v1/folder/5b211e5a323d120001c7a828/download" # Use yt Hub's API to download the snapshot
unzip data/m12f/snapdir_600.zip -d data/m12f/ # Unzip the snapshot
rm data/m12f/snapdir_600.zip # Delete the zipped file
curl --output data/m12f/m12f_res7100_center.txt -X GET --header 'Accept: text/plain' 'https://girder.hub.yt/api/v1/file/5b330ba8323d120001bfe384/download?contentDisposition=attachment'

# Download m12i z=0 snapshot
mkdir data/m12i
curl --output data/m12i/snapdir_600.zip -X GET --header 'Accept: application/zip' 'https://girder.hub.yt/api/v1/folder/5b211e5a323d120001c7a819/download'
unzip data/m12i/snapdir_600.zip -d data/m12i/
rm data/m12i/snapdir_600.zip
curl --output data/m12i/m12i_res7100_center.txt -X GET --header 'Accept: text/plain' 'https://girder.hub.yt/api/v1/file/5b330bca323d120001bfe390/download?contentDisposition=attachment'

# Download m12m z=0 snapshot
mkdir data/m12m
curl --output data/m12m/snapdir_600.zip -X GET --header 'Accept: application/zip' 'https://girder.hub.yt/api/v1/folder/5b211e5a323d120001c7a839/download'
unzip data/m12m/snapdir_600.zip -d data/m12m/
rm data/m12m/snapdir_600.zip
curl --output data/m12m/m12m_res7100_center.txt -X GET --header 'Accept: text/plain' 'https://girder.hub.yt/api/v1/file/5b330bfe323d120001bfe39c/download?contentDisposition=attachment'

# Download the gizmo_read library
# If you'd like to download it manually you can do so from:
# https://girder.hub.yt/#collection/5b0427b2e9914800018237da/folder/5b211e42323d120001c7a813
curl --output gizmo_read.zip -X GET --header 'Accept: application/zip' 'https://girder.hub.yt/api/v1/folder/5b332e25323d120001bfe44c/download'
unzip gizmo_read.zip
rm gizmo_read.zip

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
