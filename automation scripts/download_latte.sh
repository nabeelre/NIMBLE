#!/usr/bin/env bash

# The Latte data is quite large (upwards of 10 GB each), downloading will take some time
# Comment out the code blocks of the Latte galaxies you don't want to download
# You can manually download them from https://girder.hub.yt/#collection/5b0427b2e9914800018237da/folder/5b211e42323d120001c7a813
# Make sure to organize the files as NIMBLE expects. Taking m12f as an example:
  # ../data/m12f/snapdir_600/ contains the raw .hd5f files
  # ../data/m12f/ contains m12f_res7100_center.txt
  # other files are not needed

  mkdir ../data
  # Download m12f snapshot
  if [ ! -d "../data/m12f" ]
  then
    mkdir ../data/m12f # create directory for m12f
    echo Downloading m12f snapshot, total size ~9000M
    curl --output ../data/m12f/snapdir_600.zip -X GET --header "Accept: application/zip" "https://girder.hub.yt/api/v1/folder/5b211e5a323d120001c7a828/download" # Use yt Hub's API to download the snapshot
    unzip ../data/m12f/snapdir_600.zip -d ../data/m12f/ # Unzip the snapshot
    if [ ! -d "../data/m12f/snapdir_600" ]
    then
      echo "Failed to unzip snapshot"
      exit
    fi
    rm ../data/m12f/snapdir_600.zip # Delete the zipped file
    curl --output ../data/m12f/m12f_res7100_center.txt -X GET --header 'Accept: text/plain' 'https://girder.hub.yt/api/v1/file/5b330ba8323d120001bfe384/download?contentDisposition=attachment'
  fi

  Download m12i snapshot
  if [ ! -d "../data/m12i" ]
  then
    mkdir ../data/m12i
    echo Downloading m12i snapshot, total size ~7200M
    curl --output ../data/m12i/snapdir_600.zip -X GET --header 'Accept: application/zip' 'https://girder.hub.yt/api/v1/folder/5b211e5a323d120001c7a819/download'
    unzip ../data/m12i/snapdir_600.zip -d ../data/m12i/
    if [ ! -d "../data/m12i/snapdir_600" ]
    then
      echo "Failed to unzip snapshot"
      exit
    fi
    rm ../data/m12i/snapdir_600.zip
    curl --output ../data/m12i/m12i_res7100_center.txt -X GET --header 'Accept: text/plain' 'https://girder.hub.yt/api/v1/file/5b330bca323d120001bfe390/download?contentDisposition=attachment'
  fi

  # Download m12m snapshot
  if [ ! -d "../data/m12m" ]
  then
    mkdir ../data/m12m
    echo Downloading m12m snapshot, total size ~16000M
    curl --output ../data/m12m/snapdir_600.zip -X GET --header 'Accept: application/zip' 'https://girder.hub.yt/api/v1/folder/5b211e5a323d120001c7a839/download'
    unzip ../data/m12m/snapdir_600.zip -d ../data/m12m/
    if [ ! -d "../data/m12m/snapdir_600" ]
    then
      echo "Failed to unzip snapshot"
      exit
    fi
    rm ../data/m12m/snapdir_600.zip
    curl --output ../data/m12m/m12m_res7100_center.txt -X GET --header 'Accept: text/plain' 'https://girder.hub.yt/api/v1/file/5b330bfe323d120001bfe39c/download?contentDisposition=attachment'
  fi

  # Download the gizmo_read library
  # If you'd like to download it manually you can do so from:
  # https://girder.hub.yt/#collection/5b0427b2e9914800018237da/folder/5b211e42323d120001c7a813
  if [ ! -d "gizmo_read" ]
  then
    curl --output gizmo_read.zip -X GET --header 'Accept: application/zip' 'https://girder.hub.yt/api/v1/folder/5b332e25323d120001bfe44c/download'
    unzip gizmo_read.zip
    rm gizmo_read.zip
  fi
