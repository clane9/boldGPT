#!/bin/bash

mkdir data 2>/dev/null

# Download NSD stimuli images.
mkdir -p data/NSD/nsddata_stimuli/stimuli/nsd &&
    aws s3api get-object --no-sign-request \
    --bucket natural-scenes-dataset \
    --key nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 \
    data/NSD/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5

# Download NSD stimuli info. For a description of the columns, see:
# https://cvnlab.slite.page/p/NKalgWd__F/Experiments#bf18f984
mkdir -p data/NSD/nsddata/experiments/nsd &&
    aws s3api get-object --no-sign-request \
    --bucket natural-scenes-dataset \
    --key nsddata/experiments/nsd/nsd_stim_info_merged.csv \
    data/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.csv

# Download COCO annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip && \
    unzip annotations_trainval2017.zip -d data/COCO && \
    rm annotations_trainval2017.zip

# Download individual trial beta maps from the NSD dataset.
# Note that the held out test sessions will fail to download due to `ACCESS DENIED`.
aws s3 sync --no-sign-request \
  s3://natural-scenes-dataset data/NSD \
  --exclude "*" --include "nsddata_betas/ppdata/subj*/fsaverage/betas_fithrf_GLMdenoise_RR/*"
