#!/usr/bin/env bash

if [ -z "$1" ]
then
  echo "Usage: evaluate_scannet_mapping_experiments.sh <MAPPING_EXPERIMENTS_DIR>"
  exit 1
fi

SCANS_DIR=/media/scratch1/albanesg/scans
EXPERIMENTS_DIR=$1

for DIR in `find ${EXPERIMENTS_DIR} -maxdepth 1 -name "scene*"`
do 
  SCAN_ID=$(basename "${DIR}")
  SCAN_GT_DIR=${SCANS_DIR}/${SCAN_ID}
  if [ ! -d ${SCAN_GT_DIR} ]
  then
    echo "Groundtruth data dir ${SCAN_ID} not found. Experiments won't be evaluated"
    continue
  fi
  roslaunch panoptic_mapping_evaluation evaluate_experiments.launch \
    experiments_dir:=${DIR} \
    ground_truth_vertex_labels_file:=${SCAN_GT_DIR}/panoptic_vertex_labels.txt \
    ground_truth_pointcloud_file:=${SCAN_GT_DIR}/${SCAN_ID}_vh_clean_2.ply
done