#!/usr/bin/env bash

HELP_MSG="Usage: evaluate_scannet_mapping_experiments.sh <MAPPING_EXPERIMENTS_DIR> <SCANNET_SCANS_DIR>"

if [ -z "$1" ] || [ -z "$2" ]
then
  echo $HELP_MSG
  exit 1
fi

EXPERIMENTS_DIR=$1
SCANS_DIR=$2

for DIR in `find ${EXPERIMENTS_DIR} -maxdepth 1 -name "scene*"`
do 
  SCAN_ID=$(basename "${DIR}")
  SCAN_GT_DIR=${SCANS_DIR}/${SCAN_ID}
  if [ ! -d ${SCAN_GT_DIR} ]
  then
    echo "Groundtruth data dir ${SCAN_ID} not found. Experiments won't be evaluated"
    continue
  fi
  roslaunch active_panoptic_mapping_utils evaluate_experiments.launch \
    experiments_dir:=${DIR} \
    ground_truth_vertex_labels_file:=${SCAN_GT_DIR}/panoptic_vertex_labels.txt \
    ground_truth_pointcloud_file:=${SCAN_GT_DIR}/${SCAN_ID}_vh_clean_2.ply
done