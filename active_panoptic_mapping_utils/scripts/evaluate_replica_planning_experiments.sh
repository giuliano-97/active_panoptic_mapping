#!/usr/bin/env bash

if [ -z "$1" ]
then
  echo "Usage: evaluate_replica_planning_experiments.sh <PLANNING_EXPERIMENTS_DIR>"
  exit 1
fi

REPLICA_SCENE_DIR=/media/scratch1/albanesg/replica/frl_apartment_0
EXPERIMENTS_DIR=$1

for DIR in `find ${EXPERIMENTS_DIR} -maxdepth 1`
do 
  EXPERIMENT_NAME=$(basename "${DIR}")
  RUN_DIRS_NUM=$(find ${DIR} -maxdepth 1 -name "run_*" | wc -l)
  if [ "$RUN_DIRS_NUM" -eq "0" ]
  then
    continue
  fi

  echo "Evaluating ${EXPERIMENT_NAME}."
  if [ -f "${DIR}/metrics.csv" ]
  then
    echo "${EXPERIMENT_NAME} has already been evaluated. Skipped."
    continue
  fi
  
  if [ ! -d ${SCAN_GT_DIR} ]
  then
    echo "Groundtruth data dir ${SCAN_ID} not found. Experiments won't be evaluated"
    continue
  fi
  roslaunch panoptic_mapping_evaluation evaluate_experiments.launch \
    experiments_dir:=${DIR} \
    experiment_type:="planning" \
    ground_truth_vertex_labels_file:=${REPLICA_SCENE_DIR}/habitat/panoptic_vertex_labels.txt \
    ground_truth_pointcloud_file:=${REPLICA_SCENE_DIR}/habitat/mesh_semantic_decimated.ply
done
