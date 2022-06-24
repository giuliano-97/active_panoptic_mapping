#!/usr/bin/env bash

if [ -z $1 ]
then
  echo "Usage: run_experiments.sh <EXPERIMENTS_DIR>"
  exit 1
fi 

EXPERIMENTS_DIR=$1

mkdir -p $EXPERIMENTS_DIR

for PLANNER in "uncertainty_weighted_voxel_weight"  "reconstruction" "uncertainty_weighted_tsdf_entropy" "panoptic_uncertainty" "exploration" 
do
  CUR_EXPERIMENT_DIR="${EXPERIMENTS_DIR}/${PLANNER}"
  mkdir -p ${CUR_EXPERIMENT_DIR}
  export ROS_LOG_DIR="${CUR_EXPERIMENT_DIR}/logs"
  mkdir -p ${ROS_LOG_DIR}
  roslaunch active_panoptic_mapping_utils run_experiment.launch \
    experiment_name:="${PLANNER}" planner_config:="${PLANNER}" n_reps:=1
done
