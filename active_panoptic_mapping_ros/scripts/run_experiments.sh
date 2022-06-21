#!/usr/bin/env bash

EXPERIMENTS_DIR=/media/scratch1/albanesg/planning_experiments/mask2former

mkdir -p $EXPERIMENTS_DIR

for PLANNER in "uncertainty_weighted_voxel_weight"  "reconstruction" # "uncertainty_weighted_tsdf_entropy" "panoptic_uncertainty"   "exploration" "panoptic_uncertainty"
do
  CUR_EXPERIMENT_DIR="${EXPERIMENTS_DIR}/${PLANNER}"
  mkdir -p ${CUR_EXPERIMENT_DIR}
  export ROS_LOG_DIR="${CUR_EXPERIMENT_DIR}/logs"
  mkdir -p ${ROS_LOG_DIR}
  roslaunch active_panoptic_mapping_ros run_experiment.launch \
    experiment_name:="${PLANNER}" planner_config:="${PLANNER}" n_reps:=1
done
