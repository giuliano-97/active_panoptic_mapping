#!/usr/bin/env bash

if [ -z "$1" ]
then
  echo "Usage: predict_scannet_panoptic_labels.sh <SCANNET_SCANS_DIR> [<MODEL_DIR>]"
  exit 1
fi

SCANNET_SCANS_DIR_PATH=$1
MODEL_DIR_PATH=$2

if[ ! -d $MODEL_DIR_PATH ]
then
  MODEL_DIR_PATH=$(rospack find active_panoptic_mapping_ros)/models/mask2former_swin_tiny_scannet
fi

for DIR in $(find ${SCANNET_SCANS_DIR_PATH} -maxdepth 1 -type d -name scene\*)
do
  echo "Generating labels for scene $(basename "$DIR")"
  rosrun active_panoptic_mapping_utils predict.py \
    --model-dir ${MODEL_DIR_PATH} \
    --images-dir ${DIR}/color \
    --output-dir ${DIR}/mask2former_swin_tiny
done