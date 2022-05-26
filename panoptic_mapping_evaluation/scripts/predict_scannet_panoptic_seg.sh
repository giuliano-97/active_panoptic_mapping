#!/usr/bin/env bash

SCANNET_SCANS_DIR_PATH=/media/scratch1/albanesg/scans

for DIR in $(find ${SCANNET_SCANS_DIR_PATH} -maxdepth 1 -type d -name scene\*)
do
  echo "Generating labels for scene $(basename "$DIR")"
  rosrun panoptic_segmentation predict.py \
    --model-dir /home/albanesg/robostack_catkin_ws/src/active_panoptic_mapping/active_panoptic_mapping/models/mask2former_swin_tiny_scannet \
    --images-dir ${DIR}/color \
    --output-dir ${DIR}/mask2former_swin_tiny
done