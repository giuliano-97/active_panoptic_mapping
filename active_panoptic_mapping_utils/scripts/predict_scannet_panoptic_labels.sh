#!/usr/bin/env bash

if [ -z "$1" ]
then
  echo "Usage: predict_scannet_panoptic_labels.sh <SCANNET_SCANS_DIR> [<MODEL_DIR>]"
  exit 1
fi

SCANNET_SCANS_DIR_PATH=$1
MODEL_DIR_PATH=$2

if [[ ! -d $MODEL_DIR_PATH ]]
then
  MODEL_DIR_PATH=$(rospack find active_panoptic_mapping_ros)/models/mask2former_swin_tiny_scannet
fi

echo "Using model: $MODEL_DIR_PATH"

for DIR in $(find ${SCANNET_SCANS_DIR_PATH} -maxdepth 1 -type d -name "scene*")
do
  SCAN_NAME=$(basename $DIR)
  PANO_SEG_DIR=${DIR}/pano_seg
  IMAGES_DIR=${DIR}/color
  if [ -d $PANO_SEG_DIR ]
  then
    NUM_PANO_SEG_IMAGES=$(find $PANO_SEG_DIR -type f -name "*.png" | wc -l)
    NUM_COLOR_IMAGES=$(find $IMAGES_DIR -type f -name "*.jpg" | wc -l)
    if [[ $NUM_PANO_SEG_IMAGES -eq $NUM_COLOR_IMAGES ]]
    then
      echo "Skipping $SCAN_NAME"
      continue
    fi
  fi

  if [ -d $IMAGES_DIR ]
  then
    echo "Generating labels for scene $SCAN_NAME"
    rosrun active_panoptic_mapping_utils predict.py \
      --model-dir $MODEL_DIR_PATH \
      --images-dir $IMAGES_DIR \
      --output-dir $PANO_SEG_DIR
  fi
done