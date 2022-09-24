#!/usr/bin/env bash

cd "$(dirname "$(realpath "$0")")";

if [ -z "$1" ]
then
    echo "Usage: extract_scans_data.sh <SCANNET_SCANS_DIR>"
    exit 1
fi

SCANS_DIR_PATH=$1

for SENS_FILE in $(find ${SCANS_DIR_PATH} -type f -name "*.sens")
do  
    SCAN_DIR="$(dirname "$SENS_FILE")"
    SCAN_NAME=$(basename "$SCAN_DIR")
    echo "Extracting scan $SCAN_NAME"
    python3 extract_scan_data.py \
        --filename $SENS_FILE \
        --output_path $SCAN_DIR \
        --export_color \
        --export_depth \
        --export_pose \
        --export_intrinsic \
        --image_size 640 480
done