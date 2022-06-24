#!/usr/bin/env bash

cd "$(dirname "$(realpath "$0")")";

if [ -z "$1" ]
then
    echo "Usage: extract_scans_data.sh <SCANNET_SCANS_DIR>"
    exit 1
fi

SCANS_DIR_PATH=$1

for SENS_FILE in $(find ${SCAN_DIR_PATH} -type f -name *.sens)
do  
    SCAN_DIR="$(dirname "$SENS_FILE")"
    python3 extract_scans_data \
        --filename $SENS_FILE \
        --output_path $SCAN_DIR \
        --export_color \
        --export_depth \
        --export_pose \
        --export_intrinsic \
        --image-size 640 480
done