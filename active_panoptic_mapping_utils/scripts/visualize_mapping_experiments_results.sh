#!/usr/bin/env bash

if [ -z "$1" ] || [ -z "$2" ]
then
    echo "Usage: visualize_mapping_experiments_results.sh <EXPERIMENTS_DIR> <SCANNET_SCANS_DIR>"
    exit 1
fi

EXPERIMENTS_DIR=$1
SCANS_DIR=$2

for SCENE in `ls ${EXPERIMENTS_DIR} | grep scene`
do  
    GT_MESH_FILE="${SCANS_DIR}/${SCENE}/${SCENE}_vh_clean_2.ply"
    if [ ! -f ${GT_MESH_FILE} ] 
    then
        echo "Ground truth not found for scene ${SCENE}"
        continue
    fi
    for DIR in `ls ${EXPERIMENTS_DIR}/${SCENE}`
    do
        if [ ! -d "${EXPERIMENTS_DIR}/${SCENE}/${DIR}" ]
        then
            continue
        fi
        PRED_VERTEX_LABELS_FILE="${EXPERIMENTS_DIR}/${SCENE}/${DIR}/final.txt"
        if [ ! -f ${PRED_VERTEX_LABELS_FILE} ]
        then
            echo "${SCENE}/${DIR} has no final.txt" 
            continue
        fi
        rosrun active_panoptic_mapping_utils visualize_vertex_labels.py \
            ${PRED_VERTEX_LABELS_FILE} ${GT_MESH_FILE}
    done
done

