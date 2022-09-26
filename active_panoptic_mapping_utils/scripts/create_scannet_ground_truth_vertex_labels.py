#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
from plyfile import PlyData
from tqdm import tqdm

_SEMANTIC_LABELED_MESH_FILE_TEMPLATE = "{}_vh_clean_2.labels.ply"
_SEGS_FILE_TEMPLATE = "{}_vh_clean_2.0.010000.segs.json"
_AGGREGATION_FILE_TEMPLATE = "{}_vh_clean.aggregation.json"

NYU40_STUFF_CLASSES = [1, 2, 22]
NYU40_THING_CLASSES = [i for i in range(1, 41) if i not in NYU40_STUFF_CLASSES]
NYU40_IGNORE_LABEL = 0
PANOPTIC_LABEL_DIVISOR = 1000


def _is_thing(semantic_id: int):
    return semantic_id not in NYU40_STUFF_CLASSES and semantic_id != NYU40_IGNORE_LABEL


def _get_segments_to_object_id_dict(seg_groups: List):
    segment_to_object_id = dict()
    for group in seg_groups:
        object_id = group["objectId"]
        for seg in group["segments"]:
            segment_to_object_id[seg] = object_id
    return segment_to_object_id


def create_scannet_ground_truth_vertex_labels(
    scans_dir_path: Path,
):
    if not scans_dir_path.is_dir():
        raise FileNotFoundError(f"{scans_dir_path} is not a valid directory")

    scan_dirs = [
        p
        for p in scans_dir_path.iterdir()
        if p.is_dir()
        and p.joinpath(_SEMANTIC_LABELED_MESH_FILE_TEMPLATE.format(p.name)).is_file()
        and p.joinpath(_SEGS_FILE_TEMPLATE.format(p.name)).is_file()
        and p.joinpath(_AGGREGATION_FILE_TEMPLATE.format(p.name)).is_file()
    ]

    for scan_dir_path in tqdm(scan_dirs):
        scene_name = scan_dir_path.stem

        # Load the over-segmentation
        segs_file_path = scan_dir_path / _SEGS_FILE_TEMPLATE.format(scene_name)
        print("Loading instance segmentation info")
        with segs_file_path.open("r") as f:
            segs_dict = json.load(f)
        seg_indices = segs_dict["segIndices"]

        # Load the aggregation file
        aggregation_file_path = scan_dir_path / _AGGREGATION_FILE_TEMPLATE.format(
            scene_name
        )
        with aggregation_file_path.open("r") as f:
            aggregation_dict = json.load(f)
        seg_groups = aggregation_dict["segGroups"]

        # Get mapping from segments to object id
        segments_to_object_id = _get_segments_to_object_id_dict(seg_groups)

        # Load semantic vertex labels
        semantic_mesh_file_path = (
            scan_dir_path / _SEMANTIC_LABELED_MESH_FILE_TEMPLATE.format(scene_name)
        )
        print("Load semantic segmentation mesh")
        semantic_mesh = PlyData.read(str(semantic_mesh_file_path))
        semantic_vertex_labels = np.array(semantic_mesh["vertex"].data["label"])

        # Create panoptic vertex labels
        print("Creating panoptic vertex labels")
        panoptic_vertex_labels = np.zeros_like(semantic_vertex_labels, dtype=np.uint32)
        object_id_to_instance_id = dict()
        next_valid_instance_id = 0
        for idx, semantic_label in enumerate(semantic_vertex_labels):
            panoptic_vertex_labels[idx] = semantic_label * PANOPTIC_LABEL_DIVISOR
            if _is_thing(semantic_label):
                object_id = segments_to_object_id[seg_indices[idx]]
                instance_id = object_id_to_instance_id.get(object_id, None)
                if instance_id is None:
                    # Grab the next valid instance id
                    while (
                        next_valid_instance_id == NYU40_IGNORE_LABEL
                        or next_valid_instance_id in NYU40_STUFF_CLASSES
                    ):
                        next_valid_instance_id += 1
                    instance_id = next_valid_instance_id
                    object_id_to_instance_id[object_id] = instance_id
                    next_valid_instance_id += 1

                # Add 1 because object ids start at 0
                panoptic_vertex_labels[idx] += instance_id

        print("Saving to file.")
        # Save to file
        panoptic_vertex_labels_file = scan_dir_path / "panoptic_vertex_labels.txt"
        with panoptic_vertex_labels_file.open("w") as f:
            np.savetxt(f, panoptic_vertex_labels, fmt="%d")

        print("Done")


def _parse_args():

    parser = argparse.ArgumentParser(
        description="Create ground truth panoptic labels"
        "for ScanNet scans for evaluation."
    )

    parser.add_argument(
        "scans_dir",
        type=lambda p: Path(p).absolute(),
        help="Path to the directory containing ScanNet scans to generate vertex labels for.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    create_scannet_ground_truth_vertex_labels(args.scans_dir)
