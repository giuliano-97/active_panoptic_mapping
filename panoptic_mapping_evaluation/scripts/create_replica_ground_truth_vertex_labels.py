#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from plyfile import PlyData
from tqdm import tqdm


def create_replica_ground_truth_vertex_labels(
    replica_dir_path: Path, use_decimated_mesh: bool
):
    if not replica_dir_path.is_dir():
        raise FileNotFoundError(f"{str(replica_dir_path)} is not a valid directory!")

    mesh_semantic_file_name = (
        "mesh_semantic_decimated.ply" if use_decimated_mesh else "mesh_semantic.ply"
    )

    scene_dirs = []
    for mesh_semantic_file_path in replica_dir_path.glob(
        f"**/{mesh_semantic_file_name}"
    ):
        if mesh_semantic_file_path.parent.name == "habitat":
            scene_dirs.append(mesh_semantic_file_path.parents[1])

    for scene_dir_path in tqdm(scene_dirs):
        print(f"Creating labels for scene {scene_dir_path.name}")

        mesh_semantic_file_path = scene_dir_path / "habitat" / mesh_semantic_file_name
        info_semantic_file_path = scene_dir_path / "habitat" / "info_semantic.json"

        print("Loading semantic mesh")
        mesh_semantic = PlyData.read(str(mesh_semantic_file_path))
        face_object_ids = np.array(mesh_semantic["face"].data["object_id"])
        face_vertex_indices = mesh_semantic["face"].data["vertex_indices"]

        print("Computing vertex to object ownership.")
        object_ids_to_vertex_indices = defaultdict(set)
        for object_id, vertex_indices in zip(face_object_ids, face_vertex_indices):
            for index in vertex_indices:
                object_ids_to_vertex_indices[object_id].add(index)

        print("Masking ambiguous vertices.")
        # Find vertices that belong to multiple objects
        vertex_sets = list(object_ids_to_vertex_indices.values())
        ambiguous_vertices = set()
        for i in range(len(vertex_sets)):
            s_i = vertex_sets[i]
            for j in range(i + 1, len(vertex_sets)):
                s_j = vertex_sets[j]
                s_i_j = s_i.intersection(s_j)
                if len(s_i_j) > 0:
                    ambiguous_vertices = ambiguous_vertices.union(s_i_j)

        print("Creating panoptic vertex labels.")
        with info_semantic_file_path.open("r") as f:
            info_semantic = json.load(f)

        object_id_to_class_nyu40 = dict()
        for object_info in info_semantic["objects"]:
            class_id_nyu40 = REPLICA_TO_NYU40.get(object_info["class_id"], 0)
            object_id_to_class_nyu40[object_info["id"]] = class_id_nyu40

        vertex_panoptic_labels = np.zeros(
            (mesh_semantic["vertex"].count,), dtype=np.int32
        )
        for object_id, vertex_set in object_ids_to_vertex_indices.items():
            # Remove ambiguous vertices
            vertex_set_unique = vertex_set.difference(ambiguous_vertices)
            if object_id == 0 or object_id not in object_id_to_class_nyu40:
                continue
            label = object_id_to_class_nyu40[int(object_id)]
            if label <= 0 or label > 40:
                continue
            if label in [1, 2, 22]:
                panoptic_label = label * 1000
            else:
                panoptic_label = label * 1000 + object_id
            np.put(vertex_panoptic_labels, list(vertex_set_unique), int(panoptic_label))

        print("Saving result to file")
        vertex_panoptic_labels_file_path = (
            scene_dir_path / "habitat" / "panoptic_vertex_labels.txt"
        )
        with open(vertex_panoptic_labels_file_path, "w") as f:
            np.savetxt(f, vertex_panoptic_labels, fmt="%d")

        print("Done")


def _parse_args():

    parser = argparse.ArgumentParser(
        description="Create ground truth panoptic labels"
        "for Replica scenes for evaluation."
    )

    parser.add_argument(
        "replica_dir",
        type=lambda p: Path(p).absolute(),
        help="Path to the directory containing the replica dataset.",
    )

    parser.add_argument(
        "--use-decimated-mesh",
        action="store_true",
        help="Whether the original or decimated semantic mesh should be used.",
    )

    return parser.parse_args()


REPLICA_TO_NYU40 = {
    1: 37,
    2: 40,
    3: 40,
    4: 39,
    5: 38,
    6: 39,
    7: 40,
    8: 39,
    9: 40,
    10: 40,
    11: 40,
    12: 13,
    13: 23,
    14: 40,
    15: 29,
    16: 40,
    17: 40,
    18: 3,
    19: 40,
    20: 5,
    21: 40,
    22: 40,
    23: 21,
    24: 21,
    25: 40,
    26: 40,
    27: 40,
    28: 40,
    29: 18,
    30: 16,
    31: 21,
    32: 12,
    33: 12,
    34: 14,
    35: 40,
    36: 40,
    37: 8,
    38: 40,
    39: 34,
    40: 2,
    41: 3,
    42: 40,
    43: 40,
    44: 40,
    45: 40,
    46: 40,
    47: 35,
    48: 40,
    49: 39,
    50: 20,
    51: 40,
    52: 40,
    53: 40,
    54: 32,
    55: 40,
    56: 40,
    57: 40,
    58: 40,
    59: 11,
    60: 38,
    61: 18,
    62: 38,
    63: 39,
    64: 40,
    65: 40,
    66: 39,
    67: 24,
    68: 40,
    69: 40,
    70: 40,
    71: 10,
    72: 21,
    73: 39,
    74: 34,
    75: 40,
    76: 6,
    77: 38,
    78: 39,
    79: 40,
    80: 7,
    81: 40,
    82: 40,
    83: 26,
    84: 33,
    85: 40,
    86: 27,
    87: 25,
    88: 39,
    89: 40,
    90: 40,
    91: 40,
    92: 38,
    93: 1,
    94: 3,
    95: 40,
    96: 3,
    97: 9,
    98: 20,
    99: 40,
    100: 37,
    101: 21,
}


if __name__ == "__main__":
    args = _parse_args()
    create_replica_ground_truth_vertex_labels(
        args.replica_dir,
        args.use_decimated_mesh,
    )
