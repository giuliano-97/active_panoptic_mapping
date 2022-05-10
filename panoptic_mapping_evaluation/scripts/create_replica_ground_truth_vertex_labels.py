#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from plyfile import PlyData
from tqdm import tqdm


def create_replica_ground_truth_vertex_labels(
    replica_dir_path: Path,
):
    if not replica_dir_path.is_dir():
        raise FileNotFoundError(f"{str(replica_dir_path)} is not a valid directory!")

    scene_dirs = [
        p
        for p in replica_dir_path.iterdir()
        if p.is_dir() and p.joinpath("habitat").is_dir()
    ]

    for scene_dir_path in tqdm(scene_dirs):
        print(f"Creating labels for scene {scene_dir_path.name}")

        mesh_semantic_file_path = scene_dir_path / "habitat" / "mesh_semantic.ply"
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

        object_id_to_label_nyu40 = [
            REPLICA_TO_NYU40[x] if x in REPLICA_TO_NYU40 else 0
            for x in info_semantic["id_to_label"]
        ]

        vertex_panoptic_labels = np.zeros(
            (mesh_semantic["vertex"].count,), dtype=np.int32
        )
        for object_id, vertex_set in object_ids_to_vertex_indices.items():
            # Remove ambiguous vertices
            vertex_set_unique = vertex_set.difference(ambiguous_vertices)
            if object_id == 0:
                continue
            label = object_id_to_label_nyu40[int(object_id)]
            if label <= 0 or label > 40:
                continue
            if label in [1, 2, 22]:
                panoptic_label = label * 1000
            else:
                panoptic_label = label * 1000 + object_id
            np.put(vertex_panoptic_labels, list(vertex_set_unique), int(panoptic_label))

        print("Saving result to file")
        vertex_panoptic_labels_file_path = (
            scene_dir_path / "habitat" / "vertex_panoptic_labels.txt"
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

    return parser.parse_args()


REPLICA_TO_NYU40 = [
    0,
    37,
    40,
    40,
    39,
    38,
    39,
    40,
    39,
    40,
    40,
    40,
    13,
    23,
    40,
    29,
    40,
    40,
    3,
    40,
    5,
    40,
    40,
    21,
    21,
    40,
    40,
    40,
    40,
    18,
    16,
    21,
    12,
    12,
    14,
    40,
    40,
    8,
    40,
    34,
    2,
    3,
    40,
    40,
    40,
    40,
    40,
    35,
    40,
    39,
    20,
    40,
    40,
    40,
    32,
    40,
    40,
    40,
    40,
    11,
    38,
    18,
    38,
    39,
    40,
    40,
    39,
    24,
    40,
    40,
    40,
    10,
    21,
    39,
    34,
    40,
    6,
    38,
    39,
    40,
    7,
    40,
    40,
    26,
    33,
    40,
    27,
    25,
    39,
    40,
    40,
    40,
    38,
    1,
    3,
    40,
    3,
    9,
    20,
    40,
    37,
    21,
]


if __name__ == "__main__":
    args = _parse_args()
    create_replica_ground_truth_vertex_labels(args.replica_dir)
