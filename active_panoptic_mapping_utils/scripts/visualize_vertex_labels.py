#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from active_panoptic_mapping_utils.constants import (
    PANOPTIC_LABEL_DIVISOR,
    SCANNET_NYU40_EVALUATION_CLASSES,
)
from active_panoptic_mapping_utils.visualization import (
    colorize_panoptic_labels,
    colorize_semantic_labels,
)


def visualize_vertex_labels(
    vertex_labels_file_path: Path,
    gt_mesh_file_path: Path,
    only_panoptic: bool,
    only_scannet_eval: bool,
):
    if not vertex_labels_file_path.is_file():
        raise FileNotFoundError(f"{str(vertex_labels_file_path)} is not a valid file")

    if not gt_mesh_file_path.is_file():
        raise FileNotFoundError(f"{str(gt_mesh_file_path)} is not a valid file")

    # Load the labels
    labels = np.loadtxt(vertex_labels_file_path).astype(np.int64)
    labels = np.where(labels == -1, 0, labels)
    semantic_labels = labels // PANOPTIC_LABEL_DIVISOR

    if only_scannet_eval:
        for l in np.unique(semantic_labels):
            if l not in SCANNET_NYU40_EVALUATION_CLASSES:
                mask = semantic_labels == l
                labels[mask] = 0
                semantic_labels[mask] = 0

    # Load the gt mesh
    gt_mesh = PlyData.read(str(gt_mesh_file_path))

    if len(labels) != gt_mesh["vertex"].count:
        raise ValueError(
            "The number of labels and the number of vertices do not match!"
        )

    colors, _ = colorize_panoptic_labels(labels)

    # Generate vertex colors
    gt_mesh["vertex"].data["red"] = colors[:, 0]
    gt_mesh["vertex"].data["green"] = colors[:, 1]
    gt_mesh["vertex"].data["blue"] = colors[:, 2]

    # Save colorized mesh to disk
    PlyData(
        elements=[
            PlyElement.describe(gt_mesh["vertex"].data, "vertex"),
            PlyElement.describe(gt_mesh["face"].data, "face"),
        ],
        text=False,
    ).write(
        vertex_labels_file_path.parent.joinpath(
            vertex_labels_file_path.name.split(".")[0]
        ).with_suffix(".panoptic_labels.ply")
    )

    if not only_panoptic:
        colors = colorize_semantic_labels(semantic_labels)
        gt_mesh["vertex"].data["red"] = colors[:, 0]
        gt_mesh["vertex"].data["green"] = colors[:, 1]
        gt_mesh["vertex"].data["blue"] = colors[:, 2]

        PlyData(
            elements=[
                PlyElement.describe(gt_mesh["vertex"].data, "vertex"),
                PlyElement.describe(gt_mesh["face"].data, "face"),
            ],
            text=False,
        ).write(        vertex_labels_file_path.parent.joinpath(
            vertex_labels_file_path.name.split(".")[0]
        ).with_suffix(".semantic_labels.ply"))


def _parse_args():

    parser = argparse.ArgumentParser(
        description="Visualize vertex labels on ground truth mesh."
    )

    parser.add_argument(
        "vertex_labels_file",
        type=lambda p: Path(p).absolute(),
        help="Path to the vertex labels file.",
    )

    parser.add_argument(
        "gt_mesh_file",
        type=lambda p: Path(p).absolute(),
        help="Path to the ground truth mesh file",
    )

    parser.add_argument(
        "--only-panoptic",
        action="store_true",
        help="Only panoptic labels will be colorized",
    )

    parser.add_argument(
        "--only-scannet-eval",
        action="store_true",
        help="Only consider scannet eval categories",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    visualize_vertex_labels(
        args.vertex_labels_file,
        args.gt_mesh_file,
        args.only_panoptic,
        args.only_scannet_eval,
    )
