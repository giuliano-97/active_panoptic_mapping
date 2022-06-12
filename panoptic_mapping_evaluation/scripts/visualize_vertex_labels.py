#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from panoptic_mapping_evaluation.constants import PANOPTIC_LABEL_DIVISOR
from panoptic_mapping_evaluation.visualization import colorize_panoptic_labels


def visualize_vertex_labels(
    vertex_labels_file_path: Path,
    gt_mesh_file_path: Path,
):
    if not vertex_labels_file_path.is_file():
        raise FileNotFoundError(f"{str(vertex_labels_file_path)} is not a valid file")

    if not gt_mesh_file_path.is_file():
        raise FileNotFoundError(f"{str(gt_mesh_file_path)} is not a valid file")

    # Load the labels
    labels = np.loadtxt(vertex_labels_file_path).astype(np.int64)
    labels = np.where(labels == -1, 0, labels)

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
    ).write(vertex_labels_file_path.with_suffix(".semantic.ply"))


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

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    visualize_vertex_labels(
        args.vertex_labels_file,
        args.gt_mesh_file,
    )
