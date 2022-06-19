#!/usr/bin/env python3

import argparse
import yaml
from pathlib import Path

import rospkg
import numpy as np
from tqdm import tqdm
from plyfile import PlyData

from habitat_ros.utils import vec_habitat_to_ros


_CONFIG_TEMPLATE = """
initial_position:
  x: 0.0
  y: 0.0
  z: 0.0

map_bounding_volume:
  x_min:  {x_min}
  x_max: {x_max}
  y_min:  {y_min}
  y_max: {y_max}
  z_min:  {z_min}
  z_max: {z_max}

target_bounding_volume:
  x_min:  {x_min}
  x_max: {x_max}
  y_min:  {y_min}
  y_max: {y_max}
  z_min:  {z_min}
  z_max: {z_max}
"""


def create_replica_scene_config(
    replica_dir_path: Path,
):
    if not replica_dir_path.is_dir():
        raise FileNotFoundError(f"{replica_dir_path} is not a valid directory path.")

    output_dir_path = (
        Path(rospkg.RosPack().get_path("active_panoptic_mapping"))
        / "config"
        / "habitat"
        / "environments"
        / "replica"
    )

    scene_dirs = [
        p.parents[1]
        for p in replica_dir_path.glob("**/mesh_semantic.ply")
    ]

    for scene_dir_path in tqdm(scene_dirs):
        # Load the mesh
        semantic_mesh_file_path = scene_dir_path / "habitat" / "mesh_semantic.ply"
        semantic_mesh = PlyData.read(str(semantic_mesh_file_path))
        output_file_path = output_dir_path.joinpath(scene_dir_path.name).with_suffix(".yaml")

        vertices = np.column_stack(
            [
                np.array(semantic_mesh["vertex"].data["x"]),
                np.array(semantic_mesh["vertex"].data["y"]),
                np.array(semantic_mesh["vertex"].data["z"]),
            ]
        )

        # Compute min and max coordinates
        x_min, y_min, z_min = vec_habitat_to_ros(
            np.round(np.min(vertices, axis=0)).astype(np.int64)
        )
        x_max, y_max, z_max = vec_habitat_to_ros(
            np.round(np.max(vertices, axis=0)).astype(np.int64)
        )

        config = _CONFIG_TEMPLATE.format(
            **{
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "z_min": z_min,
                "z_max": z_max,
            }
        )

        with output_file_path.open("w") as f:
            yaml.dump(yaml.safe_load(config), f)


def _parse_args():

    parser = argparse.ArgumentParser(
        description="Compute environment config for a scene of the replica dataset."
    )

    parser.add_argument(
        "replica_dir_path",
        type=lambda p: Path(p).absolute(),
        help="Path to replica scene.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    create_replica_scene_config(
        args.replica_dir_path,
    )
