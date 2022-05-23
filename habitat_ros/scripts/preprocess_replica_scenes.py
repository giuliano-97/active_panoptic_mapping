import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
from tqdm import tqdm


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def preprocess_replica_scenes(replica_dir_path: Path):
    if not replica_dir_path.is_dir():
        raise FileNotFoundError(f"{str(replica_dir_path)} is not a valid directory.")

    scene_dirs = [
        p
        for p in replica_dir_path.iterdir()
        if p.joinpath("habitat/mesh_semantic.ply").is_file()
        and p.joinpath("habitat/info_semantic.json").is_file()
    ]

    for scene_dir_path in tqdm(scene_dirs):
        mesh_semantic_file_path = scene_dir_path / "habitat/mesh_semantic.ply"
        info_semantic_file_path = scene_dir_path / "habitat/info_semantic.json"

        # Get up dir
        with info_semantic_file_path.open("r") as f:
            scene_info = json.load(f)
            z_scene = -1 * np.array(scene_info["gravity_dir"])

        # Compute the inverse rotation scene to world
        x_world = np.array([1, 0, 0], dtype=np.float64)

        x_scene = normalize_vector(x_world - np.dot(x_world, z_scene) * z_scene)
        y_scene = np.cross(z_scene, x_scene)

        R_scene = np.column_stack([x_scene, y_scene, z_scene])
        R_scene_inv = np.linalg.inv(R_scene)

        # Now load the mesh and compensate for gravity
        mesh_data = PlyData.read(str(mesh_semantic_file_path))
        x = np.array(mesh_data["vertex"].data["x"])
        y = np.array(mesh_data["vertex"].data["y"])
        z = np.array(mesh_data["vertex"].data["z"])
        vertices = np.row_stack([x, y, z])
        vertices_compensated = np.dot(R_scene_inv, vertices)

        mesh_data["vertex"].data["x"] = vertices_compensated[0, :]
        mesh_data["vertex"].data["y"] = vertices_compensated[1, :]
        mesh_data["vertex"].data["z"] = vertices_compensated[2, :]

        preprocessed_mesh_semantic_file_path = (
            mesh_semantic_file_path.parent / "mesh_semantic_preprocessed.ply"
        )
        with preprocessed_mesh_semantic_file_path.open("wb") as f:
            mesh_data.write(f)


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "replica_dir",
        type=lambda p: Path(p).absolute(),
        help="Path to the replica dataset directory",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    preprocess_replica_scenes(
        args.replica_dir,
    )
