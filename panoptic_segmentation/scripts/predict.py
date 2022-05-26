#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

from panoptic_segmentation import build_predictor
from panoptic_segmentation.visualization import colorize_panoptic_segmentation


def predict(
    model_dir_path: Path,
    images_dir_path: Path,
    output_dir_path: Path,
    visualize: bool = True,
):
    assert model_dir_path.is_dir()
    assert images_dir_path.is_dir()
    output_dir_path.mkdir(exist_ok=True, parents=True)

    # Create default predictor
    predictor = build_predictor(
        predictor_type="mask2former",
        model_dir_path=model_dir_path,
        visualize=False,
    )

    # Collect all the images in the target dir
    images_files = [p for p in images_dir_path.glob("*.jpg")]

    # Run inference over every image
    for image_file_path in tqdm(images_files):
        image = cv2.imread(str(image_file_path))
        predictions = predictor(image)
        panoptic_seg = predictions["panoptic_seg"].astype(np.uint16)
        segments_info = predictions["segments_info"]
        confidence_scores = predictions["mask_probs"].astype(np.float64)

        # Save the segmentation as png
        panoptic_seg_file_path = output_dir_path / (
            image_file_path.stem + "_segmentation.png"
        )
        Image.fromarray(panoptic_seg).save(panoptic_seg_file_path)

        # Save the segments info as json
        segments_info_file_path = output_dir_path / (
            image_file_path.stem + "_segments_info.json"
        )
        with segments_info_file_path.open("w") as f:
            json.dump(segments_info, f)

        # Save the uncertainty as tiff
        confidence_map_file_path = output_dir_path / (
            image_file_path.stem + "_uncertainty.tiff"
        )
        Image.fromarray(confidence_scores).save(confidence_map_file_path)

        if visualize:
            colorized_panoptic_seg_file_path = output_dir_path / (
                image_file_path.stem + "_segmentation_colorized.jpg"
            )
            colorized_panoptic_seg, _ = colorize_panoptic_segmentation(
                panoptic_seg, segments_info
            )
            Image.fromarray(colorized_panoptic_seg).save(
                colorized_panoptic_seg_file_path
            )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with given models on the selected images."
    )

    parser.add_argument(
        "--model-dir",
        required=True,
        type=lambda p: Path(p).absolute(),
    )

    parser.add_argument(
        "--images-dir",
        required=True,
        type=lambda p: Path(p).absolute(),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=lambda p: Path(p).absolute(),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    predict(
        model_dir_path=args.model_dir,
        images_dir_path=args.images_dir,
        output_dir_path=args.output_dir,
    )
