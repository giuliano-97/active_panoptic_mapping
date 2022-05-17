import json
import os
from pathlib import Path
from typing import List, Dict

from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog


def get_scannet_frames_25k_panoptic_dataset_items(
    scans,
    images_info_list,
    annotations_list,
    scannet_frames_25k_dir_path,
    scannet_panoptic_dir_path,
    metadata,
) -> List[Dict]:
    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    # Get images info - convert to dict for easier lookup - and annotations
    images_info_dict = {item["id"]: item for item in images_info_list}

    # Generate detectron2-compatible dataset items
    dataset_items = []
    for pano_seg_annotation in sorted(annotations_list, key=lambda k: k["image_id"]):
        image_id = pano_seg_annotation["image_id"]
        scan, _ = image_id.split("__")
        if scan not in scans:
            continue

        image_info = images_info_dict[image_id]

        # Convert category id
        segments_info = [
            _convert_category_id(s, metadata)
            for s in pano_seg_annotation["segments_info"]
        ]

        dataset_items.append(
            {
                "image_id": image_id,
                "file_name": str(
                    scannet_frames_25k_dir_path.joinpath(
                        image_info["file_name"]
                    ).with_suffix(".jpg")
                ),
                "height": image_info["height"],
                "width": image_info["width"],
                "pan_seg_file_name": str(
                    scannet_panoptic_dir_path / pano_seg_annotation["file_name"]
                ),
                "segments_info": segments_info,
            }
        )

    return dataset_items


def get_scannet_frames_25k_metadata(category_info_list):
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.

    thing_classes = [
        ci["name"] if ci["isthing"] == True else "" for ci in category_info_list
    ]
    thing_colors = [
        ci["color"] if ci["isthing"] == True else [0, 0, 0] for ci in category_info_list
    ]
    stuff_classes = [ci["name"] for ci in category_info_list]
    stuff_colors = [ci["color"] for ci in category_info_list]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(category_info_list):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


def register_scannet_frames_25k_panoptic(scannet_frames_25k_dir_path):
    if not scannet_frames_25k_dir_path.is_dir():
        return

    # Make panoptic gt labels exist
    scannet_panoptic_dir_path = scannet_frames_25k_dir_path / "scannet_panoptic"
    assert scannet_panoptic_dir_path.is_dir()

    scannet_panoptic_json_file_path = (
        scannet_frames_25k_dir_path / "scannet_panoptic.json"
    )
    assert scannet_panoptic_json_file_path.is_file()
    with scannet_panoptic_json_file_path.open("r") as f:
        scannet_panoptic_info = json.load(f)

    scannetv2_train_split_file_path = Path(
        "/cluster/home/albanesg/mt_ipp_panoptic_mapping/ScanNet/Tasks/Benchmark/scannetv2_train.txt"
    )
    with scannetv2_train_split_file_path.open("r") as f:
        scannetv2_train_scans = [l.rstrip("\n") for l in f.readlines()]
    scannetv2_val_split_file_path = Path(
        "/cluster/home/albanesg/mt_ipp_panoptic_mapping/ScanNet/Tasks/Benchmark/scannetv2_val.txt"
    )
    with scannetv2_val_split_file_path.open("r") as f:
        scannetv2_val_scans = [l.rstrip("\n") for l in f.readlines()]

    metadata = get_scannet_frames_25k_metadata(scannet_panoptic_info["categories"])

    for split, scans in [
        ("train", scannetv2_train_scans),
        ("val", scannetv2_val_scans),
    ]:
        dataset_name = f"scannet_frames_25k_{split}"
        DatasetCatalog.register(
            dataset_name,
            lambda: get_scannet_frames_25k_panoptic_dataset_items(
                scans,
                scannet_panoptic_info["images"],
                scannet_panoptic_info["annotations"],
                scannet_frames_25k_dir_path,
                scannet_panoptic_dir_path,
                metadata,
            ),
        )

        MetadataCatalog.get(dataset_name).set(
            ignore_label=0,
            label_divisor=1000,
            evaluator_type="coco_panoptic_seg",
            **metadata,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_scannet_frames_25k_panoptic(Path(_root) / "scannet_frames_25k")
