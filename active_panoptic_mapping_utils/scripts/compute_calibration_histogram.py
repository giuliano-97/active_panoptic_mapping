#!/usr/bin/env python3

import argparse
from pathlib import Path

from importlib_metadata import metadata
from black import detect_target_versions

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from detectron2.data import DatasetCatalog, MetadataCatalog

from active_panoptic_mapping_core.mask2former.data import datasets
from active_panoptic_mapping_core.predictors import build_predictor


from detectron2.evaluation import DatasetEvaluator
from panopticapi.utils import rgb2id


def _ids_to_counts(id_grid: np.ndarray):
    """Given a numpy array, a mapping from each unique entry to its count."""
    ids, counts = np.unique(id_grid, return_counts=True)
    return dict(zip(ids, counts))


def match_segments(gt_labels, pred_labels, match_iou_threshold=0.5, offset=256 ** 2):

    gt_segment_areas = _ids_to_counts(gt_labels)
    pred_segment_areas = _ids_to_counts(pred_labels)

    intersection_ids = gt_labels.astype(np.int64) * offset + pred_labels.astype(
        np.int64
    )
    intersection_areas = _ids_to_counts(intersection_ids)

    gt_matched = set()
    pred_matched = set()
    matches = dict()

    for intersection_id, intersection_area in intersection_areas.items():
        gt_panoptic_label = intersection_id // offset
        pred_panoptic_label = intersection_id % offset

        gt_class_id = gt_panoptic_label // 1000
        pred_class_id = pred_panoptic_label // 1000

        if pred_class_id <= 0:
            continue

        union = (
            gt_segment_areas[gt_panoptic_label]
            + pred_segment_areas[pred_panoptic_label]
            - intersection_area
        )

        iou = intersection_area / union
        if iou > match_iou_threshold:
            # Sanity check on FP mathces
            if gt_class_id != pred_class_id:
                continue
            # Record a TP
            gt_matched.add(gt_panoptic_label)
            pred_matched.add(pred_panoptic_label)
            matches.update({pred_panoptic_label: gt_panoptic_label})
    return matches


class PanopticSegmentorCalibrationErrorEvaluator(DatasetEvaluator):
    def __init__(
        self,
        metadata,
        confidence_n_bins,
        position_x_n_bins,
        position_y_n_bins,
        min_bin_count=8,
    ):
        self.metadata = metadata

        self.min_bin_count = min_bin_count

        self.confidence_n_bins = confidence_n_bins
        self.confidence_bins = np.linspace(
            0, 1, self.confidence_n_bins + 1, dtype=np.float32
        )
        self.position_x_n_bins = position_x_n_bins
        self.position_x_bins = np.linspace(
            0, 1, self.position_x_n_bins + 1, dtype=np.float32
        )
        self.position_y_n_bins = position_y_n_bins
        self.position_y_bins = np.linspace(
            0, 1, self.position_y_n_bins + 1, dtype=np.float32
        )

        # Number of TPs for each bin
        self._correct_count_bins = np.zeros(
            shape=(
                self.confidence_n_bins,
                self.position_x_n_bins,
                self.position_y_n_bins,
            ),
            dtype=np.uint64,
        )

        # Number of samples for each bin
        self._total_count_bins = np.zeros_like(
            self._correct_count_bins, dtype=np.uint64
        )
        # Sum of tp confidence scores
        self._total_confidence_bins = np.zeros_like(
            self._correct_count_bins, dtype=np.float64
        )
        self._total_count = 0

    @staticmethod
    def _make_panoptic_labels(id_image, segments_info):
        ids = np.unique(id_image)
        contiguous_ids = np.arange(ids.size)
        id_to_contiguous_id = dict(zip(ids, contiguous_ids))
        panoptic_labels = np.zeros_like(id_image, dtype=np.int64)
        for sinfo in segments_info:
            category_id_plus_one = sinfo["category_id"]
            id = sinfo["id"]

            if sinfo["isthing"]:
                instance_id = id_to_contiguous_id[id]
                panoptic_labels[id_image == id] = (
                    category_id_plus_one * 1000 + instance_id
                )
            else:
                panoptic_labels[id_image == id] = category_id_plus_one * 1000

        return panoptic_labels

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            panoptic_seg_img, segments_info = output["panoptic_seg"]
            confidence_scores = output["mask_probs"]
            panoptic_labels = (
                PanopticSegmentorCalibrationErrorEvaluator._make_panoptic_labels(
                    panoptic_seg_img, segments_info
                )
            )

            # Load ground truth
            panoptic_seg_img_gt_file_path = Path(input["pan_seg_file_name"])
            panoptic_seg_img_gt = np.array(Image.open(panoptic_seg_img_gt_file_path))
            if len(panoptic_seg_img_gt.shape) == 3:
                panoptic_seg_img_gt = rgb2id(panoptic_seg_img_gt)
            segments_info_gt = input["segments_info"]

            panoptic_labels_gt = (
                PanopticSegmentorCalibrationErrorEvaluator._make_panoptic_labels(
                    panoptic_seg_img_gt, segments_info_gt
                )
            )

            # Match predicted to ground truth segments
            tp_id_to_gt_id = match_segments(
                panoptic_labels_gt,
                panoptic_labels,
                offset=256000,
            )

            # Remap source ids to ground truth ids - ignore unmatched segments
            remapped_panoptic_labels = np.zeros_like(panoptic_labels, dtype=np.int64)
            for tp_id, gt_id in tp_id_to_gt_id.items():
                remapped_panoptic_labels[panoptic_labels == tp_id] = gt_id

            # Compute correct predictions
            valid_mask = np.logical_and(
                panoptic_seg_img_gt != 0,  # or -1?
                panoptic_seg_img != 0,
            )
            correct_mask = np.logical_and(
                remapped_panoptic_labels == panoptic_labels_gt,
                valid_mask,
            )

            # Create sample
            height, width = panoptic_seg_img.shape
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            y_relative = y / height
            x_relative = x / width
            sample = np.stack(
                [
                    confidence_scores,
                    x_relative,
                    y_relative,
                ],
                axis=2,
            )
            sample_valid = sample[valid_mask]
            sample_valid_confidence_scores = confidence_scores[valid_mask]
            sample_correct = sample[correct_mask]

            # Compute histogram for correct predictions
            histogram_correct, _ = np.histogramdd(
                sample_correct,
                bins=[
                    self.confidence_bins,
                    self.position_x_bins,
                    self.position_y_bins,
                ],
            )

            # Compute histogram for all valid predictions
            histogram_valid, _ = np.histogramdd(
                sample_valid,
                bins=[
                    self.confidence_bins,
                    self.position_x_bins,
                    self.position_y_bins,
                ],
            )

            # Compute histogram for all valid prediction using confidence as weights
            histogram_confidence, _ = np.histogramdd(
                sample_valid,
                bins=[
                    self.confidence_bins,
                    self.position_x_bins,
                    self.position_y_bins,
                ],
                weights=sample_valid_confidence_scores,
            )

            # Update counts
            self._correct_count_bins += histogram_correct.astype(np.uint64)
            self._total_count_bins += histogram_valid.astype(np.uint64)
            self._total_confidence_bins += histogram_confidence.astype(np.float64)
            self._total_count += np.count_nonzero(valid_mask)

    def evaluate(self):
        inv_total_count_bins = np.reciprocal(
            self._total_count_bins.astype(np.float64),
            out=np.zeros_like(self._total_count_bins, dtype=np.float64),
            where=self._total_count_bins > self.min_bin_count,
        )
        relative_frequency_bins = np.multiply(
            self._correct_count_bins,
            inv_total_count_bins,
        )

        average_confidence_bins = np.multiply(
            self._total_confidence_bins,
            inv_total_count_bins,
        )

        weights_bins = np.divide(
            self._total_count_bins,
            self._total_count,
            dtype=np.float64,
        )

        ece_bins = np.multiply(
            np.abs(relative_frequency_bins - average_confidence_bins),
            weights_bins,
        )
        return np.sum(ece_bins, axis=None)

    def get_calibration_histogram(self):
        bins = [self.confidence_bins, self.position_x_bins, self.position_y_bins]
        inv_total_count_bins = np.reciprocal(
            self._total_count_bins.astype(np.float64),
            out=np.zeros_like(self._total_count_bins, dtype=np.float64),
            where=self._total_count_bins > self.min_bin_count,
        )
        calibrated_confidence_scores_lut = np.multiply(
            self._correct_count_bins,
            inv_total_count_bins,
        )
        return bins, calibrated_confidence_scores_lut


def main(
    model_dir_path: Path,
    dataset_name: str,
):
    # Get dataset and dataset metadata
    if dataset_name not in MetadataCatalog.list():
        raise ValueError(f"Dataset {dataset_name} is not registered!")

    data_items = DatasetCatalog.get(dataset_name)
    metadata = DatasetCatalog.get(dataset_name)

    # Load model
    predictor = build_predictor(
        "mask2former",
        model_dir_path=model_dir_path,
        visualize=False,
        use_dataset_category_ids=False,
    )

    # Calibrate model
    print("Estimating expected calibration error.")
    dece_evaluator = PanopticSegmentorCalibrationErrorEvaluator(metadata, 15, 15, 15)
    for data_sample_dict in tqdm(data_items):
        image_file_path = Path(data_sample_dict["file_name"])
        image = cv2.imread(str(image_file_path))
        predictions_dict = predictor(image)
        panoptic_seg_img = predictions_dict["panoptic_seg"]
        segments_info = predictions_dict["segments_info"]
        mask_probs = predictions_dict["mask_probs"]
        dece_evaluator.process(
            [data_sample_dict],
            [
                {
                    "panoptic_seg": (panoptic_seg_img, segments_info),
                    "mask_probs": mask_probs,
                }
            ],
        )

    print("Done.")
    print(f"Detection expected calibration error (D-ECE): {dece_evaluator.evaluate()}")

    # Export calibration histogram
    print(f"Exporting calibration histogram to {str(model_dir_path)}")
    bins, calibrated_confidence_scores_lut = dece_evaluator.get_calibration_histogram()
    calibration_histogram_file_path = model_dir_path / "calibration_histogram.npy"
    np.savez(
        str(calibration_histogram_file_path),
        bins=bins,
        calibrationed_confidence_scores_lut=calibrated_confidence_scores_lut,
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compute histogram for histogram "
        "binning calibration for panoptic seg model.",
    )

    parser.add_argument(
        "--model-dir",
        type=lambda p: Path(p).resolve(),
        required=True,
        help="Path to model dir",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        args.model_dir,
        args.dataset,
    )
