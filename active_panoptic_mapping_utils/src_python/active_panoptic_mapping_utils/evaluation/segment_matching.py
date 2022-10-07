from dataclasses import dataclass
from typing import Mapping, Dict, Set, Tuple

import numpy as np
from .constants import (
    TP_IOU_THRESHOLD,
    MIN_SEGMENT_AREA,
    NYU40_NUM_CLASSES,
    NYU40_IGNORE_LABEL,
    PANOPTIC_LABEL_DIVISOR,
)


_OFFSET = 256 * 256


@dataclass
class SegmentMatchingResult:
    iou_per_class: np.ndarray
    tp_per_class: np.ndarray
    fp_per_class: np.ndarray
    fn_per_class: np.ndarray
    matching: Dict[int, int]
    unmatched_gt_segments: Set[int]
    matched_segments_confusion_matrix: np.ndarray


def _ids_to_counts(id_grid: np.ndarray) -> Mapping[int, int]:
    """Given a numpy array, a mapping from each unique entry to its count."""
    ids, counts = np.unique(id_grid, return_counts=True)
    return dict(zip(ids, counts))


def match_segments(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    match_iou_threshold: float = TP_IOU_THRESHOLD,
    fp_min_area: int = MIN_SEGMENT_AREA,
):
    assert gt_labels.shape == pred_labels.shape

    iou_per_class = np.zeros(NYU40_NUM_CLASSES + 1, dtype=np.float64)
    tp_per_class = np.zeros(NYU40_NUM_CLASSES + 1, dtype=np.uint64)
    fp_per_class = np.zeros(NYU40_NUM_CLASSES + 1, dtype=np.uint64)
    fn_per_class = np.zeros(NYU40_NUM_CLASSES + 1, dtype=np.uint64)

    gt_segment_areas = _ids_to_counts(gt_labels)
    pred_segment_areas = _ids_to_counts(pred_labels)

    # Combine the groundtruth and predicted labels. Dividing up the voxels
    # based on which groundtruth segment and which predicted segment they belong
    # to, this will assign a different 64-bit integer label to each choice
    # of (groundtruth segment, predicted segment), encoded as
    #   gt_panoptic_label * offset + pred_panoptic_label.
    intersection_ids = gt_labels.astype(np.int64) * _OFFSET + pred_labels.astype(
        np.int64
    )
    intersection_areas = _ids_to_counts(intersection_ids)

    def prediction_ignored_overlap(pred_panoptic_label):
        return intersection_areas.get(pred_panoptic_label, 0)

    gt_matched = set()
    pred_matched = set()
    matching: Dict[int, int] = {}
    unmatched_gt_segments = set()
    matched_segments_confusion_matrix = np.zeros(
        (NYU40_NUM_CLASSES + 1, NYU40_NUM_CLASSES + 1), dtype=np.ulonglong
    )

    def record_match(pred_panoptic_label, gt_panoptic_label):
        matching.update({pred_panoptic_label: gt_panoptic_label})
        pred_semantic_label = pred_panoptic_label // PANOPTIC_LABEL_DIVISOR
        gt_semantic_label = gt_panoptic_label // PANOPTIC_LABEL_DIVISOR
        matched_segments_confusion_matrix[gt_semantic_label, pred_semantic_label] += 1

    for intersection_id, intersection_area in intersection_areas.items():
        gt_panoptic_label = intersection_id // _OFFSET
        pred_panoptic_label = intersection_id % _OFFSET

        gt_class_id = gt_panoptic_label // PANOPTIC_LABEL_DIVISOR
        pred_class_id = pred_panoptic_label // PANOPTIC_LABEL_DIVISOR

        if pred_class_id <= NYU40_IGNORE_LABEL:
            continue

        union = (
            gt_segment_areas[gt_panoptic_label]
            + pred_segment_areas[pred_panoptic_label]
            - intersection_area
            - prediction_ignored_overlap(pred_panoptic_label)
        )

        iou = intersection_area / union
        if iou > match_iou_threshold:
            record_match(pred_panoptic_label, gt_panoptic_label)
            # Sanity check on FP mathces
            if gt_class_id != pred_class_id:
                continue
            # Record a TP
            tp_per_class[gt_class_id] += 1
            iou_per_class[gt_class_id] += iou
            gt_matched.add(gt_panoptic_label)
            pred_matched.add(pred_panoptic_label)

    for gt_panoptic_label in gt_segment_areas:
        if gt_panoptic_label == NYU40_IGNORE_LABEL:
            continue
        if gt_panoptic_label in gt_matched:
            continue
        class_id = gt_panoptic_label // PANOPTIC_LABEL_DIVISOR
        # Failing to detect a void segment is not a false negative.
        if class_id == NYU40_IGNORE_LABEL:
            continue
        fn_per_class[class_id] += 1
        unmatched_gt_segments.add(gt_panoptic_label)

    # Count false positives for each category.
    for pred_panoptic_label in pred_segment_areas:
        if pred_panoptic_label in pred_matched:
            continue

        if pred_segment_areas[pred_panoptic_label] < fp_min_area:
            continue
        # A false positive is not penalized if is mostly ignored in the ignored ground truth
        if (
            prediction_ignored_overlap(pred_panoptic_label)
            / pred_segment_areas[pred_panoptic_label]
        ) > TP_IOU_THRESHOLD:
            continue
        class_id = pred_panoptic_label // PANOPTIC_LABEL_DIVISOR
        if class_id == NYU40_IGNORE_LABEL:
            continue
        fp_per_class[class_id] += 1
        matching.update({pred_panoptic_label: 0})

    return SegmentMatchingResult(
        tp_per_class=tp_per_class,
        iou_per_class=iou_per_class,
        fp_per_class=fp_per_class,
        fn_per_class=fn_per_class,
        matching=matching,
        unmatched_gt_segments=unmatched_gt_segments,
        matched_segments_confusion_matrix=matched_segments_confusion_matrix,
    )
