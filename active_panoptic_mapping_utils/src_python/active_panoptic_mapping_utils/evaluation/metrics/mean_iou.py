import numpy as np

import numpy as np
from sklearn import metrics as skmetrics

from ..constants import (
    NYU40_IGNORE_LABEL,
    MIOU_KEY,
    IOU_KEY_SUFFIX,
    NYU40_NUM_CLASSES,
    NYU40_CLASSES,
    NYU40_CLASS_IDS_TO_NAMES,
    SCANNET_NYU40_EVALUATION_CLASSES,
    PANOPTIC_LABEL_DIVISOR,
)

def _compute_mean_iou(
    confusion_matrix: np.ndarray,
):
    tp_per_class = np.zeros(NYU40_NUM_CLASSES + 1, dtype=np.ulonglong)
    fp_per_class = np.zeros(NYU40_NUM_CLASSES + 1, dtype=np.ulonglong)
    fn_per_class = np.zeros(NYU40_NUM_CLASSES + 1, dtype=np.ulonglong)
    for class_id in SCANNET_NYU40_EVALUATION_CLASSES:
        tp_per_class[class_id] = np.longlong(confusion_matrix[class_id, class_id])
        fn_per_class[class_id] = (
            np.longlong(np.sum(confusion_matrix[class_id, :])) - tp_per_class[class_id]
        )
        not_ignored = [l for l in SCANNET_NYU40_EVALUATION_CLASSES if not l == class_id]
        fp_per_class[class_id] = np.longlong(
            np.sum(confusion_matrix[not_ignored, class_id])
        )

    
    with np.errstate(divide="ignore", invalid="ignore"):
        iou_per_class = np.nan_to_num(
            tp_per_class / (tp_per_class + fp_per_class + fn_per_class),
            nan=0.0,
        )

    # Compute iou only for evaluation classes
    result = {}

    result[MIOU_KEY] = np.mean(iou_per_class[SCANNET_NYU40_EVALUATION_CLASSES])
    for class_id in SCANNET_NYU40_EVALUATION_CLASSES:
        class_name = NYU40_CLASS_IDS_TO_NAMES[class_id]
        result[f"{class_name}_{IOU_KEY_SUFFIX}"] = iou_per_class[class_id]

    return result


def _compute_confusion_matrix(gt_panoptic_labels, pred_panoptic_labels):
    gt_semantic_labels = gt_panoptic_labels // PANOPTIC_LABEL_DIVISOR
    pred_semantic_labels = pred_panoptic_labels // PANOPTIC_LABEL_DIVISOR

    # Ignore areas were ground truth is void
    pred_semantic_labels_valid = pred_semantic_labels[
        gt_semantic_labels != NYU40_IGNORE_LABEL
    ]
    gt_semantic_labels_valid = gt_semantic_labels[
        gt_semantic_labels != NYU40_IGNORE_LABEL
    ]

    return skmetrics.confusion_matrix(
        y_true=gt_semantic_labels_valid,
        y_pred=pred_semantic_labels_valid,
        labels=[NYU40_IGNORE_LABEL] + NYU40_CLASSES,
    )


class MeanIoU:
    def __init__(self):
        self.confusion_matrix = np.zeros(
            (NYU40_NUM_CLASSES + 1, NYU40_NUM_CLASSES + 1),
            dtype=np.uint64,
        )

    def update(
        self,
        gt_panoptic_labels,
        pred_panoptic_labels,
    ):
        # Compute confusion matrix and add it
        cmat = _compute_confusion_matrix(
            gt_panoptic_labels,
            pred_panoptic_labels,
        ).astype(np.uint64)

        self.confusion_matrix += cmat

    def compute(self):
        return _compute_mean_iou(self.confusion_matrix)


def mean_iou(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray
):
    confusion_matrix = _compute_confusion_matrix(gt_labels, pred_labels)
    return _compute_mean_iou(confusion_matrix)