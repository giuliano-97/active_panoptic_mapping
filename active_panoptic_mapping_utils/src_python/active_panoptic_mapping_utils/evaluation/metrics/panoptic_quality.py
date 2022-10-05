import numpy as np
from sklearn import metrics as skmetrics

from ..constants import (
    PQ_KEY,
    PQ_THING_KEY,
    PQ_STUFF_KEY,
    SQ_KEY,
    RQ_KEY,
    TP_KEY,
    FN_KEY,
    FP_KEY,
    TP_IOU_THRESHOLD,
    NYU40_NUM_CLASSES,
    NYU40_THING_CLASSES,
    NYU40_STUFF_CLASSES,
    NYU40_CLASS_IDS_TO_NAMES,
    SCANNET_NYU40_EVALUATION_CLASSES,
)
from ..segment_matching import match_segments

_THING_CLASSES_MASK = np.isin(np.arange(NYU40_NUM_CLASSES + 1), NYU40_THING_CLASSES)
_STUFF_CLASSES_MASK = np.isin(np.arange(NYU40_NUM_CLASSES + 1), NYU40_STUFF_CLASSES)
_EVAL_CLASSES_MASK = np.isin(
    np.arange(NYU40_NUM_CLASSES + 1), SCANNET_NYU40_EVALUATION_CLASSES
)


def _compute_qualities(
    iou_per_class: np.ndarray,
    tp_per_class: np.ndarray,
    fp_per_class: np.ndarray,
    fn_per_class: np.ndarray,
):
    """
    Computes Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ)
    for the given semantic segment matching result.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        sq_per_class = np.nan_to_num(iou_per_class / tp_per_class)
        rq_per_class = np.nan_to_num(
            tp_per_class / (tp_per_class + 0.5 * fp_per_class + 0.5 * fn_per_class)
        )
    pq_per_class = np.multiply(sq_per_class, rq_per_class)

    # Evaluate only on classes which appear at least once in the groundtruth
    # and are in the validation classes used by the ScanNet benchmark
    valid_classes_mask = _EVAL_CLASSES_MASK & np.not_equal(
        tp_per_class + fp_per_class + fn_per_class, 0
    )

    # Eval metrics,
    qualities_per_class = np.row_stack((pq_per_class, sq_per_class, rq_per_class))
    counts_per_class = np.row_stack((tp_per_class, fp_per_class, fn_per_class))
    tp, fp, fn = np.sum(
        counts_per_class[:, valid_classes_mask],
        axis=1,
    )

    if np.count_nonzero(valid_classes_mask) > 0:
        pq, sq, rq = np.mean(
            qualities_per_class[:, valid_classes_mask],
            axis=1,
        )
    else:
        pq, sq, rq = 0, 0, 0

    # Also compute pq for thing and stuff classes only
    valid_thing_classes_mask = valid_classes_mask & _THING_CLASSES_MASK
    if np.count_nonzero(valid_thing_classes_mask) > 0:
        pq_th = np.mean(qualities_per_class[0][valid_thing_classes_mask])
    else:
        pq_th = 0

    valid_stuff_classes_mask = valid_classes_mask & _STUFF_CLASSES_MASK
    if np.count_nonzero(valid_stuff_classes_mask) > 0:
        pq_st = np.mean(qualities_per_class[0][valid_stuff_classes_mask])
    else:
        pq_st = 0

    result = {
        PQ_KEY: pq,
        PQ_THING_KEY: pq_th,
        PQ_STUFF_KEY: pq_st,
        SQ_KEY: sq,
        RQ_KEY: rq,
        TP_KEY: tp,
        FP_KEY: fp,
        FN_KEY: fn,
    }

    # Add per-class metrics
    for class_id in SCANNET_NYU40_EVALUATION_CLASSES:
        class_name = NYU40_CLASS_IDS_TO_NAMES[class_id]
        result[f"{PQ_KEY}_{class_name}"] = pq_per_class[class_id]
        result[f"{SQ_KEY}_{class_name}"] = sq_per_class[class_id]
        result[f"{RQ_KEY}_{class_name}"] = rq_per_class[class_id]
        result[f"{TP_KEY}_{class_name}"] = tp_per_class[class_id]
        result[f"{FP_KEY}_{class_name}"] = fp_per_class[class_id]
        result[f"{FN_KEY}_{class_name}"] = fn_per_class[class_id]

    return result

class PanopticQuality:
    def __init__(self, iou_threshold: float = TP_IOU_THRESHOLD):
        self.iou_threshold = iou_threshold
        self._iou_per_class = np.zeros(NYU40_NUM_CLASSES + 1, dtype=np.float64)
        self._tp_per_class = np.zeros(NYU40_NUM_CLASSES + 1, dtype=np.uint128)
        self._fn_per_class = np.zeros(NYU40_NUM_CLASSES + 1, dtype=np.uint128)
        self._fp_per_class = np.zeros(NYU40_NUM_CLASSES + 1, dtype=np.uint128)

    def update(self, gt_labels, pred_labels):
        if gt_labels.shape != pred_labels.shape:
            raise ValueError("Shape of gt and pred labels must be the same!")

        matching_result = match_segments(
            gt_labels,
            pred_labels,
            self.iou_threshold,
        )

        self._iou_per_class += matching_result.iou_per_class
        self._tp_per_class += matching_result.tp_per_class
        self._fp_per_class += matching_result.fp_per_class
        self._fn_per_class += matching_result.fn_per_class

    def compute(self):
        return _compute_qualities(
            self._iou_per_class,
            self._tp_per_class,
            self._fp_per_class,
            self._fn_per_class,
        )


def panoptic_quality(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    match_iou_threshold: float = TP_IOU_THRESHOLD,
):
    if gt_labels.shape != pred_labels.shape:
        raise ValueError("Label arrays must have the same shape!")

    matching_result = match_segments(
        gt_labels=gt_labels,
        pred_labels=pred_labels,
        match_iou_threshold=match_iou_threshold,
    )

    iou_per_class = matching_result.iou_per_class
    tp_per_class = matching_result.tp_per_class
    fp_per_class = matching_result.fp_per_class
    fn_per_class = matching_result.fn_per_class

    return _compute_qualities(
        iou_per_class=iou_per_class,
        tp_per_class=tp_per_class,
        fp_per_class=fp_per_class,
        fn_per_class=fn_per_class,
    )
