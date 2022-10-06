from .mean_iou import mean_iou, MeanIoU
from .panoptic_quality import panoptic_quality, PanopticQuality
from .coverage import coverage

from typing import List, Callable


class Compose:
    def __init__(self, metrics: List):
        self.metrics = metrics
        
    def update(self, gt_labels, pred_labels):
        for metric in self.metrics:
            metric.update(gt_labels, pred_labels)

    def compute(self):
        result = {}
        for metric in self.metrics:
            result.update(metric.compute())
        
        return result
    
def compose(metrics: List[Callable]):
    
    def compute_metrics(gt_labels, pred_labels):
        result = {}
        for metric in metrics:
            result.update(metric(gt_labels, pred_labels))
        return result

    return compute_metrics