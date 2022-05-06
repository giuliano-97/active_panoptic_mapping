from typing import Dict

import numpy as np
from overrides import overrides

from pano_seg.uncertainty.uncertainty_estimator_base import UncertaintyEstimatorBase


class MaxLogit(UncertaintyEstimatorBase):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    @overrides
    def __call__(self, prediction: Dict[str, np.ndarray]) -> np.ndarray:
        mask_logits = prediction["mask_logits"]

        max_mask_logits = np.max(mask_logits, axis=-1)

        normalized_mask_logits = (max_mask_logits + self.min) / (self.max - self.min)

        return normalized_mask_logits
