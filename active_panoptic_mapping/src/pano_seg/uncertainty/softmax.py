from typing import Dict

import numpy as np
from overrides import overrides
from scipy.special import softmax

from pano_seg.uncertainty.uncertainty_estimator_base import UncertaintyEstimatorBase


class Softmax(UncertaintyEstimatorBase):
    @overrides
    def __call__(self, prediction: Dict[str, np.ndarray]) -> np.ndarray:
        mask_probs = prediction.get("mask_probs", None)
        if mask_probs is not None:
            return mask_probs

        mask_logits = prediction["mask_logits"]

        mask_probs = softmax(mask_logits, axis=-1)

        return mask_probs
