from typing import Dict

import numpy as np
from overrides import overrides
from scipy.stats import entropy
from scipy.special import softmax

from pano_seg.uncertainty.uncertainty_estimator_base import UncertaintyEstimatorBase

_ENTROPY_LOG_BASE = 2


class Entropy(UncertaintyEstimatorBase):
    @overrides
    def __call__(self, prediction: Dict[str, np.ndarray]) -> np.ndarray:
        mask_logits = prediction["mask_logits"]

        mask_logits_softmax = softmax(mask_logits, axis=-1)

        max_entropy = np.log2(mask_logits.shape[2])
        # Compute entropy
        mask_logits_entropy = entropy(
            pk=mask_logits_softmax,
            qk=None,  # No KL divergence
            base=_ENTROPY_LOG_BASE,
            axis=2,
        )

        # Normalize the entropy
        uncertainty_map = mask_logits_entropy / max_entropy
        return uncertainty_map
