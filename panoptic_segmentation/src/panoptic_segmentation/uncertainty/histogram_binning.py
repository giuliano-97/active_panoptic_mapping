from pathlib import Path
from typing import Dict, List

import numpy as np
from overrides import overrides

from panoptic_segmentation.uncertainty.uncertainty_estimator_base import (
    UncertaintyEstimatorBase,
)


class HistogramBinning(UncertaintyEstimatorBase):
    def __init__(
        self,
        calibration_histogram_file_path: Path,
    ):
        if not calibration_histogram_file_path.is_file():
            raise FileNotFoundError(
                f"{str(calibration_histogram_file_path)} is not a valid file path!"
            )
        npzfile = np.load(str(calibration_histogram_file_path))
        bins = npzfile["bins"]
        calibrated_confidence_scores_lut = npzfile["calibrated_confidence_scores_lut"]
        self.confidence_bins, self.position_x_bins, self.position_y_bins = bins
        self.calibrated_confidence_scores_lut = calibrated_confidence_scores_lut

    @overrides
    def __call__(self, prediction: Dict[str, np.ndarray]) -> np.ndarray:
        mask_probs = prediction.get("mask_probs", None)

        if mask_probs is None:
            return None

        height, width = mask_probs.shape
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        y_relative = y / height
        x_relative = x / width

        # Digitize values and substract one so we can index the LUT
        digitized_confidence = np.digitize(mask_probs, self.confidence_bins) - 1
        digitized_x_relative = np.digitize(x_relative, self.position_x_bins) - 1
        digitized_y_relative = np.digitize(y_relative, self.position_y_bins) - 1

        sample = np.stack(
            [
                digitized_confidence,
                digitized_x_relative,
                digitized_y_relative,
            ],
            axis=2,
        )

        # Flatten for indexing
        sample = np.reshape(sample, (-1, 3))

        # Query LUT to get confidence scores
        calibrated_confidence_scores = self.calibrated_confidence_scores_lut[
            sample[:, 0], sample[:, 1], sample[:, 2]
        ]
        # Reshape back into image dims
        calibrated_confidence_scores = calibrated_confidence_scores.reshape(
            (height, width)
        )

        return calibrated_confidence_scores
