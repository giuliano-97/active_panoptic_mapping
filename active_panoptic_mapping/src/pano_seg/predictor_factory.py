from pathlib import Path

from pano_seg.predictors.predictor_base import PredictorBase


class PredictorFactory:
    @staticmethod
    def get_predictor(predictory_type: str, model_dir_path: Path, visualize: bool = False) -> PredictorBase:
        if predictory_type.lower() == "maxdeeplab":
            from pano_seg.predictors.max_deeplab import MaXDeepLabPredictor

            raise NotImplementedError("MaxDeepLab predictor not implemented!")
        elif predictory_type.lower() == "mask2former":
            from pano_seg.predictors.mask2former import Mask2FormerPredictor

            return Mask2FormerPredictor(model_dir_path, visualize=visualize)
        else:
            raise ValueError("Invalid predictor type!")
