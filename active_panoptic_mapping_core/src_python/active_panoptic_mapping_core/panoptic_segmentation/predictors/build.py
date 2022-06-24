from .predictor_base import PredictorBase


def build_predictor(predictor_type: str, **kwargs) -> PredictorBase:
    if predictor_type.lower() == "maxdeeplab":
        from panoptic_segmentation.predictors.max_deeplab import MaXDeepLabPredictor

        raise NotImplementedError("MaxDeepLab predictor not implemented!")
    elif predictor_type.lower() == "mask2former":
        from .mask2former import Mask2FormerPredictor

        return Mask2FormerPredictor(**kwargs)
    else:
        raise ValueError("Invalid predictor type!")
