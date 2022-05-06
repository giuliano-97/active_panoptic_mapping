from .predictor_base import PredictorBase


def build_predictor(predictor_type: str, **kwargs) -> PredictorBase:
    if predictor_type.lower() == "maxdeeplab":
        from pano_seg.predictors.max_deeplab import MaXDeepLabPredictor

        raise NotImplementedError("MaxDeepLab predictor not implemented!")
    elif predictor_type.lower() == "mask2former":
        from .mask2former import Mask2FormerPredictor

        model_dir_path = kwargs["model_dir_path"]
        visualize = kwargs["visualize"]

        return Mask2FormerPredictor(model_dir_path, visualize=visualize)
    else:
        raise ValueError("Invalid predictor type!")
