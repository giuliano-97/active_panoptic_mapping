from .uncertainty_estimator_base import UncertaintyEstimatorBase


def build_uncertainty_estimator(
    estimator_type: str, **kwargs
) -> UncertaintyEstimatorBase:
    if estimator_type.lower() == "entropy":
        from .entropy import Entropy

        return Entropy()

    elif estimator_type.lower() == "max_logit":
        from .max_logit import MaxLogit

        min = kwargs["min"]
        max = kwargs["max"]
        return MaxLogit(min, max)
    elif estimator_type.lower() == "softmax":
        from .softmax import Softmax

        return Softmax()
    elif estimator_type.lower() == "histogram_binning":
        from .histogram_binning import HistogramBinning

        return HistogramBinning(**kwargs)
