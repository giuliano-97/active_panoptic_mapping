from typing import Dict, str
from abc import ABC, abstractmethod

import numpy as np


class UncertaintyEstimatorBase(ABC):
    @abstractmethod
    def __call__(self, prediction: Dict[str, np.ndarray]) -> Dict:
        pass
