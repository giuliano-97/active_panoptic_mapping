from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class PredictorBase(ABC):
    
    @abstractmethod
    def __call__(self, image: np.ndarray) -> Dict:
        pass
