from abc import abstractmethod
from pytorch_optimize.model import Model
from pytorch_optimize.batch import Batch
from typing import List


class AbstractObjective:
    """
    Abstract class for implementing objectives or loss functions
    """
    @abstractmethod
    def __call__(self, model: Model, batch: Batch) -> List[float]:
        raise NotImplementedError
