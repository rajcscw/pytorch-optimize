from abc import abstractmethod
from pytorch_optimize.model import Model
from typing import List
from dataclasses import dataclass


@dataclass(init=True)
class Samples:
    """
    A dataclass to hold input data for evaluation of objectives
    """
    pass


class Objective:
    """
    Abstract class for implementing objectives or loss functions
    """
    @abstractmethod
    def __call__(self, model: Model, inputs: Samples, device: str) -> List[float]:
        raise NotImplementedError
