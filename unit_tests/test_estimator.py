from torch.nn import Module
from pytorch_optimize.objective import Objective, Samples
from pytorch_optimize.optimizer import ESOptimizer
from pytorch_optimize.model import Model
from torch.optim import SGD
import torch
from dataclasses import dataclass
import pytest


class TestModule(Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.Linear(1, 1)

    def forward(self, input: torch.Tensor):
        return self.layer_1(input)


class TestObjective(Objective):
    def __call__(self, model: Model, samples: Samples, device: str):
        # objective is inverse of squarred loss L = (y - w.x) ^-2
        # dL/dW = 2 (y - wx)^-1 (x)
        output = model.forward(samples.x)
        squarred_loss = (samples.y - output) ** 2
        obj_value = 1/squarred_loss
        return [obj_value]


@dataclass(init=True)
class TestSamples(Samples):
    x: torch.Tensor
    y: torch.Tensor


def test_estimator_step():
    samples = TestSamples(x=torch.ones(1, 1), y=torch.ones(1, 1) * 2)
    regressor = TestModule()
    model = Model(regressor)
    objective = TestObjective()
    optimizer = SGD(regressor.parameters(), 1e-2)
    estimator = ESOptimizer(model, optimizer, objective,
                            [1.0], 1e-4, 50, ["cpu"])
    gradient = estimator.gradient_step(samples)
    assert gradient.shape == (2,1)
