from pytorch_optimize.evaluator import ModelEvaluator
from pytorch_optimize.model import Model
from pytorch_optimize.objective import Objective, Inputs
from torch.nn import Module, Linear
import torch
import pytest
from dataclasses import dataclass


class TestModule(Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = Linear(3, 1, bias=False)

    def forward(self, input: torch.Tensor):
        return self.layer_1(input)


class TestObjective(Objective):
    def __call__(self, model: Model, inputs: Inputs):
        output = model.forward(inputs.x)
        obj_value = [output.sum().item()]
        return obj_value


@dataclass(init=True)
class TestInputs(Inputs):
    x: torch.Tensor


def test_model_evaluator():
    inputs = TestInputs(x=torch.ones(1, 3) * 5)
    model = Model(TestModule())
    objective = TestObjective()
    layer_name, _ = model.sample()
    layer_value = torch.ones(1, 3)
    evaluator = ModelEvaluator(model, objective, layer_name, inputs)
    obj_value = evaluator(layer_value)
    assert len(obj_value) == 1
    assert obj_value[0] == 15.0
