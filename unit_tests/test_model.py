from pytorch_optimize.model import Model, SamplingStrategy
from torch.nn import Module, Linear
import torch
from dataclasses import dataclass


class TestModule(Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = Linear(10, 5)
        self.layer_2 = Linear(5, 3)
        self.layer_3 = Linear(3, 1)

    def forward(self, input: torch.Tensor):
        input = self.layer_1(input)
        input = self.layer_2(input)
        input = self.layer_3(input)
        return input


def test_forward():
    model = Model(TestModule())
    input = torch.rand(size=(1, 10))
    output = model.forward(input)
    assert output.shape[0] == 1
    assert output.shape[1] == 1


def test_bottom_up_sampling():
    model = Model(TestModule(), SamplingStrategy.BOTTOM_UP)
    assert model.sample()[0] == "layer_1"
    assert model.sample()[0] == "layer_2"
    assert model.sample()[0] == "layer_3"
    assert model.sample()[0] == "layer_1"


def test_top_down_sampling():
    model = Model(TestModule(), SamplingStrategy.TOP_DOWN)
    assert model.sample()[0] == "layer_3"
    assert model.sample()[0] == "layer_2"
    assert model.sample()[0] == "layer_1"
    assert model.sample()[0] == "layer_3"


def test_random_sampling():
    model = Model(TestModule(), SamplingStrategy.RANDOM)
    assert model.sample()[0] in ["layer_3", "layer_2", "layer_1"]
    assert model.sample()[0] in ["layer_3", "layer_2", "layer_1"]


def test_all_at_once():
    model = Model(TestModule(), SamplingStrategy.ALL)
    layer_name, layer_value = model.sample()
    assert layer_name == "all"
    assert layer_value.shape[0] == 77


def test_set_layer_values():
    layer_weight = torch.rand(size=(5, 10))
    layer_bias = torch.rand(size=(5,))
    flattened_layer_weight = torch.cat((layer_weight.flatten(), layer_bias.flatten())).reshape(-1, 1)
    model = Model(TestModule())
    model.set_layer_value("layer_1", flattened_layer_weight)
    assert torch.allclose(model.net.layer_1.weight, layer_weight)
    assert torch.allclose(model.net.layer_1.bias, layer_bias)


def test_set_gradients():
    model = Model(TestModule(), SamplingStrategy.BOTTOM_UP)
    weight_grads = torch.rand(size=(5, 10))
    bias_grads = torch.rand(size=(5,))
    flattened_layer_grads = torch.cat((weight_grads.flatten(), bias_grads.flatten())).reshape(-1, 1)
    model.set_gradients("layer_1", flattened_layer_grads)
    assert torch.allclose(model.net.layer_1.weight.grad, weight_grads)
    assert torch.allclose(model.net.layer_1.bias.grad, bias_grads)