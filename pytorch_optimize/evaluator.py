from pytorch_optimize.objective import Objective, Inputs
from pytorch_optimize.model import Model
from copy import deepcopy
import torch


class ModelEvaluator:
    """
    Evaluator for spinning up the model with sampled weights and
    evaluates it for the specified objective function
    """
    def __init__(self, model: Model, objective_function: Objective,
                 current_parameter_name: str, inputs: Inputs):
        self.model = model
        self.objective_function = objective_function
        self.current_parameter_name = current_parameter_name
        self.inputs = inputs

    def __call__(self, layer_weights: torch.Tensor):
        model = deepcopy(self.model)
        model.set_layer_value(self.current_parameter_name, layer_weights)
        objective_values = self.objective_function(model, self.inputs)
        return objective_values
