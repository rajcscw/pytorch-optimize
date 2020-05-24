from pytorch_optimize.objective import AbstractObjective
from pytorch_optimize.model import Model
from copy import deepcopy


class PerturbationEvaluator:
    """
    Evaluator for spinning up the model with sampled weights and
    evaluates it for the specified objective function
    """
    def __init__(self, model: Model, objective_function: AbstractObjective,
                 current_parameter_name: str, current_parameter_value: torch.Tensor):
        self.model = model
        self.objective_function = objective_function
        self.current_parameter_name = current_parameter_name
        self.current_parameter_value = current_parameter_value

    def __call__(self, perturbation: float):
        perturbed = self.current_estimate + perturbation
        model = deepcopy(self.model)
        model.set_layer_value(self.current_parameter_name, perturbed)
        objective_values = self.objective_function(model, batch)
        return objective_values
