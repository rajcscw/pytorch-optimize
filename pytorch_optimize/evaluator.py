from pytorch_optimize.objective import Objective
from pytorch_optimize.model import Model
from copy import deepcopy
from pytorch_optimize.batch import Batch


class ModelEvaluator:
    """
    Evaluator for spinning up the model with sampled weights and
    evaluates it for the specified objective function
    """
    def __init__(self, model: Model, objective_function: Objective,
                 current_parameter_name: str, batch: Batch):
        self.model = model
        self.objective_function = objective_function
        self.current_parameter_name = current_parameter_name
        self.batch = batch

    def __call__(self, layer_weights: torch.Tensor):
        model = deepcopy(self.model)
        model.set_layer_value(self.current_parameter_name, layer_weights)
        objective_values = self.objective_function(model, self.batch)
        return objective_values
