import numpy as np
from torch.multiprocessing import Pool
import torch
from functools import reduce
from typing import List, Tuple, Callable, Dict
from pytorch_optimize.model import Model
from pytorch_optimize.objective import Objective, Inputs
from pytorch_optimize.scaler import RankScaler
from pytorch_optimize.evaluator import ModelEvaluator
from torch.optim import Optimizer


class ESOptimizer:
    """
    An optimizer class that implements Evolution Strategies (ES)
    """
    def __init__(self, model: Model, sgd_optimizer: Optimizer, sigma: float, objective_fn: Objective,
                 obj_weights: torch.Tensor, n_samples: int, n_workers=4):
        self.sigma = sigma
        self.n_samples = n_samples
        self.objective_fn = objective_fn
        self.model = model
        self.obj_weights = obj_weights
        self.pool = Pool(processes=n_workers)

    def _compute_gradient(self, obj_value: List[float], delta: float):
        """
        Computes the gradient for one sample
        """
        obj = torch.Tensor(obj)
        weighted_sum = torch.dot(obj, self.weight)
        grad = delta * weighted_sum
        return grad

    def _fit_scalers_for_objectives(self, objectives: List[List[float]]) -> List[RankScaler]:
        """
        Fits rank scalers for each objectives
        """
        rank_scalers = []
        n_objectives = self.weights.shape[0]
        n_values = len(objectives)
        for obj_ix in range(n_objectives):
            values = [objectives[obj_ix][i] for i in range(n_values)]
            scaler = RankScaler(values)
            rank_scalers.append(scaler)
        return rank_scalers

    def _gradients_from_objectives(self, current_value: torch.Tensor,
                                   obj_values: List[List[float]] -> torch.Tensor):
        """
        Computes average gradients using multi-objective values
        """
        total_gradients = torch.zeros(current_estimate.shape)
        rank_scalers = self._fit_scalers_for_objectives(obj_values)

        for obj in obj_values:
            obj, delta = obj

            # rank scale them
            obj = [rank_scalers[ix].transform(value) for ix, value in obj]

            # compute gradient
            gradient = self._compute_gradient(obj, delta)
            total_gradients += gradient

        # average the gradients
        grad = total_gradients / (self.n_samples * self.sigma)

        return grad

    def _generate_perturbations(self, current_value):
        # create mirrored sampled perturbations
        perturbs = [torch.randn_like(current_estimate) for i in range(int(self.n_samples/2))]
        mirrored = [-i for i in perturbs]
        perturbs += mirrored
        return perturbs

    def gradient_step(self, inputs: Inputs) -> torch.Tensor:

        # sample some parameters here
        parameter_name, parameter_value = self.model.sample()

        # generate unit gaussian perturbations
        unit_perturbations = self._generate_perturbations(current_value)

        # apply user selected deviation
        perturbations = [perturb * self.sigma for perturb in unit_perturbations]

        # run the evaluator
        evaluator = ModelEvaluator(self.model, self.objective_fn, parameter_name)

        # get the objective values
        obj_values = self.p.map(evaluator, perturbations)

        # compute gradients
        gradients = self._gradients_from_objectives(current_estimate, obj_values)

        # update the model paramters and take a gradient step
        self.model.set_gradients(parameter_name, gradients)
        self.optimizer.step()
