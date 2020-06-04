import numpy as np
from torch.multiprocessing import Pool
import torch
from functools import reduce
from typing import List, Tuple, Callable, Dict
from pytorch_optimize.model import Model
from pytorch_optimize.objective import Objective, Samples
from pytorch_optimize.scaler import RankScaler
from pytorch_optimize.evaluator import ModelEvaluator
from torch.optim import Optimizer
import random


class ESOptimizer:
    """
    An optimizer class that implements Evolution Strategies (ES)
    """
    def __init__(self, model: Model, sgd_optimizer: Optimizer, objective_fn: Objective,
                 obj_weights: List[float], sigma: float, n_samples: int, devices: List, n_workers=4):
        self.model = model
        self._optimizer = sgd_optimizer
        self.sigma = sigma
        self.n_samples = n_samples
        self.objective_fn = objective_fn
        self.obj_weights = torch.Tensor(obj_weights)
        self.n_objectives = len(obj_weights)
        self.devices = devices
        self.pool = Pool(processes=n_workers)

        # evaluator
        self.evaluator = ModelEvaluator(self.model, self.objective_fn, None, None)

    def _compute_gradient(self, obj_value: List[float], delta: float):
        """
        Computes the gradient for one sample
        """
        obj_value = torch.Tensor(obj_value)
        weighted_sum = torch.dot(obj_value, self.obj_weights)
        grad = delta * weighted_sum
        return grad

    def _fit_scalers_for_objectives(self, objectives: List[List[float]]) -> List[RankScaler]:
        """
        Fits rank scalers for each objectives
        """
        rank_scalers = []
        n_values = len(objectives)
        for obj_ix in range(self.n_objectives):
            values = [objectives[i][obj_ix] for i in range(n_values)]
            scaler = RankScaler(values)
            rank_scalers.append(scaler)
        return rank_scalers

    def _gradients_from_objectives(self, current_value: torch.Tensor,
                                   obj_values: List[List[float]],
                                   perturbations: List[torch.Tensor]) -> torch.Tensor:
        """
        Computes average gradients using multi-objective values
        """
        total_gradients = torch.zeros(current_value.shape)
        rank_scalers = self._fit_scalers_for_objectives(obj_values)

        for obj, delta in zip(obj_values, perturbations):
            # rank scale them
            obj = [rank_scalers[ix].transform(value) for ix, value in enumerate(obj)]

            # compute gradient
            gradient = self._compute_gradient(obj, delta)
            total_gradients += gradient

        # average the gradients
        grad = total_gradients / (self.n_samples * self.sigma)

        return grad

    def _generate_perturbations(self, current_value):
        # create mirrored sampled perturbations
        perturbs = [torch.randn_like(current_value) for i in range(int(self.n_samples/2))]
        mirrored = [-i for i in perturbs]
        perturbs += mirrored
        return perturbs

    def gradient_step(self, samples: Samples):
        """
        Performs a gradient ascent step

        Args:
            samples (Samples): samples
        """

        # sample some parameters here
        parameter_name, parameter_value = self.model.sample()

        # generate unit gaussian perturbations
        unit_perturbations = self._generate_perturbations(parameter_value)

        # apply user selected deviation
        perturbations = [parameter_value + perturb * self.sigma for perturb in unit_perturbations]

        # sample devices
        devices = random.choices(self.devices, k=len(unit_perturbations))

        # get the objective values
        self.evaluator.current_parameter_name = parameter_name
        self.evaluator.samples = samples
        obj_values = self.pool.starmap(self.evaluator, zip(perturbations, devices))

        # compute gradients
        gradients = self._gradients_from_objectives(parameter_value, obj_values, unit_perturbations)

        # update the model paramters and take a gradient step
        self.model.set_gradients(parameter_name, -gradients)
        self._optimizer.step()

        return gradients
