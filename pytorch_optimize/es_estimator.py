import numpy as np
from torch.multiprocessing import Pool
import torch
from functools import reduce
from typing import List, Tuple, Callable, Dict
from sprl_package.core_components.models import PyTorchModel


class ESEstimator:
    """
    An optimizer class that implements Evolution Strategies (ES)
    """
    def __init__(self, sigma: float, samples: int, loss_function: Callable,
                 model: PyTorchModel, device_list: List[str],
                 obj_weights: torch.Tensor, use_parallel_gpu=True,
                 parallel_workers=4):
        self.sigma = sigma
        self.samples = samples
        self.loss_function = loss_function
        self.model = model
        self.weights = weights
        self.p = Pool(processes=parallel_workers)

    def _compute_ranks(self, x: List[float]):
        """
        Returns ranks in [0, len(x))
        Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
        """
        assert x.ndim == 1
        ranks = np.empty(len(x), dtype=int)
        ranks[x.argsort()] = np.arange(len(x))
        return ranks

    def _compute_centered_ranks(self, x: List[float]):
        """
        Computes centered ranks
        """
        y = self._compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
        y /= (x.size - 1)
        y -= .5
        return y

    def _fit_rank_scaler(self, values: List[float]) -> Dict[float, float]:
        """Fits and returns a rank scaler (a dict mapping)
        """
        ranked_values = self._compute_centered_ranks(values)

        # now, create a mapping of value as key and its rank transformed
        scaler_mapping = {}
        for value, ranked in zip(values, ranked_values):
            scaler_mapping[value] = ranked
        return scaler_mapping

    def _compute_gradient(self, obj_value: List[float], delta: float):
        """
        Computes the gradient for one sample
        """
        obj = torch.Tensor(obj)
        weighted_sum = torch.dot(obj, self.weight)
        grad = delta * weighted_sum
        return grad

    def _fit_scalers_for_objectives(self, objectives: List[List[float]]) -> List[Dict[float, float]]:
        """
        Fits rank scalers for each objectives
        """
        rank_scalers = []
        n_objectives = self.weights.shape[0]
        n_values = len(objectives)
        for obj_ix in range(n_objectives):
            values = [objectives[obj_ix][i] for i in range(n_values)]
            scaler = self._fit_rank_scaler(np.array(all_reward_scores))
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
            obj = [rank_scalers[ix] for ix, value in obj]

            # compute gradient
            gradient = self._compute_weighted_total_objective(obj, delta)
            total_gradients += gradient

        # average the gradients
        grad = total_gradients / (self.k * self.sigma)

        return grad

    def _run_parellely(self, current_estimate, current_parameter):

        # create mirrored sampled perturbations
        perturbs = [torch.randn_like(current_estimate) for i in range(int(self.k/2))]
        mirrored = [-i for i in perturbs]
        perturbs += mirrored

        # seeds for grid envs (also mirrored)
        seeds = list(range(int(self.k/2)))
        seeds += seeds

        # run parallely for all k mirrored candidates
        self.runner.sigma = self.sigma
        self.runner.current_estimate = current_estimate
        self.runner.current_parameter = current_parameter
        self.runner.device = self.device_list[0]
        obj_values = self.p.starmap(self.runner.run_for_perturb, zip(perturbs, seeds))
        behaviors = [item[1] for item in obj_values]
        gradient = self._gradients_from_objectives(current_estimate, obj_values)
        return gradient, behaviors

    def estimate_gradient(self, current_parameter: str, current_value: torch.Tensor,
                          **kwargs) -> torch.Tensor:
        """[summary]

        Arguments:
            current_parameter {str} -- [description]
            current_value {torch.Tensor} -- [description]

        Returns:
            torch.Tensor -- [description]
        """
        g_t = self._run_parellely(current_estimate, current_parameter)
        return g_t
