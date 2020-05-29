from typing import List
import numpy as np


class RankScaler:
    def __init__(self, values: List[float]):
        self.values = values
        self._mapping = self._map_to_ranks(values)

    def transform(self, value: float) -> float:
        """
        Returns its rank transformed value
        """
        return self._mapping[value]

    @staticmethod
    def _map_to_ranks(values: List[float]):
        ranked_values = RankScaler._compute_centered_ranks(values)

        # now, create a mapping of value as key and its rank transformed
        scaler_mapping = {}
        for value, ranked in zip(values, ranked_values):
            scaler_mapping[value] = ranked
        return scaler_mapping

    @staticmethod
    def _compute_ranks(values: np.array):
        """
        Returns ranks in [0, len(x))
        Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
        """
        assert values.ndim == 1
        ranks = np.empty(len(values), dtype=int)
        ranks[values.argsort()] = np.arange(len(values))
        return ranks

    @staticmethod
    def _compute_centered_ranks(x: List[float]):
        """
        Computes centered ranks
        """
        x = np.array(x)
        y = RankScaler._compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
        y /= (x.size - 1)
        y -= .5
        return y


if __name__ == "__main__":
    scaler = RankScaler([-10, 6, 10, -5])
    print(scaler.transform(-10))
    print(scaler.transform(6))
    print(scaler.transform(10))
