import logging
from typing import Dict, Iterable, Self

from numpy import (
    array,
    float64,
    ndarray,
    int64,
    ones,
    vectorize,
    zeros,
    full,
    nan,
    isnan,
)
from BanditAgents.src.solvers.solver import Solver
from BanditAgents.src.domain import actionKey, solverKey
from scipy.stats import gamma


class SamplingSolver(Solver):
    action_counts: ndarray[int64]
    weights: ndarray[float64]
    max_sample_reached: ndarray[int64]
    max_sample_size: int
    n_sampling: int
    targets: ndarray[float64]

    def __init__(
        self,
        action_keys: Iterable[actionKey],
        n_sampling: int = 1,
        max_sample_size: int = 1000,
        solver_id: solverKey = None,
    ) -> None:
        """_summary_

        Parameters
        ----------
        action_keys : Iterable[actionKey]
            _description_
        n_sampling : int, optional
            _description_, by default 1
        max_sample_size : int, optional
            _description_, by default 1000
        solver_id: solverKey, optional
            _description_, by default None
        """
        self.logger: logging.Logger = logging.getLogger(__name__)

        super().__init__(action_keys=action_keys, solver_id=solver_id)

        self.action_counts = zeros(len(self.action_keys))
        self.weights = ones([3, len(self.action_keys)])
        self.max_sample_size = max_sample_size
        self.max_sample_reached = zeros(len(self.action_keys))
        self.n_sampling = n_sampling
        self.targets = full((self.max_sample_size, len(self.action_keys)), nan)

    def fit(self, x: ndarray[int], y: ndarray[float]) -> Self:
        """_summary_

        Parameters
        ----------
        x : ndarray[int]
            _description_
        y : ndarray[float]
            _description_

        Returns
        -------
        Self
            _description_
        """
        self.logger.debug("fitting sample solver")
        assert x.size == y.size

        for action_index, target in zip(x, y):
            self.targets[
                int(self.action_counts[action_index]), action_index
            ] = target

            if self.action_counts[action_index] < self.max_sample_size - 1:
                self.action_counts[action_index] += 1
            else:
                self.action_counts[action_index] = 0
                self.max_sample_reached[action_index] += 1

        self.logger.debug(self.targets.shape)

        for i in range(self.targets.shape[1]):
            self.weights[:, i] = self._fit_gamma_on_targets(self.targets[:, i])

        return self

    def info(self) -> Dict[str, any]:
        info = super().info()
        info["targets"] = self.targets

        return info

    def predict(self) -> int:
        """_summary_

        Returns
        -------
        int
            _description_
        """

        def sample_distribution(
            alpha: float, loc: float, scale: float
        ) -> float:
            mean_sample: float = self._sample_distribution(
                alpha=alpha, loc=loc, scale=scale
            )

            return mean_sample

        vec_sample_distribution = vectorize(sample_distribution)

        samples: ndarray[float64] = vec_sample_distribution(
            self.weights[0, :], self.weights[1, :], self.weights[2, :]
        )

        return samples.argmax()

    def _sample_distribution(self, alpha, loc, scale) -> float:
        """_summary_

        Parameters
        ----------
        alpha : _type_
            _description_
        loc : _type_
            _description_
        scale : _type_
            _description_

        Returns
        -------
        float
            _description_
        """
        samples: ndarray[float64] = gamma.rvs(
            a=alpha, loc=loc, scale=scale, size=self.n_sampling
        )

        mean_sample: float = sum(samples) / self.n_sampling

        return mean_sample

    def _fit_gamma_on_targets(
        self, targets: ndarray[float64]
    ) -> ndarray[float64]:
        """_summary_

        Parameters
        ----------
        targets : ndarray[float64]
            _description_

        Returns
        -------
        ndarray[float64]
            _description_
        """
        training_targets = targets[~isnan(targets)]

        if training_targets.var() > 0:
            shape, loc, scale = gamma.fit(training_targets)

        else:
            shape, loc, scale = 1, 1, 1

        return array([shape, loc, scale])
