import logging
from typing import Iterable, Self

from numpy import argmax, array, float64, ndarray, int64, ones, vectorize, zeros, square, sqrt, apply_along_axis
from BanditSolvers.src.solvers.base_solver import BaseSolver
from BanditSolvers.src.domain import actionKey
from scipy.stats import gamma


class SamplingSolver(BaseSolver):
    action_counts: ndarray[int64]
    weights: ndarray[float64]
    max_sample_reached: int
    max_sample_size: int
    n_sampling: int
    targets: ndarray[float64]

    def __init__(
            self, 
            action_keys: Iterable[actionKey], 
            n_sampling: int = 1,
            max_sample_size: int = 1000
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
        """
        self.logger: logging.Logger = logging.getLogger(__name__)

        super().__init__(action_keys=action_keys)

        self.action_counts = zeros(len(self.action_keys))
        self.weights = ones([3, len(self.action_keys)])
        self.max_sample_size = max_sample_size
        self.max_sample_reached = 0
        self.n_sampling = n_sampling
        self.targets = zeros((self.max_sample_size, len(self.action_keys)))
        

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
        assert x.size == y.size

        total_action_count: int = sum(self.action_counts)

        if total_action_count >= self.max_sample_size:
            self.max_sample_reached += 1
            self.action_counts = zeros(len(self.action_keys))

            total_action_count = 0

        for action_index, target in zip(x, y):
            self.targets[int(total_action_count), action_index] = target
            self.action_counts[action_index] += 1

        if self.max_sample_reached:
            targets_to_fit: ndarray[float64] = self.targets
        
        else:
            total_action_count: int = sum(self.action_counts)
            targets_to_fit: ndarray[float64] = self.targets[0:int(total_action_count), :]

        gamma_params: ndarray[float64] = apply_along_axis(
            self._fit_gamma_on_targets, 
            axis=0, 
            arr=targets_to_fit
        )

        self.weights = gamma_params

        return self
            

    def predict(self) -> int:
        """_summary_

        Returns
        -------
        int
            _description_
        """
        def sample_distribution(
            alpha: float,
            loc: float,
            scale: float
        ) -> float:
            mean_sample: float = self._sample_distribution(
                alpha=alpha, 
                loc=loc, 
                scale=scale
            )

            return mean_sample

        vec_sample_distribution = vectorize(sample_distribution)

        samples: ndarray[float64] = (
            vec_sample_distribution(self.weights[0, :], self.weights[1, :], self.weights[2, :])
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
            a=alpha, 
            loc=loc, 
            scale=scale, 
            size=self.n_sampling
        )

        mean_sample: float = sum(samples) / self.n_sampling

        return mean_sample

    @staticmethod
    def _fit_gamma_on_targets(targets: ndarray[float64]) -> ndarray[float64]:
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
        if targets.var() > 0:
            shape, loc, scale = gamma.fit(targets)

        else:
            shape, loc, scale = 1, 1, 1

        return array([shape, loc, scale])