from typing import Iterable, Self

from BanditAgents.src.domain import actionKey
from BanditAgents.src.solvers.base_solver import BaseSolver


class BaseSolverBuilder:
    action_keys: Iterable[actionKey] = None

    def with_action_keys(self, action_keys: Iterable[actionKey]) -> Self:
        self.action_keys = action_keys

        return self

    def with_default_action_keys(self) -> Self:
        self.action_keys = ("action_a", "action_b")

        return self

    def build(self) -> BaseSolver:
        if self.action_keys is None:
            self.with_default_action_keys()

        base_solver = BaseSolver(self.action_keys)

        return base_solver
