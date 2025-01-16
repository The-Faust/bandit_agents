from typing import Callable, Self, Tuple

from numpy import ndarray

from BanditAgents.src.contexts.simulation_context import SimulationContext
from BanditAgents.src.domain import actionKey


class SimulationContextBuilder:
    actions: Tuple[
        Tuple[actionKey, Callable[[]], float]
        | Tuple[float, float, float]
        | ndarray[float],
        ...,
    ]

    def __init__(self) -> None:
        self.actions = None

    def with_custom_actions(
        self,
        actions: Tuple[
            Tuple[actionKey, Callable[[]], float]
            | Tuple[float, float, float]
            | ndarray[float],
            ...,
        ],
    ) -> Self:
        self.actions = actions

        return self

    def with_default_actions(self) -> Self:
        actions: tuple[
            tuple[str, Callable[[], float]], tuple[str, Callable[[], float]]
        ] = (("action_a", lambda: 4.0), ("action_b", lambda: 2.0))

        self.actions = actions

        return self

    def build(self) -> SimulationContext:
        if self.actions is None:
            self.with_default_actions()

        simulation_context = SimulationContext(actions=self.actions)

        return simulation_context
