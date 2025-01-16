import logging
from typing import Any, Callable, Dict, Iterable, Never, Self, Tuple, Type
from numpy import arange, array, dtype, empty, float64, int64, ndarray
from scipy.stats import gamma
from BanditAgents.src.domain import actionKey
from BanditAgents.src.domain.context_action_args import MakeGammaActionArgs
from BanditAgents.src.solvers.base_solver import BaseSolver


class SimulationContextActionFactory:
    def make_gamma_action_from_data(
        self, y: ndarray[float]
    ) -> Callable[[], float]:
        """_summary_

        Parameters
        ----------
        y : ndarray[float]
            _description_

        Returns
        -------
        Callable[[], float]
            _description_
        """
        shape, loc, scale = gamma.fit(y)

        f: Callable[[], float] = self.make_gamma_action(
            alpha=shape, loc=loc, scale=scale
        )

        return f

    def make_gamma_action(
        self, alpha: float, loc: float = None, scale: float = None
    ) -> Callable[[], float]:
        """_summary_

        Parameters
        ----------
        alpha : float
            _description_
        loc : float, optional
            _description_, by default None
        scale : float, optional
            _description_, by default None

        Returns
        -------
        Callable[[], float]
            _description_
        """
        g_kwargs = dict(a=alpha)

        if loc is not None:
            g_kwargs[MakeGammaActionArgs.LOC.value] = loc

        if scale is not None:
            g_kwargs[MakeGammaActionArgs.SCALE.value] = scale

        f: Callable[[], float] = lambda: gamma.rvs(**g_kwargs, size=1)[0]

        return f


class SimulationContext:
    action_dict: Dict[actionKey, Callable[[any], float]]
    simulation_context_action_factory: SimulationContextActionFactory

    def __init__(
        self,
        actions: Iterable[
            Tuple[
                actionKey,
                Callable[[any], float] | Tuple[float, ...] | ndarray[float],
            ]
        ],
    ) -> None:
        """_summary_

        Parameters
        ----------
        actions : Iterable[
            Tuple[actionKey, Callable[[any], float]
            |  Tuple[float, ...]
            |  ndarray[float]]
        ]
            _description_
        """
        self.logger: logging.Logger = logging.getLogger(__name__)

        self.simulation_context_action_factory = (
            SimulationContextActionFactory()
        )
        self.action_dict = dict()

        self.add_actions(actions)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.logger.debug(traceback)

        return

    def add_action(
        self,
        action_key: actionKey,
        action_data: (
            Callable[[any], float] | Tuple[float, ...] | ndarray[float]
        ),
        is_return: bool = True,
    ) -> Self | None:
        """_summary_

        Parameters
        ----------
        action_key : actionKey
            _description_
        action_data : Callable[[any], float]
            | Tuple[float, ...]
            | ndarray[float]
            _description_
        is_return : bool, optional
            _description_, by default True

        Returns
        -------
        Self | None
            _description_
        """
        if callable(action_data):
            action_func: Callable[[any], float] = action_data

        elif isinstance(action_data, Tuple[float, float, float]):
            assert len(action_data) < 3

            action_func: Callable[[], float] = (
                self.simulation_context_action_factory.make_gamma_action(
                    *action_data
                )
            )

        elif isinstance(action_data, ndarray[float]):
            factory: SimulationContextActionFactory = (
                self.simulation_context_action_factory
            )
            action_func: Callable[[], float] = (
                factory.make_gamma_action_from_data(action_data)
            )

        self.action_dict[action_key] = action_func

        if is_return:
            return self

        return

    def add_actions(
        self,
        actions: Iterable[
            Tuple[
                actionKey,
                Callable[[any], float] | Tuple[float, ...] | ndarray[float],
            ]
        ],
    ) -> Self:
        """_summary_

        Parameters
        ----------
        actions : Iterable[
            Tuple[
                actionKey,
                Callable[[any], float]
                |  Tuple[float, ...]
                |  ndarray[float]
            ]
        ]
            _description_

        Returns
        -------
        Self
            _description_
        """
        for action_key, action_func in actions:
            self.add_action(action_key, action_func, False)

        return self

    def run(
        self,
        n_steps: int,
        solver: Type[(BaseSolver,)],
        steps_by_ticks: int = 1,
        act_args_func: Callable[[actionKey], Tuple[any, ...]] = None,
        as_dict: bool = False,
    ) -> (
        Tuple[ndarray[int64], ndarray[int64], ndarray[str], ndarray[float64]]
        | Dict[str, ndarray]
    ):
        """_summary_

        Parameters
        ----------
        n_steps : int
            _description_
        solver : Type[
            _description_
        steps_by_ticks : int, optional
            _description_, by default 1
        act_args_func : Callable[[actionKey], Tuple[any, ...]], optional
            _description_, by default None
        as_dict : bool, optional
            _description_, by default False

        Returns
        -------
        Tuple[ndarray[int64], ndarray[int64], ndarray[str], ndarray[float64]]
        | Dict[str, ndarray]
            _description_
        """
        steps: ndarray[int64] = arange(0, n_steps)
        indexes: ndarray[int64] = empty(n_steps, dtype=int64)
        action_keys: ndarray[str] = empty(n_steps, dtype="<U100")
        targets: ndarray[float64] = empty(n_steps)
        last_training_index: int = 0

        self.logger.debug("Running simulation")

        for i in range(n_steps):
            self.logger.debug(
                f"Simulation step {i}\n---------------------------------------"
                "----------------------------------------------------"
            )
            action_index: int = solver.predict()
            indexes[i] = action_index

            if (i % steps_by_ticks == 0 or i == n_steps - 1) and i != 0:
                self.logger.debug(f"step {i} is a training step")

                difference: int = (
                    (i - last_training_index)
                    if i == n_steps - 1
                    else steps_by_ticks
                )
                start_index: int = i - difference
                end_index: int = i + 1 if i == n_steps - 1 else i

                self.logger.debug(
                    "training on targets indexes: "
                    f"{start_index} to {end_index}"
                )

                indexes_to_execute: ndarray[int64] = indexes[
                    start_index:end_index
                ]
                self.logger.debug(
                    f"Solvers decision indexes were {indexes_to_execute}"
                )

                action_keys_to_execute: Tuple[actionKey] = array(
                    [
                        action_key
                        for action_key in solver.indexes_to_action_keys(
                            indexes_to_execute
                        )
                    ],
                    dtype="<U100",
                )
                self.logger.debug(
                    f"Which corresponds to actions {action_keys_to_execute}"
                )

                action_keys[start_index:end_index] = action_keys_to_execute

                self.logger.debug("Executing decisions")
                if act_args_func is not None:
                    act_args = [
                        act_args_func(action_key)
                        for action_key in action_keys_to_execute
                    ]

                else:
                    act_args: Tuple[Never] = [
                        () for _ in action_keys_to_execute
                    ]

                tick_targets: ndarray[Any, dtype[Any]] = array(
                    [
                        self.act(action_key, *act_args[i])
                        for i, action_key in enumerate(action_keys_to_execute)
                    ]
                )
                self.logger.debug(
                    f"Decisions wielded following targets {tick_targets}"
                )

                targets[start_index:end_index] = tick_targets

                self.logger.debug("Fitting solver with targets")
                solver = solver.fit(x=indexes_to_execute, y=tick_targets)
                self.logger.debug(
                    "the training resulted in the following weights "
                    f"{solver.weights}"
                )

                last_training_index = i

        self.logger.debug(
            "----------------------------------------------------------"
            "---------------------------------\nRun completed!\n"
        )

        if as_dict:
            results: Dict[str, ndarray] = {
                "steps": steps,
                "action_indexes": indexes,
                "action_keys": action_keys,
                "targets": targets,
            }

        else:
            results = (steps, indexes, action_keys, targets)

        self.logger.debug(f"The results are {results}")

        return results

    def act(self, action_key: actionKey, *args, **kwargs) -> float:
        """_summary_

        Parameters
        ----------
        action_key : actionKey
            _description_

        Returns
        -------
        float
            _description_
        """
        target: float = self.action_dict[action_key](*args, **kwargs)

        return target
