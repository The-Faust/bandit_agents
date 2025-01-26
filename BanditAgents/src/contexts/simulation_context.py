import logging
from typing import Any, Callable, Dict, Iterable, List, Self, Tuple, Type
from numpy import arange, array, dtype, empty, float64, int64, ndarray
from scipy.stats import gamma
from BanditAgents.src.domain import actionKey
from BanditAgents.src.domain.context_action_args import MakeGammaActionArgs
from BanditAgents.src.solvers.base_solver import BaseSolver


class SimulationContextActionFactory:
    def make_gamma_action_from_data(
        self, y: ndarray[float]
    ) -> Callable[[], float]:
        """Sometimes you might want to run a simulation
        with actions that try to mimic real data.

        This method facilitate this process

        Parameters
        ----------
        y : ndarray[float]
            Vector of values to compute the distribution

        Returns
        -------
        Callable[[], float]
            a function that calls a gamma distributions fitted on y
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
        """Through this method you can add actions to the Simulation

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
            An Iterable, could be a list, a tuple, a generator, etc. of pairs.

            - The first member of every pair must be an actionKey.
            - The second member can be a Callable that must return a float,
                a tuple containing alpha, loc and scale parameters
                for a gamma distribution,
                or an array of floats that will be used
                to fit a gamma distribution on them

        Returns
        -------
        Self
            The SimulationContext with the added actions
        """
        for action_key, action_func in actions:
            self.add_action(action_key, action_func, False)

        return self

    def run(
        self,
        n_steps: int,
        solvers: List[Type[(BaseSolver,)]],
        steps_by_ticks: int = 1,
        as_dict: bool = False,
    ) -> Iterable[
        Tuple[
            str,
            Dict[str, any]
            | Tuple[
                ndarray[int64], ndarray[int64], ndarray[str], ndarray[float64]
            ],
        ]
    ]:
        """_summary_

        Parameters
        ----------
        n_steps : int
            Number of steps that the simulation will execute
        solvers : List[Type[(BaseSolver,)]]
            _description_
        steps_by_ticks : int, optional
            _description_, by default 1
        as_dict : bool, optional
            _description_, by default False

        Returns
        -------
        Tuple[ndarray[int64], ndarray[int64], ndarray[str], ndarray[float64]]
        | Dict[str, ndarray]
            _description_
        """
        action_keys = solvers[0].action_keys

        # insure that action keys are aligned
        assert all(
            [solver.action_keys == action_keys for solver in solvers[1:]]
        )

        steps: ndarray[int64] = arange(0, n_steps)
        targets: ndarray[float64] = empty((n_steps, len(self.action_dict)))

        for step in range(n_steps):
            targets[step] = array([self.action_dict[k]() for k in action_keys])

        for solver in solvers:
            self.logger.debug(f"Running simulation for {solver.solver_id}")

            last_training_index: int = 0
            solver_targets: ndarray[float64] = empty(n_steps)
            action_keys: ndarray[str] = empty(n_steps, dtype="<U100")
            indexes: ndarray[int64] = empty(n_steps, dtype=int64)

            for step in range(n_steps):
                self.logger.debug(
                    f"Simulation step {step}\n---------------------------------------"
                    "----------------------------------------------------"
                )
                action_index: int = solver.predict()
                indexes[step] = action_index

                if (
                    step % steps_by_ticks == 0 or step == n_steps - 1
                ) and step != 0:
                    self.logger.debug(f"step {step} is a training step")

                    difference: int = (
                        (step - last_training_index)
                        if step == n_steps - 1
                        else steps_by_ticks
                    )
                    start_index: int = step - difference
                    end_index: int = step + 1 if step == n_steps - 1 else step

                    self.logger.debug(
                        "training on targets indexes: "
                        f"{start_index} to {end_index}"
                    )

                    steps_to_execute: ndarray[int64] = steps[
                        start_index:end_index
                    ]
                    indexes_to_execute: ndarray[int64] = indexes[
                        start_index:end_index
                    ]
                    self.logger.debug(
                        f"{solver.solver_id}'s decision indexes were {indexes_to_execute}"
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

                    tick_targets: ndarray[Any, dtype[Any]] = array(
                        [
                            self.act(targets, step_to_execute, action_index)
                            for step_to_execute, action_index in zip(
                                steps_to_execute, indexes_to_execute
                            )
                        ]
                    )
                    self.logger.debug(
                        f"Decisions wielded following targets {tick_targets}"
                    )

                    solver_targets[start_index:end_index] = tick_targets

                    self.logger.debug(
                        f"Fitting solver {solver.solver_id} with targets"
                    )
                    solvers = solver.fit(x=indexes_to_execute, y=tick_targets)
                    self.logger.debug(
                        "the training resulted in the following weights "
                        f"{solver.weights}"
                    )

                    last_training_index = step

            self.logger.debug(
                "----------------------------------------------------------"
                "---------------------------------\nRun completed!\n"
            )

            if as_dict:
                results: Dict[str, ndarray] = {
                    "steps": steps,
                    "action_indexes": indexes,
                    "action_keys": action_keys,
                    "targets": solver_targets,
                }

            else:
                results = (steps, indexes, action_keys, targets)

            yield solver.solver_id, results

    def act(
        self, targets: ndarray[float64], step: int, action_index: int
    ) -> float:
        """_summary_

        Parameters
        ----------
        targets : ndarray[float64]
            _description_
        step : int
            _description_
        action_index : int
            _description_

        Returns
        -------
        float
            _description_
        """
        target: float = targets[step, action_index]

        return target
