import logging
from typing import Any, Callable, Dict, Iterable, Never, Self, Tuple, Type
from numpy import arange, array, dtype, empty, float64, int64, ndarray
from scipy.stats import gamma
from BanditSolvers.src.domain import actionKey
from BanditSolvers.src.domain.context_action_args import MakeGammaActionArgs
from BanditSolvers.src.solvers.base_solver import BaseSolver


class SimulationContextActionFactory:
    def make_gamma_action_from_data(self, y: ndarray[float]) -> Callable[[], float]:
        shape, loc, scale= gamma.fit(y)

        f: Callable[[], float] = self.make_gamma_action(alpha=shape, loc=loc, scale=scale)

        return f

    def make_gamma_action(
        self, 
        alpha: float, 
        loc: float = None, 
        scale: float = None, 
    ) -> Callable[[], float]:
        g_kwargs = dict(a=alpha)

        if loc is not None:
            g_kwargs[MakeGammaActionArgs.LOC.value] = loc

        if scale is not None:
            g_kwargs[MakeGammaActionArgs.SCALE.value] = scale

        f: Callable[[], float] = (
            lambda: gamma
            .rvs(
                **g_kwargs,
                size=1
            )[0]
        )

        return f


class SimulationContext:
    simulated_action_dict: Dict[actionKey, Callable[[any], float]]
    simulation_context_action_factory: SimulationContextActionFactory
        
    def __init__(
        self, 
        actions: Iterable[Tuple[actionKey, Callable[[any], float] | Tuple[float, ...] | ndarray[float]]]
    ) -> None:
        self.logger = logging.getLogger(__name__)

        self.simulation_context_action_factory = SimulationContextActionFactory()
        self.simulated_action_dict = dict()

        self.add_actions(actions)

    def __enter__(self) -> Self:
        return self
    
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return

    def add_action(
        self, 
        action_key: actionKey, 
        action_data: Callable[[any], float] | Tuple[float, ...] | ndarray[float], 
        is_return: bool = True
    ) -> Self | None:
        if callable(action_data):
            action_func: Callable[[any], float] = action_data
            
        elif isinstance(action_data, Tuple[float, float, float]):
            assert len(action_data) < 3

            action_func: Callable[[], float] = (
                self.simulation_context_action_factory
                .make_gamma_action(*action_data)
            )

        elif isinstance(action_data, ndarray[float]):
            action_func: Callable[[], float] = (
                self.simulation_context_action_factory
                .make_gamma_action_from_data(action_data)
            )

        self.simulated_action_dict[action_key] = action_func

        if is_return:
                return self
            
        return
    
    def add_actions(
        self, 
        actions: Iterable[Tuple[actionKey, Callable[[any], float] | Tuple[float, ...] | ndarray[float]]]
    ) -> Self:
        for action_key, action_func in actions:
            self.add_action(action_key, action_func, False) 

        return self
    
    def run(
        self,
        n_steps: int, 
        solver: Type[(BaseSolver,)], 
        steps_by_ticks: int = 1,
        act_args_func: Callable[[actionKey], Tuple[any, ...]] = None,
        as_dict: bool = False
    ) -> Tuple[ndarray[int64], ndarray[int64], ndarray[str], ndarray[float64]] | Dict[str, ndarray]:
        steps = arange(0, n_steps)
        indexes: ndarray[int64] = empty(n_steps, dtype=int64)
        action_keys: ndarray[str] = empty(n_steps, dtype='<U100')
        targets: ndarray[float64] = empty(n_steps)
        last_training_index: int = 0

        self.logger.debug(f'Running simulation')

        for i in range(n_steps):
            self.logger.debug(f'Simulation step {i}\n-------------------------------------------------------------------------------------------')
            action_index: int = solver.predict()
            indexes[i] = action_index

            if i % steps_by_ticks == 0 or i == n_steps - 1:
                self.logger.debug(f'step {i} is a training step')

                difference =  (i - last_training_index) if i == n_steps - 1 else steps_by_ticks
                start_index = i - difference
                end_index = i + 1 if i == n_steps - 1 else i

                self.logger.debug(f'training on targets indexes: {start_index} to {end_index}')

                indexes_to_execute: ndarray[int64] = indexes[start_index: end_index]
                self.logger.debug(f'Solvers decision indexes were {indexes_to_execute}')

                action_keys_to_execute: Tuple[actionKey] = array([
                    action_key for action_key 
                    in solver.indexes_to_action_keys(indexes_to_execute)
                ], dtype='<U100')
                self.logger.debug(f'Which corresponds to actions {action_keys_to_execute}')

                action_keys[start_index: end_index] = action_keys_to_execute

                self.logger.debug(f'Executing decisions')
                if act_args_func is not None:
                    act_args = [
                        act_args_func(action_key) 
                        for action_key in action_keys_to_execute
                    ]
                
                else:
                    act_args: Tuple[Never] = [() for _ in action_keys_to_execute]

                tick_targets: ndarray[Any, dtype[Any]] = array([
                    self.act(action_key, *act_args[i]) 
                    for i, action_key in enumerate(action_keys_to_execute)
                ])
                self.logger.debug(f'Decisions wielded following targets {tick_targets}')

                targets[start_index: end_index] = tick_targets

                self.logger.debug(f'Fitting solver with targets')
                solver = solver.fit(x=indexes_to_execute, y=tick_targets)
                self.logger.debug(f'the training resulted in the following weights {solver.weights}')

                last_training_index = i

        self.logger.debug(f'-------------------------------------------------------------------------------------------\nRun completed!\n')

        if as_dict:
            results: Dict[str, ndarray] = {
                'steps': steps,
                'action_indexes': indexes,
                'action_keys': action_keys,
                'targets': targets
            }

        else:
            results = (steps, indexes, action_keys, targets)

        self.logger.debug(f'The results are {results}')

        return results

    def act(self, action_key: actionKey, *args, **kwargs) -> float:
        target: float = self.simulated_action_dict[action_key](*args, **kwargs)

        return target
