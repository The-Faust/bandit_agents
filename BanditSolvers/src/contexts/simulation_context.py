from typing import Callable, Dict, Iterable, Self, Tuple
from BanditSolvers.src.domain import actionKey


class SimulationContext:
    simulated_action_dict: Dict[actionKey, Callable[[], float]]

    def __init__(self, actions: Iterable[Tuple[actionKey, Callable[[], float]]]) -> None:
        self.simulated_action_dict = {
            action_key: action_func 
            for action_key, action_func in actions
        }

    def add_action(self, action_key: actionKey, action_func: Callable[[], float], is_return: bool = True) -> Self | None:
        self.simulated_action_dict[action_key] = action_func

        if is_return:
            return self
        
        else:
            return
    
    def add_actions(self, actions: Iterable[Tuple[actionKey, Callable[[], float]]]) -> Self:
        add_actions_execution = tuple(
            self.add_action(action_key, action_func, False) 
            for action_key, action_func in actions
        )

        return self
    
class SimulationContextActionFactory:
    def make_gamma_action()
