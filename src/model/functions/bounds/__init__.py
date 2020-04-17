from typing import Callable, Iterable
from src.model.functions.bounds.ucb import ucb


def not_bounded(prediction_function: Callable[..., Iterable[float]], *args, **kwargs) -> [float]:
    return prediction_function(*args, **kwargs)


bound_functions_dict: {str: Callable[..., Iterable[float]]} = {
    "null": not_bounded,
    "ucb": ucb
}
