from typing import Callable, Iterable
from src.model.functions.prediction.mean import mean
from src.model.functions.prediction.step_size import step_size

prediction_functions_dict: {str: Callable[..., Iterable[float]]} = {
    'step_size': step_size,
    'mean': mean
}
