from src.model.functions.prediction.step_size import _step_size


def mean(
        prediction_array: [float],
        last_action_index: int,
        target: float,
        actions_count_array: [int]
):
    prediction_array[last_action_index] = _mean(
        last_prediction=prediction_array[last_action_index],
        target=target,
        step=actions_count_array[last_action_index]
    )
    return prediction_array


def _mean(
        last_prediction: float,
        target: float,
        step: int
) -> float:
    return _step_size(last_prediction, target, 1 / (step - 1))
