def step_size(
        prediction_array: [float],
        last_action_index: int,
        target: float,
        step_size_value: float
) -> [float]:
    new_prediction_array = prediction_array.copy()
    new_prediction_array[last_action_index] = _step_size(
        last_prediction=prediction_array[last_action_index],
        target=target,
        step_size_value=step_size_value
    )
    return new_prediction_array


def _step_size(
        last_prediction: float,
        target: float,
        step_size_value: float
) -> float:
    return last_prediction + step_size_value * (target - last_prediction)
