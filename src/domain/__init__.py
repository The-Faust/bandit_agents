from src.domain.bandit_precursor import BanditPrecursor


def make_bandit_precursor(
        actions_keys: [any],
        bandit_key: any,
        optimistic_value=0.0,
        step_size=0.01,
        epsilon=0.01,
        confidence=0.01,
        prediction_type='step_size',
        decision_type='e_greedy'
) -> BanditPrecursor:
    return BanditPrecursor(
        actions_keys=actions_keys,
        bandit_key=bandit_key,
        optimistic_value=optimistic_value,
        step_size=step_size,
        epsilon=epsilon,
        confidence=confidence,
        prediction_type=prediction_type,
        decision_type=decision_type
    )
