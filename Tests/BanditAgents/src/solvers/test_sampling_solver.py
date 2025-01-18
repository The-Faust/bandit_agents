from typing import List

from BanditAgents.src.solvers.sampling_solver import SamplingSolver


class TestSamplingSolver:
    mock_action_keys: List[str]
    mock_n_sampling: int
    mock_max_sample_size: int

    sampling_solver: SamplingSolver

    def setUp(self) -> None:
        self.mock_action_keys = ['action_a', 'action_b']
        self.mock_n_sampling = 1
        self.mock_max_sample_size = 10

    def test_fit_succeed(self) -> None:
        pass

    def test_info_succeed(self) -> None:
        pass

    def test_predict_succeed(self) -> None:
        pass

    def test__sample_distribution_succeed(self) -> None:
        pass

    def test__fit_gamma_on_targets_succeed(self) -> None:
        pass
