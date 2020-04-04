from math import log, sqrt
from numpy import argmax, nditer, sum, array


class UCB:
    @staticmethod
    def _get_upper_bound(speculated_reward, action_count, total_actions_count, confidence):
        return speculated_reward + confidence * (sqrt(log(total_actions_count) / action_count) if action_count > 0 else 1)

    @staticmethod
    def _get_upper_bounds(speculated_reward_array: [float], action_counts: [int], confidence: float):
        total_actions_count = sum(action_counts)
        return array([
            UCB._get_upper_bound(speculated_reward, action_count, total_actions_count, confidence)
            for speculated_reward, action_count
            in nditer([speculated_reward_array, action_counts])
        ])

    @staticmethod
    def make_decision(speculated_reward_array: [float], action_counts: [int], confidence: float):
        return argmax(UCB._get_upper_bounds(speculated_reward_array, action_counts, confidence))
