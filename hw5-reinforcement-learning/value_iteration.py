from typing import Tuple
import numpy as np
import math


class ValueIteration:
    def __init__(self, theta=0.0001, discount_factor=1.0):
        self.theta = theta
        self.discount_factor = discount_factor

    def calculate_q_values(
        self, current_capital: int, value_function: np.ndarray, rewards: np.ndarray
    ) -> np.ndarray:
        """
        Calculate expected value for every possible stake at a given capital.
        """
        # Terminal states have no meaningful actions
        if current_capital == 0 or current_capital == 100:
            return np.zeros(1, dtype=float)

        max_stake = min(current_capital, 100 - current_capital)
        q_values = np.zeros(max_stake, dtype=float)

        probabilities = {
            "lose_full": 5 / 12,
            "win": 1 / 6,
            "lose_half": 5 / 12,
        }

        for idx, stake in enumerate(range(1, max_stake + 1)):
            next_states = [
                current_capital - stake,
                current_capital + stake,
                current_capital - int(math.ceil(stake / 2)),
            ]

            q_values[idx] = (
                probabilities["lose_full"]
                * (rewards[next_states[0]] + self.discount_factor * value_function[next_states[0]])
                + probabilities["win"]
                * (rewards[next_states[1]] + self.discount_factor * value_function[next_states[1]])
                + probabilities["lose_half"]
                * (rewards[next_states[2]] + self.discount_factor * value_function[next_states[2]])
            )

        return q_values

    def value_iteration_for_gamblers(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the Gambler’s problem using value iteration.
        """
        num_states = 101

        rewards = np.zeros(num_states, dtype=float)
        rewards[-1] = 100.0

        values = np.zeros(num_states, dtype=float)

        converged = False
        while not converged:
            delta = 0.0

            for state in range(1, num_states - 1):
                action_values = self.calculate_q_values(state, values, rewards)
                updated_value = action_values.max()
                delta = max(delta, abs(updated_value - values[state]))
                values[state] = updated_value

            converged = delta < self.theta

        policy = np.zeros(num_states, dtype=int)
        for state in range(1, num_states - 1):
            action_values = self.calculate_q_values(state, values, rewards)
            policy[state] = np.argmax(action_values) + 1

        return policy, values
