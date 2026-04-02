from typing import List, Tuple
import gymnasium as gym
import numpy as np

SEED = 63
rng = np.random.default_rng(SEED)


class Qlearning:
    def __init__(
        self,
        learning_rate: float,
        gamma: float,
        state_size: int,
        action_size: int,
        epsilon: float,
    ):
        self.state_size = state_size
        self.action_space_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.qtable = np.zeros((state_size, action_size))

    def update(self, state: int, action: int, reward: float, new_state: int):
        """
        Perform a standard Q-learning update using a temporal-difference target.
        """
        current_q = self.qtable[state, action]
        future_q = np.max(self.qtable[new_state])
        td_error = reward + self.gamma * future_q - current_q
        self.qtable[state, action] = current_q + self.learning_rate * td_error

    def reset_qtable(self):
        """Reset all Q-values to zero."""
        self.qtable.fill(0.0)

    def select_epsilon_greedy_action(self, state: int) -> int:
        """
        Choose an action using an epsilon-greedy strategy.
        """
        explore = rng.random() <= self.epsilon
        if explore:
            return int(rng.integers(0, self.action_space_size))

        state_q = self.qtable[state]
        max_value = np.max(state_q)
        best_actions = np.where(state_q == max_value)[0]
        return int(rng.choice(best_actions))

    def train_episode(self, env: gym.Env) -> Tuple[float, int]:
        """
        Execute one complete episode of interaction with the environment.
        """
        state, _ = env.reset()
        cumulative_reward = 0.0
        step_count = 0

        done = False
        while not done:
            action = self.select_epsilon_greedy_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            self.update(state, action, reward, next_state)

            cumulative_reward += reward
            step_count += 1
            state = next_state
            done = terminated or truncated

        return cumulative_reward, step_count

    def run_environment(
        self, env: gym.Env, num_episodes: int
    ) -> Tuple[List[float], List[int]]:
        """
        Train the agent across multiple episodes.
        """
        rewards_history = []
        steps_history = []

        for _ in range(num_episodes):
            episode_reward, episode_steps = self.train_episode(env)
            rewards_history.append(episode_reward)
            steps_history.append(episode_steps)

        return rewards_history, steps_history
