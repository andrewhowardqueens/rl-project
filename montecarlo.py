from collections import defaultdict
import math
import random


def state_to_str(state, pretty=False):
    seperator = '\n' if pretty else ''
    return seperator.join(''.join(row) for row in state)


class MonteCarloAgent:

    def __init__(self, n: int = 3, epsilon=0.1, learning_rate=0.1, discount_factor=0.9):
        self._action_values = defaultdict(lambda: defaultdict(lambda: (0, 0)))
        self.n = n
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def get_action_value(self, state, action):
        reward_sum, n = self._action_values[state_to_str(state)][action]
        if n == 0:
            return 0
        return reward_sum/n

    def update_action_value(self, state, action, reward):
        reward_sum, n = self._action_values[state_to_str(state)][action]
        n += 1
        reward_sum += reward
        self._action_values[state_to_str(state)][action] = (reward_sum, n)

    def epsilon_greedy_policy(self, state, legal_actions):
        best_value, best_action = -math.inf, None

        for action in legal_actions:
            action_value = self.get_action_value(state, action)
            if action_value > best_value:
                best_action = action
                best_value = action_value

        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        else:
            return best_action

    def update_action_values(self, episode, reward):
        episode_length = len(episode)
        G = [0, 0]
        for step in range(episode_length-1, -1, -1):
            player = step % 2
            state, action = episode[step]
            step_reward = reward[player] if step >= episode_length - 2 else 0

            G[player] = self.discount_factor * G[player] + step_reward
            self.update_action_value(state, action, G[player])
