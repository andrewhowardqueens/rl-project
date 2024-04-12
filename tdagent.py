from collections import defaultdict
import math
import random


def state_to_str(state, pretty=False):
    seperator = '\n' if pretty else ''
    return seperator.join(''.join(row) for row in state)


class TDAgent:

    def __init__(self, n: int = 3, epsilon=0.1, learning_rate=0.1, discount_factor=0.9):
        self._action_values = defaultdict(lambda: defaultdict(lambda: 0))
        self.n = n
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def get_action_value(self, state, action):
        return self._action_values[state_to_str(state)][action]

    def update_action_value(self, state, action, value):
        self._action_values[state_to_str(state)][action] = value

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
        for step in range(episode_length):
            state, action = episode[step]
            action_value = self.get_action_value(state, action)
            if step + 2 < episode_length:
                next_state, next_action = episode[step + 2]
                next_action_value = self.get_action_value(next_state,next_action)
            else:
                next_action_value = 0

            if step >= episode_length - 2:
                step_reward = reward[step % 2]
            else:
                step_reward = 0

            updated_action_value = action_value + self.learning_rate * (
                    step_reward + self.discount_factor * next_action_value - action_value)

            self.update_action_value(state, action, updated_action_value)