from game import Abba

import numpy as np
from collections import defaultdict
import math
import random


def state_to_str(state, pretty=False):
    seperator = '\n' if pretty else ''
    return seperator.join(''.join(row) for row in state)


class TDAgent:

    def __init__(self, n: int = 3, epsilon=0.1, learning_rate=0.1, discount_factor=0.9):
        self._state_values = defaultdict(lambda: 0)
        self.n = n
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def get_state_value(self, state):
        return self._state_values[state_to_str(state)]

    def update_state_value(self, state, value):
        self._state_values[state_to_str(state)] = value

    def policy(self, state):
        next_states = game.legal_next_states(state)
        best_value, best_state = -math.inf, None
        for next_state in next_states:
            next_state_value = self.get_state_value(next_state)
            if next_state_value > best_value:
                best_state = next_state
                best_value = next_state_value

        if random.random() < self.epsilon:
            return random.choice(next_states)
        else:
            return best_state

    def update_state_values(self, episode, reward):
        episode_length = len(episode)
        for step in range(episode_length - 1):
            state = episode[step]
            state_value = self.get_state_value(state)
            next_state = episode[step + 2] if step + 2 < episode_length else None
            next_state_value = self.get_state_value(next_state) if next_state is not None else 0

            if step >= episode_length - 3:
                step_reward = reward[step % 2]
            else:
                step_reward = 0

            updated_state_value = state_value + self.learning_rate * (
                    step_reward + self.discount_factor * next_state_value - state_value)

            self.update_state_value(episode[step], updated_state_value)


def state_space(n):
    if n == 0:
        return [""]
    results = []
    for c in ['A', 'B', '*']:
        for result in state_space(n-1):
            results.append(c + result)
    return results

def episode_generator(game: Abba, model: TDAgent):
    game.reset()

    episode = [] #[game.state()]

    while not game.is_terminal():
        state = game.state()
        new_state = model.policy(state)
        game.update_state(new_state)
        episode.append(new_state)

    return episode, game.rewards()

def benchmark(game: Abba, model: TDAgent, policy, n=100):
    wins, draws = 0, 0

    for _ in range(n):
        game.reset()
        while not game.is_terminal():
            state = game.state()



if __name__ == "__main__":
    from datetime import datetime

    N = 4
    runs = 100000
    agent = TDAgent(N, epsilon=0.2, learning_rate=0.5)
    game = Abba(N)
    for epoch in range(runs):

        episode, rewards = episode_generator(game, agent)

        if epoch % (runs / 10) == 0:
            print(f'{datetime.now()}: Epoch {epoch} / {runs}')
            states = [state_to_str(state, pretty=True) for state in episode]
            print(*states, sep='\n\n')
            print(rewards)
            pass

        agent.update_state_values(episode, rewards)
        pass

