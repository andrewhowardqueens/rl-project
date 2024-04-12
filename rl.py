from game import Abba

import numpy as np
from collections import defaultdict
import math
import random
from tdagent import TDAgent
from montecarlo import MonteCarloAgent


def state_to_str(state, pretty=False):
    seperator = '\n' if pretty else ''
    return seperator.join(''.join(row) for row in state)


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

    episode = []

    while not game.is_terminal():
        state = game.state()
        legal_actions = game.legal_actions()
        action = model.epsilon_greedy_policy(state, legal_actions)
        game.apply_action(action)
        episode.append((state, action))

    return episode, game.rewards()


def benchmark(game: Abba, model: TDAgent, policy, n=100):
    wins, draws = 0, 0

    for _ in range(n):
        game.reset()
        while not game.is_terminal():
            state = game.state()

def best_policy_game()

def main():
    from datetime import datetime

    N = 4
    runs = 100000
    agent = MonteCarloAgent(N, epsilon=0.2, learning_rate=0.1, discount_factor=1)
    game = Abba(N)
    epoch = 0
    while True:

        episode, rewards = episode_generator(game, agent)

        if epoch % 10000 == 0:
            print(f'{datetime.now()}: Epoch {epoch} / {runs}')
            states = [state_to_str(state, pretty=True)+'\n'+action for state, action in episode]
            print(*states, sep='\n\n')
            print(rewards)
            pass

        agent.update_action_values(episode, rewards)
        pass

        epoch += 1

    pass

def simple():
    from datetime import datetime

    N = 4
    agent = MonteCarloAgent(N, epsilon=0.1, learning_rate=0.1, discount_factor=0.9)
    game = Abba(N)

    episode, rewards = episode_generator(game, agent)

    states = [state_to_str(state, pretty=True)+'\n'+action for state, action in episode]
    print(*states, sep='\n\n')
    print(rewards)

    agent.update_action_values(episode, rewards)


def pickle_agent(agent):
    import pickle
    with open('sarsa-1353.pkl', 'wb') as f:
        pickle.dump({state: {action: value} for state, av in agent._action_values.items() for action, value in av.items()}, f)

if __name__ == "__main__":
    main()