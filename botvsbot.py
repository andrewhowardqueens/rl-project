import random
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import seaborn as sns
import numpy as np


class QLearningAgent:
    def __init__(self, alpha=0.2, discount_factor=0.9, exploration_rate=0.5, min_exploration_rate=0.01):
        self.alpha = alpha
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.q_values = {}
        self.action_visits = {}  # Initialize action visits dictionary

    def get_q_value(self, state, action):
        return self.q_values.get((state, action), 0.0)

    def choose_action(self, game, state, legal_actions):
        best_action = None
        highest_q_value = float('-inf')
        safe_actions = []

        # First, check if there's an action that guarantees a win
        for action in legal_actions:
            mark, i, j = game.parse_action(action)
            game.board[i][j] = mark  # Temporarily apply action
            if game.check_for_win(game.board, i, j):
                game.board[i][j] = game.empty_mark  # Undo the action
                return action  # Take immediate win
            game.board[i][j] = game.empty_mark  # Undo the action

        # If no winning action, check if a move will allow the opponent to win on the next turn
        for action in legal_actions:
            mark, i, j = game.parse_action(action)
            game.board[i][j] = mark  # Temporarily apply action

            current_q_value = self.get_q_value(state, action)

            # Simulate opponent's moves to evaluate risk
            opponent_risk = 0  # Lower is better
            for opp_action in game.legal_actions():
                temp_mark, opp_i, opp_j = game.parse_action(opp_action)
                game.board[opp_i][opp_j] = temp_mark  # Opponent's turn
                if game.check_for_win(game.board, opp_i, opp_j):
                    opponent_risk += 1  # Opponent Wins
                    game.board[opp_i][opp_j] = game.empty_mark
                    break
                game.board[opp_i][opp_j] = game.empty_mark  # Undo opponent's move

            # Only consider this action if it doesn't allow an immediate win for the opponent
            if opponent_risk == 0:  # This means no immediate threat was found
                if current_q_value > highest_q_value:
                    highest_q_value = current_q_value
                    best_action = action
            else:
                safe_actions.append(action)

            game.board[i][j] = game.empty_mark  # Finally, reset the original action spot

        # If no safe actions, choose randomly
        if not best_action:
            return random.choice(legal_actions)

        return best_action
        
    def update_q_values(self, state, action, reward, next_state, next_legal_actions):
        # Get the maximum Q-value for the next state
        max_next_q_value = max([self.get_q_value(next_state, next_action) for next_action in next_legal_actions], default=0.0)
        # Compute the TD target and error
        td_target = reward + self.discount_factor * max_next_q_value
        td_error = td_target - self.get_q_value(state, action)
        # Update the Q-value for the current state and action
        self.q_values[(state, action)] = self.get_q_value(state, action) + self.alpha * td_error

class SarsaAgent:
    def __init__(self, alpha=0.2, discount_factor=0.9, exploration_rate=0.5, min_exploration_rate=0.01):
        self.alpha = alpha
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.q_values = {}

    def get_q_value(self, state, action):
        return self.q_values.get((state, action), 0.0)

    def choose_action(self, game, state, legal_actions):
        # Exploration: choose a random action
        if game.is_terminal():
            return None
        if random.random() < self.exploration_rate:
            return random.choice(legal_actions)
        # Exploitation: choose the action with the highest Q-value
        q_values = [self.get_q_value(state, action) for action in legal_actions]
        max_q_value = max(q_values)
        # If multiple actions have the same max Q-value, choose randomly among them
        best_actions = [action for action, q_value in zip(legal_actions, q_values) if q_value == max_q_value]
        return random.choice(best_actions)

    def update_q_values(self, state, action, reward, next_state, next_action):
        # Get the Q-value for the next state and action
        next_q_value = self.get_q_value(next_state, next_action)
        # Compute the TD target and error
        td_target = reward + self.discount_factor * next_q_value
        td_error = td_target - self.get_q_value(state, action)
        # Update the Q-value for the current state and action
        self.q_values[(state, action)] = self.get_q_value(state, action) + self.alpha * td_error

class QLearningAgent2:
    def __init__(self, alpha=0.2, discount_factor=0.9, exploration_rate=0.5, min_exploration_rate=0.01):
        self.alpha = alpha
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.q_values = {}
        self.action_visits = {}  # Initialize action visits dictionary

    def get_q_value(self, state, action):
        return self.q_values.get((state, action), 0.0)
    
    def choose_action(self, game, state, legal_actions):

        # Exploitation: choose the action with the highest Q-value
        q_values = [self.get_q_value(state, action) for action in legal_actions]
        max_q_value = max(q_values)
        # If multiple actions have the same max Q-value, choose randomly among them
        best_actions = [action for action, q_value in zip(legal_actions, q_values) if q_value == max_q_value]
        return random.choice(best_actions)
        
    def update_q_values(self, state, action, reward, next_state, next_legal_actions):
        # Get the maximum Q-value for the next state
        max_next_q_value = max([self.get_q_value(next_state, next_action) for next_action in next_legal_actions], default=0.0)
        # Compute the TD target and error
        td_target = reward + self.discount_factor * max_next_q_value
        td_error = td_target - self.get_q_value(state, action)
        # Update the Q-value for the current state and action
        self.q_values[(state, action)] = self.get_q_value(state, action) + self.alpha * td_error

class MonteCarloAgent:
    def __init__(self, discount_factor=0.9):
        self.discount_factor = discount_factor
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(float)
        self.policy = defaultdict(float)

    def choose_action(self, state, legal_actions):
        # Choose the action with the highest average return
        q_values = [self.policy[(state, action)] for action in legal_actions]
        max_q_value = max(q_values)
        best_actions = [action for action, q_value in zip(legal_actions, q_values) if q_value == max_q_value]
        return random.choice(best_actions)

    def update_policy(self, episode):
        # Calculate the return for each state-action pair in the episode
        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = self.discount_factor * G + reward
            sa = (state, action)
            if sa not in [(x[0], x[1]) for x in episode[0:t]]:
                self.returns_sum[sa] += G
                self.returns_count[sa] += 1.0
                self.policy[sa] = self.returns_sum[sa] / self.returns_count[sa]


class Abba:

    # Note: Only works for single digit Ns at the moment

    def __init__(self):
        self.N = 4
        self.board = list()
        self.player = 0
        self.solved = False
        self._returns = None
        self.win_count = 0
        self.loss_count = 0

        self.marks = ['A', 'B']
        self.empty_mark = '*'

        # Initialize board
        self.reset()

    def to_string(self):
        return ''.join(''.join(row) for row in self.board)

    def num_players(self):
        return 2

    def current_player(self):
        return self.player

    def reset(self):
        self._returns = None
        self.solved = False
        self.player = 0
        self.board = [[self.empty_mark for _ in range(self.N)] for _ in range(self.N)]

    def _board_solved(self, symbol):
        """Checks a sub-board to see if a player has won. Checked for 4x4 and three in a row required, not verified for
        other board sizes.
        board (list[list[str]]): 3x3 list representation of the board
        symbol (str): symbol of the player to check for
        returns a boolean indicating if the supplied symbol has a winning position
        """

        for i in range(self.N):
            for j in range(self.N - 2):

                # Check row
                if self.board[i][j] == self.board[i][j+1] == self.board[i][j+2] == symbol:
                    return True

                # Check column
                if self.board[j][i] == self.board[j+1][i] == self.board[j+2][i] == symbol:
                    return True

        for i in range(self.N - 2):
            for j in range(self.N - 2):

                # Check diagonal
                if self.board[i][j] == self.board[i+1][j+1] == self.board[i+2][j+2] == symbol:
                    return True

                # Check anti-diagonal
                if self.board[-i-1][j] == self.board[-i-2][j+1] == self.board[-i-3][j+2] == symbol:
                    return True

        return False

    def is_terminal(self):
        for mark in self.marks:
            if self._board_solved(mark):
                if self.player == 0:
                    self._returns = [1., -1.]
                else:
                    self._returns = [-1., 1.]
                return True
        if all(self.board[i][j] != self.empty_mark
                for i in range(self.N)
                for j in range(self.N)):
            self._returns = [0, 0]
            return True
        else:
            return False

    def legal_actions(self):
        """Returns all the legal actions."""
        actions = list()
        for i in range(self.N):
            for j in range(self.N):
                if self.board[i][j] == self.empty_mark:
                    actions += [f'{mark}({i},{j})' for mark in self.marks]
        return actions
    
    def apply_action(self, action: str):
        if self.solved:
            raise RuntimeError('Game has already been completed.')
        elif action not in self.legal_actions():
            raise ValueError(f'{action} is not a legal move.')

        mark, i, j = action[0], int(action[2]), int(action[4])
        self.board[i][j] = mark

        # Check if the current move solved the board for the current player
        if self._board_solved(mark):
            self.solved = True
            if self.player == 0:
                self.win_count += 1
                self._returns = [1.0, -1.0]
            else:
                self.loss_count += 1
                self._returns = [-1.0, 1.0]
            return self._returns[self.player]  # Win: +1, Lose: -1

        if self.is_terminal():
            self._returns = [0, 0]
            return 0  # Draw

        # Before switching players, check if the current move opens up a winning move for the opponent
        opponent_mark = 'B' if mark == 'A' else 'A'
        if self.can_win_next(opponent_mark):
            # Undo player switch and return a penalty since it opens up a win for the opponent
            self.player = (self.player - 1) % 2
            return 0  # Significant penalty for setting up the opponent to win

        # Switch players if no immediate win or loss caused by the move
        self.player = (self.player + 1) % 2
        return 0  # Small penalty for non-winning movesN

    def can_win_next(self, mark):
        """Check if the given player can win in their next move."""
        for action in self.legal_actions():
            _, i, j = self.parse_action(action)
            self.board[i][j] = mark
            if self._board_solved(mark):
                self.board[i][j] = self.empty_mark  # Undo the move
                return True
            self.board[i][j] = self.empty_mark  # Undo the move
        return False
    
    def check_for_win(self, board, i, j):
        for mark in ['A', 'B']:
            self.board[i][j] = mark
            if self._board_solved(mark):
                self.board[i][j] = self.empty_mark  # Undo the move
                return True
        self.board[i][j] = self.empty_mark  # Undo the move
        return False
        
    def simulate_action(self, current_board, action):
    # Create a deep copy of the board
        simulated_board = self.board.copy()
        mark, i, j = self.parse_action(action)
        simulated_board[i][j] = mark
        return simulated_board

    def parse_action(self, action):
        mark, rest = action[0], action[2:-1]
        i, j = map(int, rest.split(','))
        return mark, i, j
    
    def state_to_board(self, state):
        board = []
        for i in range(0, len(state), self.N):
            board.append(list(state[i:i+self.N]))
        return board
        
    def returns(self):
        if not self.solved:
            raise RuntimeError('Game not terminated yet!')
        return self._returns
    
    def display(self):
        for row in self.board:
            print(' '.join(row))
        print()
    
game = Abba()
# Create two agents
agent1 = QLearningAgent()
agent2 = QLearningAgent2()

num_episodes = 10000

print("Starting training...")

# Train agent1

q1_learning_rewards = []
for episode in range(num_episodes):
    #print(f"Starting episode {episode + 1}...")
    game.reset()
    state = game.to_string()  # Assuming this method converts the board to a string representation
    total_reward = 0

    while not game.is_terminal():
        legal_actions = game.legal_actions()  # Retrieve legal actions from the game instance
        action = agent1.choose_action(game, state, legal_actions)  # Pass the game instance
        reward = game.apply_action(action)  # Apply the chosen action to the game
        total_reward += reward
        next_state = game.to_string()  # Get new state after the action
        agent1.update_q_values(state, action, reward, next_state, game.legal_actions())
        state = next_state  # Update the state for the next 

    q1_learning_rewards.append(total_reward)

    if (episode + 1) % 1000 == 0:  # Print progress every 1000 episodes
        print(f"Agent1: Completed {episode + 1} / {num_episodes} episodes")

# Train agent2
q2_learning_rewards = []

for episode in range(num_episodes):
    game.reset()
    state = game.to_string()  # Assuming this method converts the board to a string representation
    total_reward = 0


    while not game.is_terminal():
        legal_actions = game.legal_actions()  # Retrieve legal actions from the game instance
        action = agent2.choose_action(game, state, legal_actions)  # Pass the game instance
        reward = game.apply_action(action)  # Apply the chosen action to the game
        total_reward += reward
        next_state = game.to_string()  # Get new state after the action
        agent2.update_q_values(state, action, reward, next_state, game.legal_actions())
        state = next_state  # Update the state for the next 

    q2_learning_rewards.append(total_reward)

    if (episode + 1) % 1000 == 0:  # Print progress every 1000 episodes
        print(f"Agent2: Completed {episode + 1} / {num_episodes} episodes")

# Let agent1 and agent2 play against each other
num_games = 100
win_count = 0
loss_count = 0
win_loss_ratios = []
agent1.exploration_rate = 0
agent2.exploration_rate = 0
q1_losses = 0
q1_wins = 0
for game_num in range(num_games):  # Let the agents play 100 games
    game.reset()
    state = game.to_string()
    
    while not game.is_terminal():
        if game.current_player() == 0:  # Agent1's turn
            action = agent1.choose_action(game, state, game.legal_actions())
        else:  # Agent2's turn
            action = agent2.choose_action(game, state, game.legal_actions())
        game.apply_action(action)
        state = game.to_string()
    
    result = game.returns()[0]
    if result == 1.0:
        q1_wins += 1
        win_count += 1
    elif result == -1.0:
        q1_losses += 1
        loss_count += 1

    win_loss_ratio = float(win_count) / (game_num + 1)  # Use the +1 as the first game has game_num = 0
    win_loss_ratios.append(win_loss_ratio)

# Plot the win/loss ratio
plt.plot(win_loss_ratios)
plt.xlabel('Game')
plt.ylabel('Win Rate')
plt.title('Win Rate Over 100 Games VS Q-Learning')
plt.show()

sarsa_agent = SarsaAgent()

# Train the Sarsa agent
sarsa_learning_rewards = []

for episode in range(num_episodes):
    game.reset()
    state = game.to_string()
    action = sarsa_agent.choose_action(game, state, game.legal_actions())
    total_reward = 0

    while not game.is_terminal():
        reward = game.apply_action(action)
        total_reward += reward
        next_state = game.to_string()
        next_action = sarsa_agent.choose_action(game, next_state, game.legal_actions())
        sarsa_agent.update_q_values(state, action, reward, next_state, next_action)
        state, action = next_state, next_action

    sarsa_learning_rewards.append(total_reward)

    if (episode + 1) % 1000 == 0:
        print(f"Sarsa Agent: Completed {episode + 1} / {num_episodes} episodes")

# Let your model and the Sarsa agent play against each other
win_count = 0
loss_count = 0
win_loss_ratios = []
sarsa_wins = 0
sarsa_losses = 0
sarsa_agent.exploration_rate = 0
for game_num in range(num_games):
    game.reset()
    state = game.to_string()

    while not game.is_terminal():
        if game.current_player() == 0:  # Your model's turn
            action = agent1.choose_action(game, state, game.legal_actions())
        else:  # Sarsa agent's turn
            action = sarsa_agent.choose_action(game, state, game.legal_actions())
        game.apply_action(action)
        state = game.to_string()

    result = game.returns()[0]
    if result == 1.0:
        win_count += 1
        sarsa_wins += 1
    elif result == -1.0:
        loss_count += 1
        sarsa_losses += 1

    win_loss_ratio = float(win_count) / (game_num + 1)  # Use the +1 as the first game has game_num = 0
    win_loss_ratios.append(win_loss_ratio)

# Plot the win/loss ratio
plt.plot(win_loss_ratios)
plt.xlabel('Game')
plt.ylabel('Win Rate')
plt.title('Win Rate Over 100 Games VS SARSA')
plt.show()

# Create a Monte Carlo agent
mc_agent = MonteCarloAgent()

# Train the Monte Carlo agent
MC_learning_rewards = []

for episode in range(num_episodes):
    game.reset()
    state = game.to_string()
    episodes = []
    total_reward = 0
    while not game.is_terminal():
        action = mc_agent.choose_action(state, game.legal_actions())
        reward = game.apply_action(action)
        total_reward += reward
        episodes.append((state, action, reward))
        state = game.to_string()
    mc_agent.update_policy(episodes)

    MC_learning_rewards = []


    if (episode + 1) % 1000 == 0:
        print(f"Monte Carlo Agent: Completed {episode + 1} / {num_episodes} episodes")

# Let your agent and the Monte Carlo agent play against each other
win_count = 0
loss_count = 0
win_loss_ratios = []
mc_wins = 0
mc_losses = 0

for game_num in range(num_games):
    game.reset()
    state = game.to_string()

    while not game.is_terminal():
        if game.current_player() == 0:  # Your agent's turn
            action = agent1.choose_action(game, state, game.legal_actions())
        else:  # Monte Carlo agent's turn
            action = mc_agent.choose_action(state, game.legal_actions())
        game.apply_action(action)
        state = game.to_string()

    result = game.returns()[0]
    if result == 1.0:
        win_count += 1
        mc_wins += 1    
    elif result == -1.0:
        loss_count += 1
        mc_losses += 1

    win_loss_ratio = float(win_count)  / (game_num + 1)  # Use the +1 as the first game has game_num = 0
    win_loss_ratios.append(win_loss_ratio)

# Plot the win/loss ratio
plt.plot(win_loss_ratios)
plt.xlabel('Game')
plt.ylabel('Win Rate')
plt.title('Win Rate Over 100 Games VS Monte-Carlo')
plt.show()

# plt.plot(q2_learning_rewards, label='Q-Learning')
# plt.plot(sarsa_learning_rewards, label='Sarsa')
# plt.plot(MC_learning_rewards, label='Monte Carlo')
# plt.plot(q1_learning_rewards, label='Improved Q-Learning')
# plt.xlabel('Episode')
# plt.ylabel('Total Reward')
# plt.title('Learning Curves')
# plt.legend()
# plt.show()

# Create a DataFrame with the win and loss counts
df = pd.DataFrame({
    'Model': ['Sarsa', 'Monte Carlo', 'Q-Learning'],
    'LBQL Wins': [sarsa_wins, mc_wins, q1_wins],
    'LBQL Losses': [sarsa_losses, mc_losses, q1_losses]
})

# Melt the DataFrame into a format suitable for a confusion matrix
df_melted = pd.melt(df, id_vars='Model', var_name='Outcome', value_name='Count')

# Create the confusion matrix
# Create the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(df_melted.pivot(index='Model', columns='Outcome', values='Count'), annot=True, fmt='d')
plt.title('Confusion Matrix of Improved Q-Learning(LBQL) VS Algorithms')
plt.show()