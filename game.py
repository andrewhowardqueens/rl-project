
import random
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, alpha=0.2, discount_factor=0.1, exploration_rate=0.2, min_exploration_rate=0.01):
        self.alpha = alpha
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.q_values = {}
        self.action_visits = {}  # Initialize action visits dictionary

    def get_q_value(self, state, action):
        return self.q_values.get((state, action), 0.0)

    def choose_action(self, state, legal_actions):
        # If it's the first move, select a random action
        if all(cell == ' ' for row in state for cell in row):
            action = random.choice(legal_actions)
        elif random.random() < self.exploration_rate:
            action = random.choice(legal_actions)
        else:
            q_values = [(action, self.get_q_value(state, action), self.action_visits.get((state, action), 0)) for action in legal_actions]
            q_values.sort(key=lambda x: (x[1], -x[2]))  # Sort by Q-value, then by inverse visit count
            action = q_values[-1][0]  # Select action with highest Q-value and lowest visit count

        self.action_visits[(state, action)] = self.action_visits.get((state, action), 0) + 1
        return action

    def update_q_values(self, state, action, reward, next_state, next_legal_actions):
        max_next_q_value = max([self.get_q_value(next_state, next_action) for next_action in next_legal_actions], default=0.0)
        td_target = reward + self.discount_factor * max_next_q_value
        td_error = td_target - self.get_q_value(state, action)
        self.q_values[(state, action)] = self.get_q_value(state, action) + self.alpha * td_error

class Abba:

    # Note: Only works for single digit Ns at the moment

    def __init__(self):
        self.N = 4
        self.board = list()
        self.player = 0
        self.solved = False
        self._returns = None

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
                self._returns = [1.0, -1.0]
            else:
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
            return -0.5  # Significant penalty for setting up the opponent to win

        # Switch players if no immediate win or loss caused by the move
        self.player = (self.player + 1) % 2
        return -0.1  # Small penalty for non-winning movesN

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

    def parse_action(self, action):
        """
        Parse an action string into its components.
        
        Args:
            action (str): An action string, e.g., 'A(1,2)'.
        
        Returns:
            tuple: A tuple containing the mark as a string and the row and column as integers.
        """
        mark = action[0]
        row, col = map(int, action[2:-1].split(','))
        return mark, row, col
    
    def returns(self):
        if not self.solved:
            raise RuntimeError('Game not terminated yet!')
        return self._returns
    
    def display(self):
        for row in self.board:
            print(' '.join(row))
        print()
    
game = Abba()
agent = QLearningAgent()

num_episodes = 1500
rewards = []

for episode in range(num_episodes):
    game.reset()
    state = game.to_string()
    total_reward = 0
    
    while not game.is_terminal():
        action = agent.choose_action(state, game.legal_actions())
        reward = game.apply_action(action)
        next_state = game.to_string()
        agent.update_q_values(state, action, reward, next_state, game.legal_actions())
        state = next_state
        total_reward += reward
        
    rewards.append(total_reward)
    print(f'Episode {episode + 1}, Total Reward: {total_reward}')

plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('QLearning Agent Learning Progress')
plt.show()

for _ in range(10):  # Let the agent play 10 games
    game.reset()
    state = game.to_string()
    
    while not game.is_terminal():
        action = agent.choose_action(state, game.legal_actions())
        game.apply_action(action)
        state = game.to_string()
    
    game.display()
    result = "Win" if game.returns()[0] == 1.0 else "Loss" if game.returns()[0] == -1.0 else "Draw"
    print(f"Game result: {result}")
# num_episodes = 500000
# for episode in range(num_episodes):
#     game.reset()
#     state = game.to_string()
#     while not game.is_terminal():
#         action = agent.choose_action(state, game.legal_actions())
#         game.apply_action(action)
#         next_state = game.to_string()
#         reward = game.returns()[0] if game.is_terminal() else 0.0
#         agent.update_q_values(state, action, reward, next_state, game.legal_actions())
#         state = next_state

agent.exploration_rate = 0

play_again = 'Y'
while play_again.upper() == 'Y':
    game.reset()
    while not game.is_terminal():
        game.display()
        if game.current_player() == 0:  # Agent's turn
            print(agent.q_values)
            action = agent.choose_action(game.to_string(), game.legal_actions())
            game.apply_action(action)
        else:  # Your turn
            action = None
            while action not in game.legal_actions():
                action = input("Enter your move (e.g., 'A(1,2)'): ")
            game.apply_action(action)
    game.display()

    returns = game.returns()
    if returns[0] > returns[1]:
        print("Game over! The agent wins!")
    elif returns[0] < returns[1]:
        print("Game over! You win!")
    else:
        print("Game over! It's a draw!")
    
    play_again = input("Do you want to play again? (Y/N): ")
