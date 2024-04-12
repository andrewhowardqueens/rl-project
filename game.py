import numpy as np

class Abba:

    # Note: Only works for single digit Ns at the moment

    def __init__(self, n):
        self.n = n
        self.board = np.array([])
        self.player = 0
        self.solved = False
        self._rewards = None

        self.marks = ['A', 'B']
        self.empty_mark = '*'

        # Initialize board
        self.reset()

    def to_string(self):
        return ''.join(''.join(row) for row in self.board)

    def state(self):
        return np.array(self.board)

    def num_players(self):
        return 2

    def current_player(self):
        return self.player

    def reset(self):
        self._rewards = None
        self.solved = False
        self.player = 0
        self.board = np.array([[self.empty_mark for _ in range(self.n)] for _ in range(self.n)])

    def _board_solved(self, symbol):
        """Checks a sub-board to see if a player has won. Checked for 4x4 and three in a row required, not verified for
        other board sizes.
        board (list[list[str]]): 3x3 list representation of the board
        symbol (str): symbol of the player to check for
        returns a boolean indicating if the supplied symbol has a winning position
        """

        for i in range(self.n):
            for j in range(self.n - 2):

                # Check row
                if self.board[i][j] == self.board[i][j+1] == self.board[i][j+2] == symbol:
                    return True

                # Check column
                if self.board[j][i] == self.board[j+1][i] == self.board[j+2][i] == symbol:
                    return True

        for i in range(self.n - 2):
            for j in range(self.n - 2):

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
                    self._rewards = [1., -1.]
                else:
                    self._rewards = [-1., 1.]
                return True
        if all(self.board[i][j] != self.empty_mark
                for i in range(self.n)
                for j in range(self.n)):
            self._rewards = [0, 0]
            return True
        else:
            return False

    def update_state(self, new_state):

        # Ensure only one change
        if not np.sum(self.board != new_state) == 1:
            raise RuntimeError("Only one change allowed per move")

        # Ensure that it changed from an empty mark to an A or B
        change_indices = np.where((self.board == '*') & ((new_state == 'A') | (new_state == 'B')))

        # Check if there's exactly one difference
        if len(change_indices[0]) != 1:
            raise RuntimeError("Changed an existing mark")

        self.board = new_state.copy()

        if self.is_terminal():
            self.solved = True
            return True
        else:
            self.player = (self.player + 1) % 2
            return False

    def legal_next_states(self, state):
        legal_states = list()
        for i in range(self.n):
            for j in range(self.n):
                if self.board[i, j] == self.empty_mark:
                    for mark in self.marks:
                        new_state = state.copy()
                        new_state[i, j] = mark
                        legal_states.append(new_state)
        return legal_states

    def legal_actions(self, state=None):
        """Returns all the legal actions."""
        actions = list()
        board = self.board if state is None else state
        for i in range(self.n):
            for j in range(self.n):
                if board[i][j] == self.empty_mark:
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
            return True
        else:
            self.player = (self.player + 1) % 2
            return False

    def rewards(self):
        if not self.solved:
            raise RuntimeError('Game not terminated yet!')
        return self._rewards
