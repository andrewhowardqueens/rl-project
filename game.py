
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

        if self.is_terminal():
            self.solved = True
            return True
        else:
            self.player = (self.player + 1) % 2
            return False

    def returns(self):
        if not self.solved:
            raise RuntimeError('Game not terminated yet!')
        return self._returns
