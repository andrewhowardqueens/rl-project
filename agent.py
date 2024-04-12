from random import *
from copy import deepcopy
import numpy as np
from math import *
import matplotlib.pyplot as plt

class MCTSNode:
    def __init__(self, board, done, parent, observation, action_index):
          
        # child nodes
        self.children = None
        
        # total rewards from MCTS exploration
        self.totalReward = 0
        
        # visit count
        self.numVisits = 0        
                
        # the environment
        self.board = board
        
        # observation of the environment
        self.observation = observation
        
        # if game is won/loss/draw
        self.done = done

        # link to parent node
        self.parent = parent
        
        # action index that leads to this node
        self.action_index = action_index

    def getUCBscore(self):
        
        # Unexplored nodes have maximum values so we favour exploration
        if self.N == 0:
            return float('inf')
        
        # We need the parent node of the current node 
        top_node = self
        if top_node.parent:
            top_node = top_node.parent
            
        # We use one of the possible MCTS formula for calculating the node value
        return (self.totalReward / self.numVisits) + sqrt(2) * sqrt(log(top_node.numVisits) / self.numVisits) 
    
    def create_child(self):

        if self.done:
            return
        
        actions = []
        games = []

        print (self.board.legal_actions())
        for action in self.board.legal_actions():
            actions.append(action)
            newGame = deepcopy(self.board)
            games.append(newGame)
        
        children = {}
        for action, game in zip(actions, games):
            reward, obs = self.board.apply_action_return_state(action)
            isDone = self.board.is_terminal()
            children[action] = MCTSNode(game, isDone, self, obs, action)
        self.children = children
    
    def explore(self):
        current = self

        while current.children:
            children = current.children
            max_Score = max(c.getUCBscore() for c in children.values())
            actions = [ a for a,c in children.items() if c.getUCBscore() == max_Score ]
            if len(actions) == 0:
                print("No Actions Available! ", max_Score)                      
            action = random.choice(actions)
            current = children[action]
        if current.numVisits < 1:
            current.totalReward = current.totalReward + current.randomRollout()
        else:
            current.create_child()
            if current.children:
                current = random.choice(current.children)
            current.totalReward = current.totalReward + current.randomRollout()
            
        current.numVisits += 1      
                
        # update statistics and backpropagate
            
        parent = current
            
        while parent.parent:
            
            parent = parent.parent
            parent.numVisits += 1
            parent.totalReward = parent.totalReward + current.totalReward

    def randomRollout(self):
        if self.done:
            return 0
        
        MCVal = 0
        done = False
        new_board = deepcopy(self.board)
        while not done:
            action = choice(new_board.legal_actions())
            reward = new_board.apply_action(action)
            MCVal += reward
            done = new_board.is_terminal()
            if done:
                new_board.reset()
                break             
        return MCVal
    
    def detach_parent(self):
        # free memory detaching nodes
        del self.parent
        self.parent = None

    def next(self):
    
        ''' 
        Once we have done enough search in the tree, the values contained in it should be statistically accurate.
        We will at some point then ask for the next action to play from the current node, and this is what this function does.
        There may be different ways on how to choose such action, in this implementation the strategy is as follows:
        - pick at random one of the node which has the maximum visit count, as this means that it will have a good value anyway.
        '''

        if self.done:
            raise ValueError("game has ended")

        if not self.child:
            raise ValueError('no children found and game hasn\'t ended')
        
        children = self.children
        
        max_N = max(node.N for node in children.values())
    
        max_children = [ c for a,c in children.items() if c.N == max_N ]
        
        if len(max_children) == 0:
            print("NO NEXT CHILD ", max_N) 
            
        max_child = random.choice(max_children)
        
        return max_child, max_children.action_index
    
MCTS_POLICY_EXPLORE = 50# MCTS exploring constant: the higher, the more reliable, but slower in execution time

def MCTS_Policy(node):  

    '''
    Our strategy for using the MCTS is quite simple:
    - in order to pick the best move from the current node:
        - explore the tree starting from that node for a certain number of iterations to collect reliable statistics
        - pick the node that, according to MCTS, is the best possible next action
    '''
    
    for i in range(MCTS_POLICY_EXPLORE):
        node.explore()
        
    next_node, next_action = node.next()
        
    # note that here we are detaching the current node and returning the sub-tree 
    # that starts from the node rooted at the choosen action.
    # The next search, hence, will not start from scratch but will already have collected information and statistics
    # about the nodes, so we can reuse such statistics to make the search even more reliable!
    node.detach_parent()
    
    return next_node, next_action



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
    
    def get_board(self):
        return self.board

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
    
    def step(self, action: str):
        if self.solved:
            raise RuntimeError('Game has already been completed.')
        elif action not in self.legal_actions():
            raise ValueError(f'{action} is not a legal move.')
        mark, i, j = action[0], int(action[2]), int(action[4])
        newBoard = deepcopy(self.board)
        newBoard[i][j] = mark


    def apply_action(self, action: str):
        if self.solved:
            raise RuntimeError('Game has already been completed.')
        elif action not in self.legal_actions():
            raise ValueError(f'{action} is not a legal move.')

        mark, i, j = action[0], int(action[2]), int(action[4])
        self.board[i][j] = mark
        if self._board_solved(mark):
            self.solved = True
            if self.player == 0:
                self._returns = [1.0, -1.0]
            else:
                self._returns = [-1.0, 1.0]
            return self._returns[self.player]  # Win: +1, Lose: -1
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
    
    def apply_action_return_state(self, action: str):

        if self.solved:
            raise RuntimeError('Game has already been completed.')
        elif action not in self.legal_actions():
            raise ValueError(f'{action} is not a legal move.')
        print(action)

        mark, i, j = action[0], int(action[2]), int(action[4])
        self.board[i][j] = mark

        # Check if the current move solved the board for the current player
        if self._board_solved(mark):
            self.solved = True
            if self.player == 0:
                self._returns = [1.0, -1.0], self.board
            else:
                self._returns = [-1.0, 1.0]
            return self._returns[self.player], self.board  # Win: +1, Lose: -1

        if self.is_terminal():
            self._returns = [0, 0]
            return 0, self.board  # Draw

        # Before switching players, check if the current move opens up a winning move for the opponent
        opponent_mark = 'B' if mark == 'A' else 'A'
        if self.can_win_next(opponent_mark):
            # Undo player switch and return a penalty since it opens up a win for the opponent
            self.player = (self.player - 1) % 2
            return -0.5, self.board  # Significant penalty for setting up the opponent to win

        # Switch players if no immediate win or loss caused by the move
        self.player = (self.player + 1) % 2
        return -0.1, self.board  # Small penalty for non-winning movesN

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
    def step(self, action):
        reward = self.apply_action(action)
        obs = self.board

    
game = Abba()
num_episodes = 50
rewards = []
moving_average = []

for episode in range(num_episodes):
    game.reset()
    state = game.to_string()
    total_reward = 0
    observation = game.get_board()
    done = False

    new_game = deepcopy(game)
    mytree = MCTSNode(new_game, False, 0, observation, 0)
    
    print('episode #' + str(e+1))
    
    while not done:
    
        mytree, action = MCTS_Policy(mytree)
        
        reward = game.apply_action(action)  
                        
        reward_e = reward_e + reward
        
        #game.render() # uncomment this if you want to see your agent in action!
                
        if done:
            print('reward_e ' + str(reward_e))
            game.close()
            break
        
    rewards.append(reward_e)
    moving_average.append(np.mean(rewards[-100:]))
    
plt.plot(rewards)
plt.plot(moving_average)
plt.show()