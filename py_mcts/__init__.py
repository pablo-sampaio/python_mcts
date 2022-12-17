"""
Monte-Carlo Tree Search implementation.

This is major refactoring based on code from: https://ai-boson.github.io/mcts/
"""

import numpy as np
import time
import abc # abstract class


def _random_rollout_policy(state, possible_actions):
    return possible_actions[np.random.randint(len(possible_actions))]


class MCTS:
    def __init__(self, rollout_policy=_random_rollout_policy, c=1.41):
        self.root = None
        self.rollout_policy = rollout_policy
        self.c_param = c

    def choose_action(self, state, duration=30):
        self.root = _MCTS_Node(state)
        assert not self.root.is_terminal_node(), "terminal node - no action to choose"

        end_time = time.time() + duration 
        
        i = 0
        while time.time() < end_time:
            # select / expand
            new_node = self.root.tree_policy(self.c_param)
            
            # simulate
            terminal_state = new_node.simulate(self.rollout_policy)
            
            # back propagate
            new_node.backpropagate(terminal_state)

        print("Analyzed nodes: ", i)        
        #chosen_node = self.root.choose_child_uct(c_param=0.0)
        chosen_node = self.root.choose_child_visits()
        print("NODE with: q_sum =", chosen_node.q_sum, ", n =", chosen_node.n)

        return chosen_node.parent_action


class ProblemState(abc.ABC):

    def get_player(self): 
        '''
        Modify according to your game or needs. 
        Returns some value to identify the player which is in turn (to do actions from this state). 
        Returns None if it is a final state.
        Default implementation: return 0 for all non-terminal states, for single player problems.
        '''
        if self.is_terminal():
            return None
        return 0

    @abc.abstractmethod
    def get_valid_actions(self): 
        '''
        Modify according to your game or needs. 
        Constructs a list of all possible actions from current state.
        Returns a list.
        '''
        pass

    @abc.abstractmethod
    def is_terminal(self):
        '''
        Modify according to your game or needs. Game over condition. 
        Returns true or false.
        '''
        pass

    @abc.abstractmethod
    def final_result(self, player):
        '''
        Modify according to your game or needs.
        Can only be called in terminal state. 
        Returns the score for any given player.
        '''
        pass

    @abc.abstractmethod
    def move(self, action):
        '''
        Modify according to your game or needs. 
        Changes the state of your problem/game when the given action is applied. 
        Returns the new state after making a move.
        '''
        pass


class _MCTS_Node():
    def __init__(self, state, parent=None, parent_action=None):
        self.parent = parent
        self.state = state
        self.parent_action = parent_action
        self.children = []
        self.n = 0.0      # number of visits / simulations
        self.q_sum = 0.0  # sum of results (for the current player)
        self.untried_actions = self.state.get_valid_actions()

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.move(action)
        child_node = _MCTS_Node(next_state, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node 
    
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal_node(self):
        return self.state.is_terminal()

    def simulate(self, rollout_policy):
        current_state = self.state
        while not current_state.is_terminal():
            possible_actions = current_state.get_valid_actions()
            action = rollout_policy(current_state, possible_actions)
            current_state = current_state.move(action)
        return current_state

    def backpropagate(self, terminal_state):
        self.n += 1.0
        if self.parent is not None:
            self.q_sum += terminal_state.final_result(self.parent.state.get_player()) # because stats in this node refer to an action taken in the parent
            self.parent.backpropagate(terminal_state)

    def choose_child_uct(self, c_param):
        choices_weights = [(child.q_sum / child.n) + c_param * np.sqrt(np.log(self.n / child.n)) for child in self.children]
        return self.children[np.argmax(choices_weights)]

    def choose_child_visits(self):
        choices_weights = [child.n for child in self.children]
        return self.children[np.argmax(choices_weights)]

    def tree_policy(self, c_param=1.41):
        current_node = self
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.choose_child_uct(c_param)
        return current_node
