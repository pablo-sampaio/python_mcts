
# Monte Carlo Tree Search (MCTS) in Python

This project offers a Python module with an extensible implementation of MCTS (Monte Carlo Tree Search).

## Description

This MCTS implemenattion can be used to find the best action for a single player or for multi-player competitive turn-based games with discrete actions.

Typical examples are two-player board games such as: checkers or chess. But it also can other games such as connect-4 or tic-tac-toe. Games with more than 2 players are also supported.

## Features

- Supports from 1 to multiple players
- Automatic random rollout policy
- Rollout policy can be replaced by custom implementation


## Using

To use this module: 
1. Copy "py_mcts" folder to your project's folder.
2. In your code, import `py_mcts`. You will use **`ProblemState`** and **`MCTS`** classes.
3. Create a new class to your problem state, as a subclass of `ProblemState`. Implement these methods:
   * get_player(self) -> in non-terminal states, returns the next player to move (use any type to identify players)
   * get_valid_actions(self) -> return a list or iterable of actions (represented by any type you want)
   * is_terminal(self) -> returns boolean to indicate if it's a terminal state
   * game_result(self, player) -> used in terminal states, indicates the score for any given player
   * move(self, action) -> indicates that `get_player()` applied the given `action`; should return a new state
4. Instantiate you MCTS solver. In the constructor:
   * you may set a custom `rollout_policy` if you want (default: random policy)
   * you may also set the `c` parameter of the tree policy (default: 1.41)
5. Then, whenever you need to find a "good" action in a given state your game:
   * instante your state class with the current state information
   * in your solver call **`choose_action(state,duration=30)`**
   * it will return the recommended action!!



## Example

This project includes an example of the implementation of a "AI" for tic-tac-toe, which you can play against.
