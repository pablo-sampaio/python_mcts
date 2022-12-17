
"""
An example implementation of AI with MCTS.
Based on https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1.
"""

from collections import namedtuple

from py_mcts import ProblemState, MCTS


_winning_combos = []
# three in a row
for start in range(0, 9, 3):
    _winning_combos.append((start, start+1, start+2))
# three in a column
for start in range(3):
    _winning_combos.append((start, start + 3, start + 6))
# down-right diagonal
_winning_combos.append((0, 4, 8))
# down-left diagonal
_winning_combos.append((2, 4, 6))


_TTTBoard = namedtuple("_TTTBoard", "tup turn x_score terminal")


# Inheriting from a namedtuple is convenient because it makes the class
# immutable and predefines __init__, __repr__, __hash__, __eq__, and others
class TicTacToeBoard(ProblemState, _TTTBoard):

    def get_player(self):
        return self.turn

    def get_valid_actions(self): 
        if self.terminal:  # if the game is finished then no moves can be made
            return []
        return [ i for (i, value) in enumerate(self.tup) if value is None ]

    def is_terminal(self):
        return self.terminal

    def final_result(self, player):
        assert self.terminal, f"Non-terminal board: {self.tup}"
        
        if self.x_score is None:
            return 0.0  # draw
        if player == 'X':
            return self.x_score 
        elif player == 'O':
            return -self.x_score
        assert False, f"Invalid player: {player}"

    def _find_winner(tup):
        "Returns None if no winner, +1 if X wins, -1 if O wins"
        for i1, i2, i3 in _winning_combos:
            v1, v2, v3 = tup[i1], tup[i2], tup[i3]
            if (v1 is not None) and (v1 == v2 == v3):
                return +1 if v1=='X' else -1
        return None
    
    def move(self, action):
        index = action
        tup = self.tup[:index] + (self.turn,) + self.tup[index+1:]
        
        x_score = TicTacToeBoard._find_winner(tup)
        is_terminal = (x_score is not None) or not any(v is None for v in tup) # there is a winner or board is full
        
        if is_terminal:
            turn = None
        else:
            turn = 'O' if self.turn=='X' else 'X'
        return TicTacToeBoard(tup, turn, x_score, is_terminal)

    def to_pretty_string(board):
        to_char = lambda v: ' ' if v==None else v
        rows = [
            [to_char(board.tup[3 * row + col]) for col in range(3)] for row in range(3)
        ]
        return (
            "\n  1 2 3\n"
            + "\n".join(str(i + 1) + " " + " ".join(row) for i, row in enumerate(rows))
            + "\n"
        )

def create_initial_board():
    # empty board, X's turn (1), no winner, not terminal
    return TicTacToeBoard(tup=(None,) * 9, turn='X', x_score=None, terminal=False)


if __name__ == "__main__":
    mcts = MCTS()
    board = create_initial_board()
    print(board.to_pretty_string())

    msg = input("Do you want to start? (y or Y for 'Yes') ").strip()
    if len(msg) > 0 and msg[0].lower()=='y':
        print("You play with X (starting piece)")
        human_piece = 'X'
        mcts_piece = 'O'
    else:
        print("You play with O")
        human_piece = 'O'
        mcts_piece = 'X'
    
    turn = 'X'

    while not board.terminal:
        if turn == human_piece:
            valid_move = False
            while not valid_move:
                row_col = input("enter row col: ").strip()
                row, col = map(int, row_col.split(" "))
                index = 3 * (row - 1) + (col - 1)
                valid_move = board.tup[index] is None
                if not valid_move:
                    print("Invalid move!")
            board = board.move(index)
            print(board.to_pretty_string())
            turn = mcts_piece
        else:
            # runs MCTS for 0.2 seconds
            index = mcts.choose_action(board, 0.2)
            board = board.move(index)
            print(board.to_pretty_string())
            turn = human_piece

    human_score = board.final_result(human_piece)
    mcts_score = board.final_result(mcts_piece)
    if human_score > mcts_score:
        print(f"You are the winner, playing with {human_piece}!")
    elif human_score < mcts_score:
        print(f"MCTS is the winner, playing with {mcts_piece}!")
    else:
        print("It's a draw!")
    
    print("Final score:")
    print(f"-> X : {board.final_result('X'):2}")
    print(f"-> O : {board.final_result('O'):2}")
