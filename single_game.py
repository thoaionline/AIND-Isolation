from isolation import Board

from sample_players import improved_score
# from game_agent import moving_area_score
from game_agent import *

from game_agent import CustomPlayer

# create an isolation board (by default 7x7)
player1 = CustomPlayer(score_fn=smart_score)
player2 = CustomPlayer(score_fn=improved_score)

game = Board(player1, player2)

# place player 1 on the board at row 2, column 3, then place player 2 on
# the board at row 0, column 5; display the resulting board state.  Note
# that .apply_move() changes the calling object
game.apply_move((2, 3))
game.apply_move((0, 5))
print(game.to_string())

# players take turns moving on the board, so player1 should be next to move
assert (player1 == game.active_player)

# get a list of the legal moves available to the active player
print(game.get_legal_moves())

# get a successor of the current state by making a copy of the board and
# applying a move. Notice that this does NOT change the calling object
# (unlike .apply_move()).
new_game = game.forecast_move((1, 1))
assert (new_game.to_string() != game.to_string())
print("\nOld state:\n{}".format(game.to_string()))
print("\nNew state:\n{}".format(new_game.to_string()))

# play the remainder of the game automatically -- outcome can be "illegal
# move" or "timeout"; it should _always_ be "illegal move" in this example
winner, history, outcome = game.play()
print("\nWinner: {}\nOutcome: {}".format(winner, outcome))
print('Winner: {}'.format("smart_score" if winner == player1 else 'improved_score()'))
print(game.to_string())
print("Move history:\n{!s}".format(history))
