"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random

directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
              (1, -2), (1, 2), (2, -1), (2, 1)]


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def get_moving_area_for_player(game, player):
    """

    Parameters
    ----------
    game : `isolation.Board`
    player : CustomPlayer

    Returns
    -------
    (int,int)
        * distance from border to initial location
        * the area that this player reach
    """

    remaining = set(game.get_blank_spaces())
    player_location = game.get_player_location(player)

    next = set([player_location])
    search_space = set()

    # We may not be able to go through all locations, let's narrow the search space
    has_move = True
    border = 0
    while has_move:
        has_move = False
        current_round = next
        next = set()
        for starting_position in current_round:
            for direction in directions:
                next_position = (starting_position[0] + direction[0], starting_position[1] + starting_position[1])
                if next_position in remaining:
                    remaining.remove(next_position)
                    next.add(next_position)
                    search_space.add(next_position)
                    has_move = True
        if has_move:
            border += 1

    return border, len(search_space)


def get_max_step_for_player(game, player):
    """
    Because a knight's movement follows a bipatite graph, we find the 2 sub graphs and assign findings there. The
    maximum numbers of moves is limited to 2x the size of the smaller set, and a bonus step if there are more positions
    in the even set

    http://mathworld.wolfram.com/KnightGraph.html
    Parameters
    ----------
    game
    player

    Returns
    -------
    (int,set)
        upper limit of number of steps we can take
    """
    remaining = set(game.get_blank_spaces())
    player_location = game.get_player_location(player)

    next = set([player_location])
    moving_area = set()

    # We may not be able to go through all locations, let's narrow the search space
    has_move = True

    odd_steps = 0
    even_steps = 0

    is_odd_step = True

    while has_move:
        has_move = False
        current_round = next
        next = set()
        for starting_position in current_round:
            for direction in directions:
                next_position = (starting_position[0] + direction[0], starting_position[1] + starting_position[1])
                if next_position in remaining:
                    remaining.remove(next_position)
                    next.add(next_position)
                    moving_area.add(next_position)

                    if is_odd_step:
                        odd_steps += 1
                    else:
                        even_steps += 1

                    has_move = True

    max_steps = min(odd_steps, even_steps)

    # Bonus step
    if even_steps > odd_steps:
        max_steps += 1

    return max_steps, moving_area


def real_steps_score(game, player):
    """

    Parameters
    ----------
    game : `isolation.Board`
    player

    Returns
    -------

    """
    if game.is_winner(player):
        return float('inf')
    elif game.is_loser(player):
        return float('-inf')

    opponent = game.get_opponent(player)

    max_p_steps, area = get_max_step_for_player(game, player)
    max_o_steps, o_area = get_max_step_for_player(game, opponent)

    score = max_p_steps - max_o_steps

    # If it's player's turn, deduct 0.5 point for the disadvantage
    if game.active_player == player:
        score -= 0.5

    # If there is partition, search for end game
    if len(area.intersection(o_area)):
        # To do: a proper end-game search
        if max_p_steps > max_o_steps:
            return float('inf')
        elif max_p_steps < max_o_steps:
            return float('-inf')

    return score


def combined_score(game, player):
    """

    Parameters
    ----------
    game: `isolation.Board`
    player

    Returns
    -------
    float
    """

    # Let's be a bit greedy during first half of the game
    if game.move_count < (game.width + game.height / 2):
        return improved_score(game, player) * 5

    # Decision time
    return real_steps_score(game, player)


def moving_area_score(game, player):
    """

    Parameters
    ----------
    game: `isolation.Board`
    player: CustomPlayer

    Returns
    -------
    float
        difference in number of possible steps

    """

    if game.is_winner(player):
        return float('inf')
    elif game.is_loser(player):
        return float('-inf')

    opponent = game.get_opponent(player)
    player_step, possibilities = get_moving_area_for_player(game, player)
    opponent_step, opp_possibilities = get_moving_area_for_player(game, opponent)
    return float(player_step * possibilities - opponent_step * opp_possibilities)


def improved_score(game, player):
    """The "Improved" evaluation function discussed in lecture that outputs a
    score equal to the difference in the number of moves available to the
    two players.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves)


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    my_moves_count = len(game.get_legal_moves(player))
    if my_moves_count == 0:
        return float('-inf')
    else:
        opponent_moves_count = len(game.get_legal_moves(game.get_opponent(player)))
        if opponent_moves_count == 0:
            return float('inf')
        else:
            return float(my_moves_count - opponent_moves_count)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=20.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        if len(legal_moves) == 0:
            return (-1, -1)

        best_score, best_move = float("-inf"), legal_moves[0]
        search_method = getattr(self, self.method)

        if not self.iterative:
            best_score, best_move = search_method(game, self.search_depth)
            return best_move

        try:
            max_depth = 1
            while True:
                best_score, best_move = search_method(game, max_depth)
                if best_score == float('inf'):
                    return best_move
                max_depth += 1

        except Timeout:
            return best_move

        return best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Terminal node
        if depth == 0:
            score_fn = self.score
            return score_fn(game, self), (-1, -1)

        # Going to the next level
        best_score = float('-inf') if maximizing_player else float('inf')
        best_move = (-1, -1)

        posible_moves = game.get_legal_moves()
        for move in posible_moves:
            sub_game = game.forecast_move(move)
            sub_score, _ = self.minimax(sub_game, depth - 1, not maximizing_player)

            if maximizing_player:
                if sub_score > best_score:
                    best_move = move
                    best_score = sub_score
                    if sub_score == float('inf'):
                        return best_score, best_move
            else:
                if sub_score < best_score:
                    best_move = move
                    best_score = sub_score
                    if sub_score == float('-inf'):
                        return best_score, best_move

        return (best_score, best_move)

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if depth == 0:
            score_fn = self.score
            return score_fn(game, self), (-1, -1)

        if maximizing_player:
            best_score, best_move = float('-inf'), (-1, -1)
            for move in game.get_legal_moves(self):

                sub_game = game.forecast_move(move)
                sub_score, _ = self.alphabeta(sub_game, depth - 1, alpha, beta, False)

                if sub_score >= best_score:
                    best_score, best_move = sub_score, move
                alpha = max(alpha, best_score)

                if beta <= alpha:
                    break

            return (best_score, best_move)

        else:
            best_score, best_move = float('inf'), (-1, -1)
            for move in game.get_legal_moves(game.get_opponent(self)):

                sub_game = game.forecast_move(move)
                sub_score, _ = self.alphabeta(sub_game, depth - 1, alpha, beta, True)

                if sub_score <= best_score:
                    best_score, best_move = sub_score, move
                beta = min(beta, best_score)

                if (beta <= alpha):
                    break
            return (best_score, best_move)
