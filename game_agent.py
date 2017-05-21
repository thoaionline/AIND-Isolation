"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random

directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
              (1, -2), (1, 2), (2, -1), (2, 1)]

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def get_moving_area_for_player(game, player):
    """

    Parameters
    ----------
    game : `isolation.Board`
    player : CustomPlayer

    Returns
    -------
    (int,set)
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

    return border, search_space

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
    Heuristic based on the difference between number of steps that each player can take
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
    Combine the improved_score with real_steps_score after N moves

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
    Scoring heurstic based on the difference between the players' available moving area/space

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
    return float(player_step * len(possibilities) - opponent_step * len(opp_possibilities))

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

def get_moves_from_position(available_squares, position):
    """
    Find the available moves from `position`

    Parameters
    ----------
    available_squares: set
    position: (int,int)

    Returns
    -------
    set of (int,int)
        moves from specified position among the set
    """
    moves = set()
    for direction in directions:
        next_move = (position[0] + direction[0], position[1] + position[1])
        if next_move in available_squares:
            moves.add(next_move)
    return moves


def knight_heuristic(game, start_position):
    """
    Estimate the the longest path that a knight can take by moving on the most narrow path (least option 1-step ahead).
    This function is used in the knight_only_score heuristic.

    Parameters
    ----------
    game: `isolation.Board`
    start_position: (int,int)

    Returns
    -------
    int
        estimated longest path for a knight
    """

    if start_position == (-1, -1):
        return 0

    search_space = set(game.get_blank_spaces())

    longest_path = 0

    current_position = start_position
    next_move = start_position

    current_move_set = get_moves_from_position(search_space, current_position)
    next_move_set = set()

    while True:
        # print('Loop, current position is {}'.format(current_position))
        # Because we can move from current position
        longest_path += 1

        # Identify a move set with the most restricted paths
        current_choice = 9
        for move in current_move_set:
            this_move_set = get_moves_from_position(search_space, current_position)
            # Ensure that we always retain a move
            if len(this_move_set) < current_choice and current_choice < 9:
                # print('Replacing {} with {}'.format(next_move_set, this_move_set))
                # print('Next move is set to {}'.format(move))
                # print('Because {} < {}'.format(len(this_move_set), current_choice))
                next_move_set = this_move_set
                next_move = move
                current_choice = len(this_move_set)

        # Make the move
        if current_choice < 9:
            search_space.remove(next_move)
            current_move_set = next_move_set
            current_position = next_move
        else:
            break

    return longest_path


def game_is_partitioned(game):
    """
    Detect whether the game board is partitioned.

    Parameters
    ----------
    game: `isolation.Board`

    Returns
    -------
    bool
        True if game board is partition, False otherwise
    """
    _, area1 = get_moving_area_for_player(game, game.active_player)
    _, area2 = get_moving_area_for_player(game, game.inactive_player)

    return len(area1.intersection(area2)) > 0


def knight_only_score(game, player):
    """
    Knight's movement heuristic, with fallback to deep search when partitioning is detected.

    Parameters
    ----------
    game: `isolation.Board`
    player

    Returns
    -------
    float
        game score from current player's perspective
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    if game_is_partitioned(game):
        my_score = knight_heuristic(game, game.get_player_location(player))
        opp_score = knight_heuristic(game, game.get_player_location(game.get_opponent(player)))
        if my_score > opp_score:
            return float('inf')
        else:
            return float('-inf')
    else:
        own_moves = len(game.get_legal_moves(player))
        opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
        return float(own_moves - opp_moves)


def meta_score(ratio):
    return lambda game, player: smart_score(game, player, ratio)

def smart_score(game, player, ratio=1):
    """
    Experimental heuristic tha run a heuristic only after ration*N steps, to be called via meta_score
    This function is intended for experimentation only and should be disrecarded in the final submission/evaluation.

    Parameters
    ----------
    game: `isolation.Board`
    player

    Returns
    -------
    float
        score from the perspective of `player`
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    if (game.move_count > min(game.width, game.height) * 2) and game_is_partitioned(game):
        my_score = knight_heuristic(game, game.get_player_location(player))
        opp_score = knight_heuristic(game, game.get_player_location(game.get_opponent(player)))
        if my_score > opp_score:
            return float('inf')
        else:
            return float('-inf')
    else:
        own_moves = len(game.get_legal_moves(player))
        opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
        return float(own_moves - opp_moves)

custom_score = knight_only_score
custom_score_2 = real_steps_score
custom_score_3 = combined_score

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

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

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        (score, best_move) = self.minimax_with_score(game, depth, True)

        return best_move

    def minimax_with_score(self, game, depth, maximizing_player=True):
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
            raise SearchTimeout()

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
            sub_score, _ = self.minimax_with_score(sub_game, depth - 1, not maximizing_player)

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


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

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

        legal_moves = game.get_legal_moves(game.active_player)

        if len(legal_moves) == 0:
            return (-1, -1)

        best_score, best_move = float("-inf"), legal_moves[0]

        try:

            max_depth = 1
            while True:

                if self.time_left() < self.TIMER_THRESHOLD:
                    raise SearchTimeout()

                best_move = self.alphabeta(game, max_depth)
                max_depth += 1

        except SearchTimeout:
            return best_move

        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

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

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        (score, best_move) = self.alphabeta_with_score(game, depth, alpha, beta, True)

        return best_move

    def alphabeta_with_score(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
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
            raise SearchTimeout()

        if depth == 0:
            score_fn = self.score
            return score_fn(game, self), (-1, -1)

        if maximizing_player:
            best_score, best_move = float('-inf'), (-1, -1)
            for move in game.get_legal_moves(self):

                sub_game = game.forecast_move(move)
                sub_score, _ = self.alphabeta_with_score(sub_game, depth - 1, alpha, beta, False)

                if sub_score >= best_score:
                    best_score, best_move = sub_score, move
                alpha = max(alpha, best_score)

                if beta <= alpha:
                    break

        else:
            best_score, best_move = float('inf'), (-1, -1)
            for move in game.get_legal_moves(game.get_opponent(self)):

                sub_game = game.forecast_move(move)
                sub_score, _ = self.alphabeta_with_score(sub_game, depth - 1, alpha, beta, True)

                if sub_score <= best_score:
                    best_score, best_move = sub_score, move
                beta = min(beta, best_score)

                if (beta <= alpha):
                    break

        return (best_score, best_move)
