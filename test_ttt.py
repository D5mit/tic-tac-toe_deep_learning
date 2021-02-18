import numpy as np

from ttt import output_board
from ttt import get_move
from ttt import insert_letter
from ttt import play_move
from ttt import get_y
from ttt import is_winner
from ttt import load_model
from ttt import make_prediction_math
from ttt import create_df_training_stats

# todo to unit testing for these functions

# def get_training_stats(df_all_played_games, int_stats, inr):
# def print_agent_move(current_player, move_array, i_print_board):
# def iconv(idata, input):
# def boardToX(iboard):
# def isBoardFull(board):
# def make_prediction(iboard, player):
# def save_game_to_df(winner, GameNr, xPlayer, oPlayer, movesHist, i_pboard):
# def play_game(player_x, player_o, i_game_nr, i_print_board):
# def print_progress(total, counter, outputs, info):


# This function will test the following functions:
# def output_board(i_board, i_print_board, header=False):
# def get_move(player, let, curr_board):
# def insert_letter(letter, pos, i_game_nr, int_board, int_moves_hist):
# def play_move(let, player_x, player_o, i_game_nr, i_board, moves_hist, i_print_board=False):
# def get_y(move):
# def is_winner(bo, le):
# def create_df_training_stats():


# TEST test if winner -----------------------------------------------
# test test if winner 1
def test_is_winner_1():
    i_board = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
    assert(is_winner(i_board, 'X') is False)


# test test if winner 2
def test_is_winner_2():
    i_board = [' ', 'X', 'X', 'X', ' ', ' ', ' ', ' ', ' ', ' ']
    assert(is_winner(i_board, 'X') is True)


# test test if winner 3
def test_is_winner_3():
    i_board = [' ', 'O', ' ', 'X', 'O', 'X', ' ', 'O', ' ', ' ']
    assert(is_winner(i_board, 'O') is True)


# TEST get output move into array-----------------------------------------------
# test get_y move 1
def test_get_y_1():
    move = 1
    test_outcome = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
    assert(np.all(get_y(move) == test_outcome))


# test get_y move 2
def test_get_y_2():
    move = 2
    test_outcome = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0])
    assert(np.all(get_y(move) == test_outcome))


# test get_y move 3
def test_get_y_3():
    move = 3
    test_outcome = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0])
    assert(np.all(get_y(move) == test_outcome))


# test get_y move 4
def test_get_y_4():
    move = 4
    test_outcome = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0])
    assert(np.all(get_y(move) == test_outcome))


# test get_y move 5
def test_get_y_5():
    move = 5
    test_outcome = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])
    assert(np.all(get_y(move) == test_outcome))


# test get_y move 6
def test_get_y_6():
    move = 6
    test_outcome = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0])
    assert(np.all(get_y(move) == test_outcome))


# test get_y move 7
def test_get_y_7():
    move = 7
    test_outcome = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0])
    assert(np.all(get_y(move) == test_outcome))


# test get_y move 8
def test_get_y_8():
    move = 8
    test_outcome = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0])
    assert(np.all(get_y(move) == test_outcome))


# test get_y move 9
def test_get_y_9():
    move = 9
    test_outcome = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
    assert(np.all(get_y(move) == test_outcome))


# test full board
def test_output_board_1():
    i_board = [' ', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X']
    i_outboard = ['X|X|X', '-----', 'X|X|X', '-----', 'X|X|X']
    assert (output_board(i_board, True) == i_outboard)


# test board 2
def test_output_board_2():
    i_board = [' ', 'O', ' ', 'X', 'X', 'X', 'X', 'X', 'X', 'X']
    i_outboard = ['O| |X', '-----', 'X|X|X', '-----', 'X|X|X']
    assert (output_board(i_board, True) == i_outboard)


# test board 3
def test_output_board_3():
    i_board = [' ', 'X', ' ', 'X', 'X', 'X', 'X', 'X', ' ', 'X']
    i_outboard = ['X| |X', '-----', 'X|X|X', '-----', 'X| |X']
    assert (output_board(i_board, True) == i_outboard)


# test board 4 empty
def test_output_board_4():
    i_board = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
    i_outboard = [' | | ', '-----', ' | | ', '-----', ' | | ']
    assert (output_board(i_board, True) == i_outboard)


# TEST get_move-----------------------------------------------
# test untrained
def test_get_move_1():

    # load Agent Smit
    loaded_modelS = load_model('S')
    # load Agent Linki
    loaded_modelL = load_model('L')

    i_board = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
    imove, imove_array = get_move('U', 'X', i_board, loaded_modelS, loaded_modelL)
    assert (0 < imove < 10)


# test agent S
def test_get_move_2():

    # load Agent Smit
    loaded_modelS = load_model('S')
    # load Agent Linki
    loaded_modelL = load_model('L')

    i_board = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
    imove, imove_array = get_move('S', 'X', i_board, loaded_modelS, loaded_modelL)
    assert (0 < imove < 10)


# test agent L
def test_get_move_3():

    # load Agent Smit
    loaded_modelS = load_model('S')
    # load Agent Linki
    loaded_modelL = load_model('L')

    i_board = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
    imove, imove_array = get_move('L', 'X', i_board, loaded_modelS, loaded_modelL)
    assert (0 < imove < 10)


# TEST insert_letter-----------------------------------------------
# test insert letter 1
def test_insert_letter_1():
    let = 'X'
    move = 6
    i_game_nr = 1
    i_board = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
    moves_hist = []
    updated_board, updated_moves_hist = insert_letter(let, move, i_game_nr, i_board, moves_hist)
    assert (updated_board == [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' '])


# test insert letter 2
def test_insert_letter_2():
    let = 'O'
    move = 9
    i_game_nr = 1
    i_board = ['O', 'X', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ']
    moves_hist = []
    updated_board, updated_moves_hist = insert_letter(let, move, i_game_nr, i_board, moves_hist)
    assert (updated_board == ['O', 'X', ' ', ' ', ' ', 'X', ' ', ' ', ' ', 'O'])


# test insert letter 3
def test_insert_letter_3():
    let = 'X'
    move = 6
    i_game_nr = 1
    i_board = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
    moves_hist = []
    updated_board, updated_moves_hist = insert_letter(let, move, i_game_nr, i_board, moves_hist)
    x = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0])
    assert (np.all(updated_moves_hist[0]['iY'] == x))


# TEST insert_letter-----------------------------------------------
def test_play_move_1():

    # load Agent Smit
    loaded_modelS = load_model('S')
    # load Agent Linki
    loaded_modelL = load_model('L')

    let = 'X'
    player_x = 'U'
    player_o = 'U'
    i_game_nr = 1
    board = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
    game_moves_log = []
    i_print_board = False

    board, game_moves_log = play_move(let, player_x, player_o, i_game_nr, board, game_moves_log, i_print_board, loaded_modelS, loaded_modelL)

    x = np.array([0, 0, 0, 0, 0.2, 0.5, 0, 0, 0])
    assert (game_moves_log[0]['iY'].shape == x.shape)


def test_make_prediction_math_1():
    # test 100% certain move, however already occupied
    # Say for example board looks as follows:
    iboard = [' ', ' ', 'X', 'O', 'O', 'X', 'X', 'O', 'X', 'O']

    # and the neural net gave the following output
    # ynew = np.array([[0.2, 0.4, 0.2, 0, 0, 0, 0, 0, 0]])
    ynew = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0]])

    ipred, iarray = make_prediction_math(iboard, ynew)

    assert (ipred == 1)


def test_make_prediction_math_2():
    # test 100% certain move
    # Say for example board looks as follows:
    iboard = [' ', ' ', ' ', 'O', 'O', 'X', 'X', 'O', 'X', 'O']

    # and the neural net gave the following output
    # ynew = np.array([[0.2, 0.4, 0.2, 0, 0, 0, 0, 0, 0]])
    ynew = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0]])

    ipred, iarray = make_prediction_math(iboard, ynew)

    assert (ipred == 2)


def test_make_prediction_math_3():
    # test 100% certain move
    # Say for example board looks as follows:
    iboard = [' ', ' ', ' ', 'O', 'O', 'X', 'X', 'O', 'X', 'O']

    # and the neural net gave the following output
    ynew = np.array([[0.1, 0.8, 0.1, 0, 0, 0, 0, 0, 0]])

    ipred, iarray = make_prediction_math(iboard, ynew)

    assert (ipred == 2)



def test_make_prediction_math_4():
    # test 100% certain move
    # Say for example board looks as follows:
    iboard = [' ', 'X', 'X', ' ', 'O', 'X', 'X', 'O', 'X', 'O']

    # and the neural net gave the following output
    ynew = np.array([[0.2, 0.4, 0.2, 0, 0, 0, 0, 0, 0]])

    ipred, iarray = make_prediction_math(iboard, ynew)
    assert (ipred == 3)


def test_create_df_training_stats_1():
    # test creation of training stats datafrem
    df_training_stats = create_df_training_stats()
    assert (df_training_stats.shape[0] > 0)

