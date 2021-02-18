# Import packages used
import random
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date
from keras.models import model_from_json

# df_columns used to moves history
df_columns = ['gameNr', 'winner', 'xPlayer', 'oPlayer',
              's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9',
              'nextPlayer', 'nextMove',
              'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9',
              'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18',
              'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9']


def load_model(agent):
    """
    Load the keras model from file
    :param agent (char1): L or S
    :return: loaded_model (keras model)
    """

    if agent == 'S':
        # load json and create model
        json_file = open('modelAgentSmit.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("modelAgentSmit.h5")
        # print("Loaded Agent Smit from disk")
        # evaluate loaded model on test data
        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return loaded_model

    if agent == 'L':
        # load json and create model
        json_file = open('modelAgentLinki.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_modelL = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_modelL.load_weights("modelAgentLinki.h5")
        # print("Loaded Agent Linki from disk")
        # evaluate loaded model on test data
        loaded_modelL.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return loaded_modelL


def create_df_all_played_games():
    """
    blank dataframe containing the game history structure
    :return: df_all_played_games (dataframe): blank dataframe containing the game history structure
    """
    # all games with the moves stored in a dataframe
    df_columns = ['gameNr', 'winner', 'xPlayer', 'oPlayer',
                  's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9',
                  'nextPlayer', 'nextMove',
                  'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9',
                  'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18',
                  'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9']
    df_all_played_games = pd.DataFrame(columns=df_columns, data=[])
    return df_all_played_games


def create_df_training_stats():
    """
    Creates and returns a blank dataframe. this is structured in order to keep stats on the results of the games played
    :return: df_training_stats (dataframe), blank dataframe to keep stats of the games
    """
    # Stats about training session
    d = {'nr': [1, 2, 3],
         'nr_of_games': [0, 0, 0],
         'X_wins': [0, 0, 0],
         'O_wins': [0, 0, 0],
         'draws': [0, 0, 0],
         'U_wins': [0, 0, 0],
         'S_wins': [0, 0, 0],
         'L_wins': [0, 0, 0]}

    df_training_stats = pd.DataFrame(data=d)
    return df_training_stats


def get_training_stats(df_all_played_games, int_stats, inr):
    """
    Determine training stat based on the df_all_played_games
        - number of games
        - number of draws
        - number of x wins
        - number of o wins
        - get the number of Untrained agent wins

    :param df_all_played_games: (dataframe) contains all stats
    :param int_stats: (dataframe) datafrane with current stats, this dataframe will be updated
    :param inr: training cycle number
    :return:
    """

    # get the number of games
    nr_of_games_played = df_all_played_games['gameNr'].nunique()
    int_stats['nr_of_games'][inr] = nr_of_games_played

    # get the number of draws
    player_wins = df_all_played_games[df_all_played_games['winner'] == '-']
    nr_of_player_wins = player_wins['gameNr'].nunique()
    int_stats['draws'][inr] = nr_of_player_wins

    # get the wins by X
    player_wins = df_all_played_games

    nr_of_player_wins = player_wins['gameNr'].nunique()
    int_stats['X_wins'][inr] = nr_of_player_wins

    # get the wins by O
    player_wins = df_all_played_games[df_all_played_games['winner'] == 'O']
    nr_of_player_wins = player_wins['gameNr'].nunique()
    int_stats['O_wins'][inr] = nr_of_player_wins

    # UU
    # get the wins by U playing X
    player_wins = df_all_played_games[(df_all_played_games['winner'] == 'X') & (df_all_played_games['xPlayer'] == 'U')]
    nr_of_player_wins_x = player_wins['gameNr'].nunique()

    # get the wins by U playing O
    player_wins = df_all_played_games[(df_all_played_games['winner'] == 'O') & (df_all_played_games['oPlayer'] == 'U')]
    nr_of_player_wins_o = player_wins['gameNr'].nunique()
    nr_of_player_wins_u = nr_of_player_wins_x + nr_of_player_wins_o

    int_stats['U_wins'][inr] = nr_of_player_wins_u

    # S
    # get the wins by S playing X
    player_wins = df_all_played_games[(df_all_played_games['winner'] == 'X') & (df_all_played_games['xPlayer'] == 'S')]
    nr_of_player_wins_x = player_wins['gameNr'].nunique()

    # get the wins by S playing O
    player_wins = df_all_played_games[(df_all_played_games['winner'] == 'O') & (df_all_played_games['oPlayer'] == 'S')]
    nr_of_player_wins_o = player_wins['gameNr'].nunique()
    nr_of_player_wins_u = nr_of_player_wins_x + nr_of_player_wins_o

    int_stats['S_wins'][inr] = nr_of_player_wins_u


    # LL
    # get the wins by L playing X
    player_wins = df_all_played_games[(df_all_played_games['winner'] == 'X') & (df_all_played_games['xPlayer'] == 'L')]
    nr_of_player_wins_x = player_wins['gameNr'].nunique()

    # get the wins by L playing O
    player_wins = df_all_played_games[(df_all_played_games['winner'] == 'O') & (df_all_played_games['oPlayer'] == 'L')]
    nr_of_player_wins_o = player_wins['gameNr'].nunique()
    nr_of_player_wins_u = nr_of_player_wins_x + nr_of_player_wins_o

    int_stats['L_wins'][inr] = nr_of_player_wins_u


    return int_stats


def print_agent_move(current_player, move_array, i_print_board):
    """
    Function prints the current player and move array
    :param current_player: The current player H, U, S, L
    :param move_array: the move in array format
    :param i_print_board: True if the print must execute
    :return:
    """

    if i_print_board is True:
        print(' ->  ', current_player, '  -> ', move_array)
        print('')

# Print board
def output_board(i_board, i_print_board, header=False):
    """
    Print the board state
    Args:
        i_board (list) of len10  (starting at 1)
        i_print_board(boolean), True if the function should print the board
        header (boolean), True if header must be printed
    Returns:
        out_board (list) representation of the board
    """

    sub = 0
    if len(i_board) == 9:
        sub = 1

    out_board = ['', '', '', '', '']
    out_board[0] = i_board[1-sub] + '|' + i_board[2-sub] + '|' + i_board[3-sub]
    out_board[1] = '-----'
    out_board[2] = i_board[4-sub] + '|' + i_board[5-sub] + '|' + i_board[6-sub]
    out_board[3] = '-----'
    out_board[4] = i_board[7-sub] + '|' + i_board[8-sub] + '|' + i_board[9-sub]

    # print board output
    if i_print_board is True:

        if header is True:
            print('')
            print('')
            print('Board   ->             Board State                ->  Agent  ->  Move       ')
            print('-------------------------------------------------------------------------------------------')

        iX = boardToX(i_board)
        # print(iX)

        print('')
        print(out_board[0])
        print(out_board[1])
        print(out_board[2])
        print(out_board[3])
        print(out_board[4], '  -> ', iX , end =' ')
        # print('           ', iX)



    return out_board


def iconv(idata, input):
    """
    compare idata and input, if it is the same return 1, if not return 0
    :param idata: (char1) char1 input
    :param input: (char1) char1 input
    :return: returns a 1 if the values are the same or a 0 if they are not the same
    """
    ret = 0
    if idata == input:
        ret = 1

    return int(ret)

# board to X
def boardToX(iboard):
    iX = np.zeros(18, dtype=int)
    iX[0] = iconv(iboard[1], 'X')
    iX[1] = iconv(iboard[2], 'X')
    iX[2] = iconv(iboard[3], 'X')
    iX[3] = iconv(iboard[4], 'X')
    iX[4] = iconv(iboard[5], 'X')
    iX[5] = iconv(iboard[6], 'X')
    iX[6] = iconv(iboard[7], 'X')
    iX[7] = iconv(iboard[8], 'X')
    iX[8] = iconv(iboard[9], 'X')
    iX[9] = iconv(iboard[1], 'O')
    iX[10] = iconv(iboard[2], 'O')
    iX[11] = iconv(iboard[3], 'O')
    iX[12] = iconv(iboard[4], 'O')
    iX[13] = iconv(iboard[5], 'O')
    iX[14] = iconv(iboard[6], 'O')
    iX[15] = iconv(iboard[7], 'O')
    iX[16] = iconv(iboard[8], 'O')
    iX[17] = iconv(iboard[9], 'O')

    return iX


def insert_letter(letter, pos, i_game_nr, int_board, int_moves_hist):
    """
    This function
     1. takes the board and insert the letter at that spesific position
     2. it updates the moves history and passes the updated board and the moves history back

    :param letter: (char1) - letter that will be played: X or O
    :param pos: (int) - position where the move will be placed
    :param i_game_nr: (int) - the game identifier
    :param int_board: (list) - the game board
    :param int_moves_hist: (tuple) - game history
    :return:
    """

    output_moves_hist = int_moves_hist
    output_board = int_board

    iboard = int_board[:]
    iX = boardToX(iboard)
    iY = np.zeros(9, dtype=int)
    if pos == 1:
        iY[0] = '1'
    if pos == 2:
        iY[1] = '1'
    if pos == 3:
        iY[2] = '1'
    if pos == 4:
        iY[3] = '1'
    if pos == 5:
        iY[4] = '1'
    if pos == 6:
        iY[5] = '1'
    if pos == 7:
        iY[6] = '1'
    if pos == 8:
        iY[7] = '1'
    if pos == 9:
        iY[8] = '1'

    output_moves_hist.append({'GameNr': i_game_nr,
                      'Winner': '',

                      # Save the state can be used by other training models
                      'S1': iboard[1],
                      'S2': iboard[2],
                      'S3': iboard[3],
                      'S4': iboard[4],
                      'S5': iboard[5],
                      'S6': iboard[6],
                      'S7': iboard[7],
                      'S8': iboard[8],
                      'S9': iboard[9],

                      # X and Y of neural network prepared
                      'iX': iX,
                      'iY': iY,
                      'NextMove': pos,
                      'Player': letter})

    output_board[pos] = letter
    return output_board, output_moves_hist


# check if the board is full
def isBoardFull(board):
    """checks if the board is full, if full the function will return a true"""
    if board.count(' ') > 1:
        return False
    else:
        return True


def make_prediction_math(iboard, ynew):
    """
    Make a prediction using the boardstate
    :param iboard: board = [' '] * 10 containing the board state
    :param ynew: (array) containing the output of the NN
    :return: (array) New output based on this function
    """
    gamma = 3
    irandomness = 0.9

    # check if 100% chance of the move
    if 1 in ynew:
        out_ynew = ynew
        prednr = np.argmax(out_ynew) + 1

        # if position is already taken, make a random move
        while iboard[prednr] != ' ':
            prednr = random.randint(1, 9)
            out_ynew = get_y(prednr)

    else:

        ynew[0] = ynew[0] ** gamma / np.sum(ynew[0] ** gamma)
        ynew[0] = np.around(ynew[0].astype(float), decimals=3)
        ynew[0] /= ynew[0].sum()

        ynew[0] = ynew[0] * irandomness        # allow some randomness

        out_ynew = np.random.multinomial(1, ynew[0])


        # do not allow a random move into a occupied space
        prednr = np.argmax(out_ynew) + 1
        while iboard[prednr] != ' ':
            # set this position prediciton to 0
            ynew[0][prednr - 1] = 0

            # adjust the gamma based on the new y
            ynew[0] = ynew[0] ** gamma / np.sum(ynew[0] ** gamma)
            ynew[0] = np.around(ynew[0].astype(float), decimals=3)
            ynew[0] /= ynew[0].sum()

            ynew[0] = ynew[0] * irandomness  # allow some randomness

            try:
                out_ynew = np.random.multinomial(1, ynew[0])
                prednr = np.argmax(out_ynew) + 1
            except:
                # if position is already taken, make a random move
                while iboard[prednr] != ' ':
                    prednr = random.randint(1, 9)
                    out_ynew = get_y(prednr)
                # print('random move:', out_ynew)
                # print(iboard)
                # print(ynew)
                # print(prednr)

    out_ynew = np.around(out_ynew, decimals=3)


    return prednr, out_ynew


def make_prediction(iboard, player, loaded_modelS, loaded_modelL):
    """
    make a prediction based on the board.
    :param iboard: array containing the board state
    :param player: Player S or L
    :param loaded_modelS: the loaded keras model containing agent S
    :param loaded_modelL: the loaded keras model containing agent L
    :return: prednr (int) position on the board that shoud be played
             out_ynew (array) output if neural net and adjustments made
    """
    iX = np.zeros((1, 18), dtype=int)
    iX[0] = np.array(boardToX(iboard))

    if player == 'S':
        ynew = loaded_modelS.predict_proba(iX)
    else:
        ynew = loaded_modelL.predict_proba(iX)

    prednr, out_ynew = make_prediction_math(iboard, ynew)

    return prednr, out_ynew



def get_y(move):
    """
    this function takes the move and returns a array to reporesent the move
    if move is 1 then the first item of the array will be 1
    if move is 2 then the second item of the array will be 1
    :param move:
    :return:
    """
    iY = np.zeros(9, dtype=int)
    pos = move
    if pos == 1:
        iY[0] = '1'
    if pos == 2:
        iY[1] = '1'
    if pos == 3:
        iY[2] = '1'
    if pos == 4:
        iY[3] = '1'
    if pos == 5:
        iY[4] = '1'
    if pos == 6:
        iY[5] = '1'
    if pos == 7:
        iY[6] = '1'
    if pos == 8:
        iY[7] = '1'
    if pos == 9:
        iY[8] = '1'

    return iY


def get_move(player, let, curr_board, loaded_modelS, loaded_modelL):
    """
    This function looks at the board and based on the board and type of user, it proposes a move
    :param player: (char1) - the type of player, H, U, etc
    :param let: (char1) - the letter that will be played (X or O)
    :param curr_board: (list) - the current board state
    :return:
    """

    move_array = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    if player == 'H':
        move = input('Player ' + let + ' select a position: ')
        move_array = get_y(move)
    elif player == 'U':
        move = random.randint(1, 9)
        move_array = get_y(move)
    elif player == 'S':
        move, move_array = make_prediction(curr_board, player, loaded_modelS, loaded_modelL)
    elif player == 'L':
        move, move_array = make_prediction(curr_board, player, loaded_modelS, loaded_modelL)
    else:
        move = random.randint(1, 9)
        move_array = get_y(move)

    return move, move_array


def play_move(let, player_x, player_o, i_game_nr, i_board, moves_hist, i_print_board, loaded_modelS, loaded_modelL):
    """
    This function checks the board and then based on the board state plays a move on the board
    :param let: (char1) - The current letter that will be played X or O
    :param player_x: (char1) - player X type: H, U, etc
    :param player_o: (char1) - player O type: H, U, etc
    :param i_game_nr: (int) - the current game number
    :param i_board: (list) - the game board
    :param moves_hist: (tuple) - log of the game moves
    :param i_print_board: (boolean) - true if values need to be printed
    :return:
        updated_board (list) - updated game board after the play has been played
        updated_moves_hist (tuple) - updated log of the game moves
    """
    run = True
    move = 0
    updated_board = i_board
    updated_moves_hist = moves_hist
    current_player = ''

    while run:
        # the player to play is player X
        if let == 'X':
            current_player = player_x
        # the player to play is player O
        elif let == 'O':
            current_player = player_o

        move, move_array = get_move(current_player, let, i_board, loaded_modelS, loaded_modelL)

        # try to make the move
        try:
            move = int(move)
            if 0 < move < 10:
                if i_board[move] == ' ':          # space is free
                    run = False
                    updated_board, updated_moves_hist = insert_letter(let, move, i_game_nr, i_board, moves_hist)
                    print_agent_move(current_player, move_array, i_print_board)

                else:
                    if let == 'X' and player_x == 'H':
                        print('Space is occupied!')
                    if let == 'O' and player_o == 'H':
                        print('Space is occupied!')
                    # if let == 'X' and player_x == 'S':
                    #     print(i_board)
                    #     print(move_array)
                    #     print(move)
                    #     print(np.argmax(move_array) + 1)
                    #     print_agent_move(current_player, move_array, True)
                    #     print('Space is occupied!')
                    # if let == 'O' and player_o == 'S':
                    #     print(i_board)
                    #     print(move)
                    #     print(np.argmax(move_array) + 1)
                    #     print_agent_move(current_player, move_array, True)
                    #     print('Space is occupied!')
                    # if let == 'X' and player_x == 'L':
                    #     print(i_board)
                    #     print(move)
                    #     print_agent_move(current_player, move_array, True)
                    #     print('Space is occupied!')
                    # if let == 'O' and player_o == 'L':
                    #     print(i_board)
                    #     print(move)
                    #     print_agent_move(current_player, move_array, True)
                    #     print('Space is occupied!')
        except:
            print('error')

    return updated_board, updated_moves_hist


# Check if input is winner
def is_winner(bo, le):
    """
    this funciton check if the input leads to a win
    :param bo: board (array len 10, first position is empty then the board starts)
    :param le: X or a O
    :return: true of a false (True for winner)
    """

    return ((bo[7] == le and bo[8] == le and bo[9] == le) or
            (bo[4] == le and bo[5] == le and bo[6] == le) or
            (bo[1] == le and bo[2] == le and bo[3] == le) or
            (bo[1] == le and bo[4] == le and bo[7] == le) or
            (bo[2] == le and bo[5] == le and bo[8] == le) or
            (bo[3] == le and bo[6] == le and bo[9] == le) or
            (bo[1] == le and bo[5] == le and bo[9] == le) or
            (bo[3] == le and bo[5] == le and bo[7] == le))


def save_game_to_df(winner, GameNr, xPlayer, oPlayer, movesHist, i_pboard):
    """
    This function take the moveshistory and coverts it into a dataframe
    :param winner: letter containing the X or O representing the winner
    :param GameNr: the game number uniquely identifying the game
    :param xPlayer: type of player X - H, U, S or L
    :param oPlayer: type of player O - H, U, S or L
    :param movesHist: gameNr's moves history
    :param i_pboard: True if the board should be printed
    :return:
    """

    if i_pboard is True:
        # print('')
        if winner == '-':
            print('         ->        ', 'Draw')
        else:
            print('          ->          ', winner + ' won!')
        print('')

    # all games with the moves stored in a dataframe
    df_columns = ['gameNr', 'winner', 'xPlayer', 'oPlayer',
                  's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9',
                  'nextPlayer', 'nextMove',
                  'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9',
                  'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18',
                  'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9'
                  ]

    # create empty lists
    i_gameNr = []
    i_winner = []
    i_xPlayer = []
    i_oPlayer = []
    i_s1 = []
    i_s2 = []
    i_s3 = []
    i_s4 = []
    i_s5 = []
    i_s6 = []
    i_s7 = []
    i_s8 = []
    i_s9 = []
    i_nextPlayer = []
    i_nextMove = []
    i_x1 = []
    i_x2 = []
    i_x3 = []
    i_x4 = []
    i_x5 = []
    i_x6 = []
    i_x7 = []
    i_x8 = []
    i_x9 = []
    i_x10 = []
    i_x11 = []
    i_x12 = []
    i_x13 = []
    i_x14 = []
    i_x15 = []
    i_x16 = []
    i_x17 = []
    i_x18 = []

    i_y1 = []
    i_y2 = []
    i_y3 = []
    i_y4 = []
    i_y5 = []
    i_y6 = []
    i_y7 = []
    i_y8 = []
    i_y9 = []

    i = 0

    # save the winner in the list
    while i < len(movesHist):
        if GameNr == movesHist[i]['GameNr']:
            movesHist[i]['Winner'] = winner
            i_gameNr.append(movesHist[i]['GameNr'])
            i_winner.append(movesHist[i]['Winner'])
            i_xPlayer.append(xPlayer)
            i_oPlayer.append(oPlayer)
            i_s1.append(movesHist[i]['S1'])
            i_s2.append(movesHist[i]['S2'])
            i_s3.append(movesHist[i]['S3'])
            i_s4.append(movesHist[i]['S4'])
            i_s5.append(movesHist[i]['S5'])
            i_s6.append(movesHist[i]['S6'])
            i_s7.append(movesHist[i]['S7'])
            i_s8.append(movesHist[i]['S8'])
            i_s9.append(movesHist[i]['S9'])

            i_nextPlayer.append(movesHist[i]['Player'])
            i_nextMove.append(movesHist[i]['NextMove'])

            i_x1.append(movesHist[i]['iX'][0])
            i_x2.append(movesHist[i]['iX'][1])
            i_x3.append(movesHist[i]['iX'][2])
            i_x4.append(movesHist[i]['iX'][3])
            i_x5.append(movesHist[i]['iX'][4])
            i_x6.append(movesHist[i]['iX'][5])
            i_x7.append(movesHist[i]['iX'][6])
            i_x8.append(movesHist[i]['iX'][7])
            i_x9.append(movesHist[i]['iX'][8])
            i_x10.append(movesHist[i]['iX'][9])
            i_x11.append(movesHist[i]['iX'][10])
            i_x12.append(movesHist[i]['iX'][11])
            i_x13.append(movesHist[i]['iX'][12])
            i_x14.append(movesHist[i]['iX'][13])
            i_x15.append(movesHist[i]['iX'][14])
            i_x16.append(movesHist[i]['iX'][15])
            i_x17.append(movesHist[i]['iX'][16])
            i_x18.append(movesHist[i]['iX'][17])

            i_y1.append(movesHist[i]['iY'][0])
            i_y2.append(movesHist[i]['iY'][1])
            i_y3.append(movesHist[i]['iY'][2])
            i_y4.append(movesHist[i]['iY'][3])
            i_y5.append(movesHist[i]['iY'][4])
            i_y6.append(movesHist[i]['iY'][5])
            i_y7.append(movesHist[i]['iY'][6])
            i_y8.append(movesHist[i]['iY'][7])
            i_y9.append(movesHist[i]['iY'][8])
        i = i + 1

    data = {
            'gameNr': i_gameNr,
            'winner': i_winner,
            'xPlayer': i_xPlayer,
            'oPlayer': i_oPlayer,
            's1': i_s1,
            's2': i_s2,
            's3': i_s3,
            's4': i_s4,
            's5': i_s5,
            's6': i_s6,
            's7': i_s7,
            's8': i_s8,
            's9': i_s9,
            'nextPlayer': i_nextPlayer,
            'nextMove': i_nextMove,
            'x1': i_x1,
            'x2': i_x2,
            'x3': i_x3,
            'x4': i_x4,
            'x5': i_x5,
            'x6': i_x6,
            'x7': i_x7,
            'x8': i_x8,
            'x9': i_x9,
            'x10': i_x10,
            'x11': i_x11,
            'x12': i_x12,
            'x13': i_x13,
            'x14': i_x14,
            'x15': i_x15,
            'x16': i_x16,
            'x17': i_x17,
            'x18': i_x18,
            'y1': i_y1,
            'y2': i_y2,
            'y3': i_y3,
            'y4': i_y4,
            'y5': i_y5,
            'y6': i_y6,
            'y7': i_y7,
            'y8': i_y8,
            'y9': i_y9,
            }

    df_games_played = pd.DataFrame(data, columns=df_columns)

    return df_games_played


# ---------------------------------------------------------------------------------------------------------------------#
# Enter Game
# ---------------------------------------------------------------------------------------------------------------------#
def play_game(player_x, player_o, i_game_nr, i_print_board, loaded_modelS, loaded_modelL):
    """
    This function executes a game of tic tac toe
    :param player_x: this indicates who the player is, H: Human, U: Untrained agent etc.
    :param player_o: this indicates who the player is, H: Human, U: Untrained agent etc.
    :param i_game_nr: this is the unique number of the current game
    :param i_print_board: true if the board state must be printed
    :return: df_game_moves_log (dataframe): this contains a list of the moves of the game and also the results
    """

    # print(loaded_modelS.weights[0][0])

    # save moves played
    game_moves_log = []
    df_game_moves_log = []

    # initialise the board and output
    board = [' '] * 10
    output_board(board, i_print_board, True)

    # continue the game until this is true
    game_end = False

    # first move should be X
    let = 'O'

    # play moves until end of game
    while not game_end:

        # alternate between X and O
        if let == 'X':
            let = 'O'
        else:
            let = 'X'

        # play a move and return the updated board and the moves log
        board, game_moves_log = play_move(let, player_x, player_o, i_game_nr, board,
                                          game_moves_log, i_print_board, loaded_modelS, loaded_modelL)
        output_board(board, i_print_board)

        # determine winner/tie or continue game
        if is_winner(board, let):  # O Winner
            df_game_moves_log = save_game_to_df(let, i_game_nr, player_x, player_o, game_moves_log, i_print_board)
            game_end = True

        elif isBoardFull(board):
            df_game_moves_log = save_game_to_df('-', i_game_nr, player_x, player_o, game_moves_log, i_print_board)
            game_end = True

    return df_game_moves_log


def print_progress(total, counter, outputs, info, p_board=False):
    """ prints the progress
        args:
            total (int) -> total number of record
            counter (int) -> counter
            outputs (int) - How many times to output progress
            info (str) -> Text to display with status
    """

    # get current time
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    # print progress
    every = total / outputs
    if counter == 1:
        print(str(counter), '/', total, info, current_time)

    elif ((counter/every).is_integer() and counter != 0) or counter == int(total):
        print('       - ', end='')
        print(str(counter), '/', total, info, current_time)


def main():

    loaded_modelS = load_model('S')
    loaded_modelL = load_model('L')

    np_played_games = play_game('L', 'H', 1, True,
                                    loaded_modelS, loaded_modelL).to_numpy()

    print('Moves made:')
    print(np_played_games)


