import importlib
import mancala

importlib.reload(mancala)

from mancala import Mancala, GameResponse

'''
Preliminary game loop for player vs player terminal functionality.
'''
def play_game():
    game = Mancala()

    while not game.over:
        game.print_board()
        p1 = (game.turn % 2 == 1)
        if p1:
            print(f'Player 1 (turn {game.turn}), enter a well to empty:')
        else:
            print(f'Player 2 (turn {game.turn}), enter a well to empty:')
        w = int(input())
        game.action(w)

    game.print_board()
    if game.wells[6] > game.wells[13]:
        print(f'Player 1 wins \np1: {game.wells[6]} \np2: {game.wells[13]}')
    elif game.wells[6] < game.wells[13]:
        print(f'Player 2 wins \np1: {game.wells[6]} \np2: {game.wells[13]}')
    else:
        print(f'Tie \np1: {game.wells[6]} \np2: {game.wells[13]}')
    
play_game()