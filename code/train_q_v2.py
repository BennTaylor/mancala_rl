import importlib
import mancala
import Qnetwork
import Qagent_v2

importlib.reload(mancala)
importlib.reload(Qnetwork)
importlib.reload(Qagent_v2)

from mancala import Mancala, GameResponse
from Qnetwork import QNetwork
from Qagent_v2 import MyQAgent
import numpy as np
import torch

'''
HYPERPARAMETERS
'''
gamma = 0.99
epsilon = 1.0
lr = 1e-4

num_games = 100000

x = 250 # during training, print p1 win rate out of last x games

def train(a1, a2):
    p1_wins = [0] * num_games
    p2_wins = [0] * num_games
    ties = 0

    num_turns = [0] * num_games

    agent1 = a1
    agent2 = a2

    for iter in range(num_games):
        g = Mancala()
        observation = g.observation()
        while not g.over:
            if g.turn % 2 == 1:
                # agent1's turn
                action = agent1.choose_action(observation)

                g.action(action, zero_ind=True)
                new_observation = g.observation()

                # get reward; for now only associate with game win
                if g.over:
                    if g.wells[6] > g.wells[13]:
                        reward = 10
                        p1_wins[iter] = p1_wins[iter-1] + 1
                        p2_wins[iter] = p2_wins[iter-1]
                    elif g.wells[6] < g.wells[13]:
                        reward = -0.5
                        p1_wins[iter] = p1_wins[iter-1]
                        p2_wins[iter] = p2_wins[iter-1] + 1
                    else:
                        reward = 0
                        p1_wins[iter] = p1_wins[iter-1]
                        p2_wins[iter] = p2_wins[iter-1]
                        ties += 1
                else:
                    reward = 0

                agent1.store_transition(observation[0], action, reward, new_observation[0], g.over)
                agent2.update_reward(reward)
                observation = new_observation
            else:
                # agent2's turn
                action = agent2.choose_action(observation)

                g.action(action, zero_ind=True)
                new_observation = g.observation()

                # get reward; for now only associate with game win
                if g.over:
                    if g.wells[13] > g.wells[6]:
                        p1_wins[iter] = p1_wins[iter-1]
                        p2_wins[iter] = p2_wins[iter-1] + 1
                        reward = 10
                    elif g.wells[13] < g.wells[6]:
                        p1_wins[iter] = p1_wins[iter-1] + 1
                        p2_wins[iter] = p2_wins[iter-1]
                        reward = -0.5
                    else:
                        reward = 0
                        p1_wins[iter] = p1_wins[iter-1]
                        p2_wins[iter] = p2_wins[iter-1]
                        ties += 1
                else:
                    reward = 0

                agent2.store_transition(observation[0], action, reward, new_observation[0], g.over)
                agent1.update_reward(reward)
                observation = new_observation
        # learning only after each game!
        agent1.learn()
        agent2.learn()

        num_turns[iter] = g.turn
        if iter > x:
            print(f'game {iter}: p1 win rate in last {x} games = {np.trunc(100 * (p1_wins[iter] - p1_wins[iter-x]) / x)}%', end='\r')
    

    print(f'\n------ Training complete ------')
    print(f'p1 wins: {p1_wins[-1]}, p2 wins: {p2_wins[-1]}')
    print('Num turns:')
    print(f'min= {min(num_turns)}')
    print(f'max= {max(num_turns)}')
    print(f'avg= {sum(num_turns) / num_games}')

    import matplotlib.pyplot as plt

    # Create the plot
    plt.plot([i+1 for i in range(num_games)], p1_wins, label='p1', linestyle='-', color='b')
    plt.plot([i+1 for i in range(num_games)], p2_wins, label='p2', linestyle='-', color='g')


    # Add labels and a title
    plt.xlabel('games played')
    plt.ylabel('wins')
    plt.title('Two Q-agents v2')

    # Add a legend
    plt.legend()

    # Display the plot
    plt.show()

    # return better agent based on last 1000 games
    if num_games < 2000:
        return agent1
    
    if (p1_wins[-1] - p1_wins[-1000]) > 500:
        print('Selecting agent1 from training')
        return agent1
    else:
        return agent2
    
def play_vs_agent(agent):
    print('Enter \'c\' to play game against agent: ')
    cont = input()
    while(cont == 'c'):
        game = Mancala()

        while not game.over:
            game.print_board()
            p1 = (game.turn % 2 == 1)
            if p1:
                print(f'(turn {game.turn}) Player, enter a well to empty:')
                w = int(input())
                game.action(w, zero_ind=False)
                print()
            else:
                w = agent.choose_action(game.observation(), playing=True)
                print(f'(turn {game.turn}) Agent picks well {w + 1}.\n')
                game.action(w, zero_ind=True)

        game.print_board()
        if game.wells[6] > game.wells[13]:
            print(f'Player 1 wins \np1: {game.wells[6]} \np2: {game.wells[13]}')
        elif game.wells[6] < game.wells[13]:
            print(f'Player 2 wins \np1: {game.wells[6]} \np2: {game.wells[13]}')
        else:
            print(f'Tie \np1: {game.wells[6]} \np2: {game.wells[13]}')
        print('-------------------------------------------------')
        print('Enter \'c\' to play another game against agent: ')
        cont = input()  
        

if __name__ == '__main__':
    agent1 = MyQAgent(gamma, epsilon, lr)
    agent2 = MyQAgent(gamma, epsilon, lr)

    # train returns better agent
    a = train(agent1, agent2)

    play_vs_agent(a)