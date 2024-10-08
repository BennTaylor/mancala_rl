import importlib
import mancala
import Qnetwork
import Qagent

importlib.reload(mancala)
importlib.reload(Qnetwork)
importlib.reload(Qagent)

from mancala import Mancala, GameResponse
from Qnetwork import QNetwork
from Qagent import QAgent
import numpy as np
import torch

'''
HYPERPARAMETERS
'''
INPUT_DIMS = 14     # just the size of the board
N_ACTIONS = 6       # wells to choose from
gamma = 0.99
epsilon = 1.0
lr = 0.001
batch_size = 64

num_games = 10000

if __name__ == '__main__':
    p1_wins = [0] * num_games
    p2_wins = [0] * num_games
    ties = 0

    num_turns = [0] * num_games

    agent1 = QAgent(gamma, epsilon, lr, INPUT_DIMS, batch_size, N_ACTIONS)
    agent2 = QAgent(gamma, epsilon, lr, INPUT_DIMS, batch_size, N_ACTIONS)

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
                        reward = -10
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
                agent1.learn()
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
                        reward = -10
                    else:
                        reward = 0
                        p1_wins[iter] = p1_wins[iter-1]
                        p2_wins[iter] = p2_wins[iter-1]
                        ties += 1
                else:
                    reward = 0

                agent2.store_transition(observation[0], action, reward, new_observation[0], g.over)
                agent2.learn()
                observation = new_observation

        num_turns[iter] = g.turn
        print(f'iter {iter}: p1 wins= {p1_wins[iter]}', end='\r')
    

    print(f'\n------ Training complete ------')
    print('Num turns:')
    print(f'min= {min(num_turns)}')
    print(f'max= {max(num_turns)}')
    print(f'avg= {sum(num_turns) / num_games}')

    import matplotlib.pyplot as plt

    # Create the plot
    plt.plot([i+1 for i in range(num_games)], p1_wins, label='p1', linestyle='-', color='b')
    plt.plot([i+1 for i in range(num_games)], p2_wins, label='p2', linestyle='-', color='g')


    # Add labels and a title
    plt.xlabel('training epoch')
    plt.ylabel('wins')
    plt.title('Training results')

    # Add a legend
    plt.legend()

    # Display the plot
    plt.show()