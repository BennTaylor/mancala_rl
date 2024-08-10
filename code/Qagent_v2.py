import importlib
import Qnetwork

importlib.reload(Qnetwork)

from mancala import Mancala, GameResponse
from Qnetwork import QNetwork
import numpy as np
import torch

'''
Class for agent that will play mancala game and update the Q-network.

This is MY implementation of the Q-agent where I've tweaked the training to better suit the mancala game
'''
class MyQAgent:
    '''
    Omitted Parameters from og QAgent:
    - input_dims: Fixed (not playing w/ state representation yet)
    - batch_size: not batch training anymore
    - n_actions: This is fixed for the game
    - max_mem_sz: batch stuff
    '''
    def __init__(self, gamma, epsilon, lr, eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon      # proportion of time taking random actions
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i+1 for i in range(6)]

        '''
        Only storing memory for one game.
        avg # of turns seems to be around 30, max of 60 out of 500 games
        '''
        self.mem_size = 200
        self.mem_cnt = 0

        self.Q_eval = QNetwork(self.lr, n_actions=6, input_dims=14, 
                                fc1_dims=256, fc2_dims=256)
        
        self.state_memory = np.zeros((self.mem_size, 14), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, 14), dtype=np.float32)
         
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory =  np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
    
    def store_transition(self, state, action, reward, new_state, done):
        if self.mem_cnt == self.mem_size:
            print('Training error: memory exceeded (this shouldn\'t happen)')
            return
        self.state_memory[self.mem_cnt] = state
        self.new_state_memory[self.mem_cnt] = new_state
        self.reward_memory[self.mem_cnt] = reward
        self.action_memory[self.mem_cnt] = action
        self.terminal_memory[self.mem_cnt] = done

        self.mem_cnt += 1

    '''
    Updates the reward of last action based on reward for opponent's response.
    In particular, adds the negative of that reward. *****eh subject to change
    '''
    def update_reward(self, opponent_reward):
        if self.mem_cnt == 0:
            return
        if opponent_reward == 10:
            r = -0.5
        elif opponent_reward == -0.5:
            r = 10
        else:
            r = 0
        self.reward_memory[self.mem_cnt - 1] += r

    '''
    playing is a boolean flag to indicate when agent should only choose model's decision 
    (i.e. when playing against human).
    '''
    def choose_action(self, observation, playing=False):
        wells, legal_actions = observation
        if playing or np.random.random() > self.epsilon:
            state = torch.tensor(wells, dtype=torch.float32).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            actions = actions[legal_actions]
            action = legal_actions[torch.argmax(actions).item()]
        else:
            action = np.random.choice(legal_actions)
        return action
    
    '''
    This is where the real difference from the original is.
    Learning step should propogate reward of winning game through the sequence of actions which got it there.
    learn() should only be called at the completion of a game, and the memory will be reset before returning.
    '''
    def learn(self):
        if self.mem_cnt == 0:
            return
        
        self.Q_eval.optimizer.zero_grad()

        indices = np.arange(self.mem_cnt, dtype=np.float32)

        states = torch.tensor(self.state_memory[0:self.mem_cnt]).to(self.Q_eval.device)
        new_states = torch.tensor(self.new_state_memory[0:self.mem_cnt]).to(self.Q_eval.device)
        actions = torch.tensor(self.action_memory[0:self.mem_cnt]).to(self.Q_eval.device)
        rewards = torch.tensor(self.reward_memory[0:self.mem_cnt]).to(self.Q_eval.device)
        terminals = torch.tensor(self.terminal_memory[0:self.mem_cnt]).to(self.Q_eval.device)

        q_eval = self.Q_eval.forward(states)[indices, actions]
        q_next = self.Q_eval.forward(new_states)
        q_next[terminals] = 0.0

        q_target = rewards + self.gamma * torch.max(q_next, dim=1)[0]
        
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

        self.mem_cnt = 0
