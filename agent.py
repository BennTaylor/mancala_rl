import importlib
import Qnetwork

importlib.reload(Qnetwork)

from mancala import Mancala, GameResponse
from Qnetwork import QNetwork
import numpy as np
import torch

'''
Class for agent that will play mancala game and update the Q-network.
source of general architecture: https://www.youtube.com/watch?v=wc-FxNENg9U
'''
class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, 
            max_mem_sz=100000, eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon      # proportion of time taking random vs greedy (policy based) actions
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_sz
        self.batch_size = batch_size
        self.mem_cnt = 0

        self.Q_eval = QNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, 
                                fc1_dims=256, fc2_dims=256)
        
        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
         
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory =  np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
    
    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cnt % self.mem_size        # modulo so memory wraps back to beginning once exceeded
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cnt += 1

    def choose_action(self, observation):
        wells, legal_actions = observation
        if np.random.random() > self.epsilon:
            state = torch.tensor(wells, dtype=torch.float32).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            actions = actions[legal_actions]
            action = legal_actions[torch.argmax(actions).item()]
        else:
            action = np.random.choice(legal_actions)
        return action
    
    def learn(self):
        if self.mem_cnt < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cnt, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.float32)
        
        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
        
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min