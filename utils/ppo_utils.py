# Helper Functions
import numpy as np
import torch
import torch.utils.data as data
import pdb

def make_dict(obs_name, obs):
    zipbObj = zip(obs_name, obs)
    return dict(zipbObj)

def R_func(obs_dict, action, eta):
    reward = - action#- 0.5 * eta[int(obs_dict["Occupancy Flag"])] * (obs_dict["Indoor Air Temp."] - obs_dict["Indoor Temp. Setpoint"] - 1)**2
    return reward#.item()
    
# Calculate the advantage estimate
def Advantage_func(rewards, gamma):
    R = torch.zeros(1, 1).double()
    T = len(rewards)
    advantage = torch.zeros((T,1)).double()
    
    for i in reversed(range(len(rewards))):
        R = gamma * R + rewards[i]
        advantage[i] = R
    return advantage

class Dataset(data.Dataset):
    def __init__(self, states, actions, disturbance, advantages, old_logprobs, x_uppers, x_lowers):
        self.states = states
        self.actions = actions
        self.disturbance = disturbance
        self.advantages = advantages
        self.old_logprobs = old_logprobs
        self.x_uppers = x_uppers
        self.x_lowers = x_lowers

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return self.states[index], self.actions[index], self.disturbance[index], self.advantages[index], self.old_logprobs[index], self.x_uppers[index], self.x_lowers[index]
    
class Replay_Memory():
    def __init__(self, ):
        self.advantages = []
        self.states = []
        self.old_logprobs = []
        self.actions = []
        self.disturbance = [] # T x n_dist
        self.x_uppers = []
        self.x_lowers = []
    
    def clear_memory(self, ):
        self.advantages = []
        self.states = []
        self.old_logprobs = []
        self.actions = []
        self.disturbance = []
        self.x_uppers = []
        self.x_lowers = []
        
    def sample(self):
        states = torch.vstack(self.states)
        actions = torch.vstack(self.actions)
        advantages = torch.vstack(self.advantages).reshape(-1)
        old_logprobs = torch.vstack(self.old_logprobs).reshape(-1)
        disturbance = torch.stack(self.disturbance) # n x T x dist
        x_uppers = torch.vstack(self.x_uppers)
        x_lowers = torch.vstack(self.x_lowers)
        self.clear_memory()
        
        return states, actions, disturbance, advantages, old_logprobs, x_uppers, x_lowers

    
