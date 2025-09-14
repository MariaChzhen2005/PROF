"""
no_proj_policy.py: This file contains the neural network without projection.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import operator
from functools import reduce

import pdb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


### Simple Neural Network without PROF projection layer
class Net(nn.Module):
    def __init__(self, n_bus, n_inverters, shared_hidden_layer_sizes, indiv_hidden_layer_sizes, n_input = 3):
        super(Net, self).__init__()
        #### Multi-headed architecture
        # "Shared" model
        # Set up non-linear network of Linear -> BatchNorm -> ReLU
        layer_sizes = [n_input * n_bus] + shared_hidden_layer_sizes[:-1]
        layers = reduce(operator.add, 
            [[nn.Linear(a,b), nn.ReLU(), ] # nn.BatchNorm1d(b), nn.Dropout(p=0.2)]
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], shared_hidden_layer_sizes[-1])]
        self.base_net = nn.Sequential(*layers)
        
        # Individual inverter model
        layer_sizes = [shared_hidden_layer_sizes[-1]] + indiv_hidden_layer_sizes
        layers = reduce(operator.add, 
            [[nn.Linear(a,b),  nn.ReLU(), ] # nn.BatchNorm1d(b), nn.Dropout(p=0.2)]
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], 2)]  # output p and q
        indiv_model = nn.Sequential(*layers)
        self.inverter_nets = nn.ModuleList(
                [deepcopy(indiv_model) for _ in range(n_inverters)]
                )

    def forward(self, state):
        '''
            Input: Vector of voltage magnitudes and angles, real and reactive power demand
            Output: Vector of inverter P setpoints, vector of inverter Q setpoints
        '''
        # Multi-headed architecture
        z = self.base_net(state)
        res = [inverter(z) for inverter in self.inverter_nets]
        Ps = torch.cat([x[:, [0]] for x in res], dim=1)
        Qs = torch.cat([x[:, [1]] for x in res], dim=1)
        return Ps, Qs


class SimpleNeuralController(nn.Module):
    def __init__(self, network, memory, lr, scaler = 1000, **env_params):
        super(SimpleNeuralController, self).__init__()
        self.nn = network
        self.optimizer = optim.RMSprop(self.nn.parameters(), lr=lr)
        self.memory = memory
        self.mse = nn.MSELoss()
        self.ReLU = nn.ReLU()
        
        self.n_bus = env_params['n_bus']
        self.gen_idx = env_params['gen_idx']
        self.other_idx = [i for i in range(self.n_bus) if i not in self.gen_idx]
        
        self.scaler = scaler
        self.V0 = env_params['V0']
        self.P0 = env_params['P0']
        self.Q0 = env_params['Q0']
        self.V_upper = env_params['V_upper']
        self.V_lower = env_params['V_lower']
        self.S_rating = env_params['S_rating']
        
    def forward(self, state, Sbus, P_av, inference_flag = True):
        '''
        Input:
            state: [dV(k-1), P_nc, Q_nc] 
          where,
                Z_nc = Z - Z0
            May get (n, dim) or (dim);
        Output:
            P, Q (with respect to the reference point) - RAW neural network outputs without projection
        '''
        ## Get information for non-controllable loads
        P_all = Sbus.real / self.scaler
        Q_all = Sbus.imag / self.scaler
        
        P_tilde, Q_tilde = self.nn(state.to(DEVICE)) # n x n_inverter
        
        # Simply use raw neural network outputs without any projection
        if inference_flag:
            P_tilde = P_tilde.squeeze()
            Q_tilde = Q_tilde.squeeze()
            P_all[self.gen_idx] = P_tilde.detach().cpu().numpy() / self.scaler
            Q_all[self.gen_idx] = Q_tilde.detach().cpu().numpy() / self.scaler
            return P_all, Q_all
        else:
            # For training, return the raw outputs scaled appropriately
            return P_tilde/self.scaler, Q_tilde/self.scaler, torch.tensor(0.0)  # No projection loss
    
    def update(self, batch_size = 64, n_batch = 16):
        for _ in range(n_batch):
            state, Sbus, P_av = self.memory.sample_batch(batch_size = batch_size)
            P, Q, _ = self.forward(state, Sbus, P_av, inference_flag = False)
            
            # Simple curtailment loss - try to maximize generation up to available power
            curtail = self.ReLU(torch.tensor(P_av).to(DEVICE) - P)
            loss = curtail.mean()
            print(f'curtail = {curtail.mean().item()}')
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
