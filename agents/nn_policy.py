import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal, Normal

from utils.network import MLP, LSTM
from agents.base import Controller, ControllerGroup
    
class NeuralController(Controller):
    def __init__(self, T, dt, network, RC_flag = True,
                 **parameters):
        super().__init__(T, dt, RC_flag = RC_flag, **parameters)
        ## Inherited Properties:
        # cp.Variable: self.u
        # cp.Parameter: self.u_diff; self.v_bar; self.w_bar;
        #               self.x0; self.d;
        #               self.x_lower; self.x_upper;
        #               self.u_lower; self.u_upper;
        # self.objective
        # self.Problem
        # self.constraints = [...]

        ## Inherited Methods:
        # updateState()
        
        ## Use ADMM update rule for the time being
        # u_update(self, v_bar, w_bar):
        
        '''
        LSTM Usage:
            lstm = LSTM(n_state, n_action, n_dist)
            mu, sigma_sq = lstm.forward(state, disturbance)
        Input:
            state: n x dim
            disturbance: T x n x dist
        Output:
            mu, sigma_sq: T x n x n_action
        '''
        self.nn = network
        self.proj_layer = CvxpyLayer(self.Problem, variables = [self.u],
                                  parameters = [self.x0, self.d,
                                                self.u_diff, self.v_bar, self.w_bar,
                                                self.x_upper, self.x_lower,
                                                self.u_upper, self.u_lower])
        self.criterion = nn.MSELoss() # reduction = 'sum'

    def forward(self, state, disturbance, x_lowers = None, x_uppers = None, detach = False):
        '''
        Input:
            state: (n, n_state)
            disturbance: (T, n, n_dist)
            x_lowers, x_uppers: (n, T)
        Output:
            actions, sigma_sq: (T, n, n_action)
            #proj_loss: scalar
        '''
        T, n_batch, n_dist = disturbance.shape
        mus, sigma_sqs = self.nn(state, disturbance)# T x n x n_action
                        
        actions = []
        #TODO: Implement multi-threading
        for i in range(n_batch):
            mu = mus[:, i] # T x n_action
            
            if n_batch==1:
                if x_lowers is None:
                    x_lower = torch.tensor(self.x_lower.value).float()
                if x_uppers is None:
                    x_upper = torch.tensor(self.x_upper.value).float()
                
            else:
                x_lower = x_lowers[i]
                x_upper = x_uppers[i]
            
            # The last value is setpoint; Do not use for projection
            dt = disturbance[:, i, :-1] # T x n_dist
            x0 = state[i]
            mu = mu.squeeze(1) # T x 1 ->T
            
            try:
                u_pred = self.proj_layer(x0, dt,
                   mu, torch.zeros_like(mu), torch.zeros_like(mu),
                   x_upper, x_lower,
                   torch.tensor(self.u_upper.value).float(),
                   torch.tensor(self.u_lower.value).float())
                actions.append(u_pred[0])
            except:
                ## The feasible set is empty; Use some heuristics
                sp = torch.mean((x_lower+x_upper)/2)
                if x0.item() < sp:
                    actions.append(torch.ones_like(mu))
                else:
                    actions.append(torch.zeros_like(mu))

        actions = torch.stack(actions).transpose(0, 1) # T x n
        proj_loss = self.criterion(mus.squeeze(-1), actions)
        return actions.unsqueeze(-1), sigma_sqs, proj_loss

class NeuralControllerGroup(ControllerGroup):
    def __init__(self, T, dt, parameters, RC_flag = True):
        super().__init__(T, dt, parameters, RC_flag = RC_flag)
        
        ## Inherited Methods:
        # updateState()
        # u_update()
        
    def _init_agents(self, parameters):
        controller_list = []
        for param in parameters:
            controller_list.append(NeuralController(T = self.T, dt = self.dt, RC_flag = self.RC_flag, **param))
        return controller_list
    
    def u_warmstart(self, x_list):
        u_inits = []
        for idx, controller in enumerate(self.controller_list):
            u_pred = controller.forward(x_list[idx].reshape(1, -1)) # 1 x n_input
            u_inits.append(u_pred.detach().numpy())
        return np.stack(u_inits)
        
    def append(self, states, u_stars):
        for idx, controller in enumerate(self.controller_list):
            controller.memory.append((states[idx], u_stars[idx]))
            
    def update_policy(self, batch_size = 32):
        losses = []
        for idx, controller in enumerate(self.controller_list):
            loss = controller.update_policy(batch_size)
            losses.append(loss)
        return np.array(losses)
        
