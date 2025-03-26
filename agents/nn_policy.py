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
        print("T, n_batch, n_dist", T, n_batch, n_dist)
                        
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
            dt = disturbance[:, i, :-1] # T x n_dist-1
            x0 = torch.tensor(self.x0.value).float()  # Now uses the stored clipped x0
            mu = mu.squeeze(1) # T x 1 ->T
            
            try:
                # Print constraint dimensions for debugging
                print(f"Constraint check - dt shape: {dt.shape}, x0 shape: {x0.shape}")
                print(f"u bounds: [{torch.tensor(self.u_lower.value).min()}, {torch.tensor(self.u_upper.value).max()}]")
                
                # Test feasibility with CVXPY directly (outside of layer)
                test_u = cp.Variable(self.T)
                test_problem = cp.Problem(
                    cp.Minimize(cp.sum_squares(test_u)),
                    self.constraints
                )
                feasibility_result = test_problem.solve(solver=cp.SCS, verbose=True)
                print(f"Feasibility test result: {test_problem.status}")
                
                # Now proceed with the differentiable layer
                u_pred = self.proj_layer(x0, dt,
                        mu, torch.zeros_like(mu), torch.zeros_like(mu),
                        x_upper, x_lower,
                        torch.tensor(self.u_upper.value).float(),
                        torch.tensor(self.u_lower.value).float(),
                        solver_args={
                                "max_iters": 50000,
                                "eps": 1e-6,
                                "scale": 10.0,  # Add scaling to improve conditioning
                                "normalize": True  # Enable normalization
                            })
                print(f"Original action (from NN): {mu}")
                print(f"Projected action: {u_pred[0]}")
                print(f"Projection difference: {torch.norm(mu - u_pred[0]).item()}")
                actions.append(u_pred[0])
                
            except Exception as e:
                print(f"Projection failed with error: {str(e)}")
                # Print constraint values for debugging
                print(f"x0: {x0}")
                print(f"x_lower min/max: {x_lower.min()}/{x_lower.max()}")
                print(f"x_upper min/max: {x_upper.min()}/{x_upper.max()}")
                ## The feasible set is empty; Use some heuristics
                sp = torch.mean((x_lower+x_upper)/2)
                print(f"Projection failed. Heuristic used. Total fails: {self.err_count}")
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
        
