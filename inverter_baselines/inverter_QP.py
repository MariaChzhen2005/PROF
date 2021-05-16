import os, sys, argparse

import numpy as np
import cvxpy as cp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

main_path = os.path.abspath(os.path.join(__file__, '..'))
sys.path.insert(0, main_path)

from env.inverter import IEEE37

from algo.ppo import PPO
from agents.inverter_policy import Net, NeuralController
from utils.inverter_utils import Replay_Memory


import pdb

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE

parser = argparse.ArgumentParser(description='GnuRL Demo: Online Learning')
parser.add_argument('--gamma', type=float, default=0.98, metavar='G',
                    help='discount factor (default: 0.98)')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 42)')
parser.add_argument('--lam', type=int, default=10, metavar='N',
                    help='random seed (default: 42)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='G',
                    help='Learning Rate')
parser.add_argument('--epsilon', type=float, default=0.2, metavar='G', help='PPO Clip Parameter')
parser.add_argument('--update_episode', type=int, default=4, metavar='N',
                    help='PPO update episode (default: 1); If -1, do not update weights')
parser.add_argument('--exp_name', type=str, default='inverter_QP',
                    help='save name')
parser.add_argument('--network_name', type=str, default='ieee37',
                    help='')
args = parser.parse_args()

class QP_solver():
    def __init__(self,  **env_params):
        self.n_bus = env_params['n_bus']
        H = env_params['H']
        self.V0 = env_params['V0']
        self.P0 = env_params['P0']
        self.Q0 = env_params['Q0']
        self.V_upper = env_params['V_upper']
        self.V_lower = env_params['V_lower']
        self.S_rating = env_params['S_rating']
        self.gen_idx = env_params['gen_idx']

        self.other_idx = [i for i in range(self.n_bus) if i not in self.gen_idx ]
        
        R = H[:, :self.n_bus]
        B = H[:, self.n_bus:]
        R_new = np.vstack([np.hstack([R[self.gen_idx][:, self.gen_idx], 
                                      R[self.gen_idx][:, self.other_idx]]), 
                            np.hstack([R[self.other_idx][:, self.gen_idx], 
                                       R[self.other_idx][:, self.other_idx]])
                            ])
        B_new = np.vstack([np.hstack([B[self.gen_idx][:, self.gen_idx], 
                                      B[self.gen_idx][:, self.other_idx]]), 
                            np.hstack([B[self.other_idx][:, self.gen_idx], 
                                       B[self.other_idx][:, self.other_idx]])
                            ])
        #pdb.set_trace()
        H_new = np.hstack([R_new, B_new])
                          
        # Set up projection onto inverter setpoint constraints and linearized voltage constraints
        self.P = cp.Variable(len(self.gen_idx))
        self.Q = cp.Variable(len(self.gen_idx))
        
        
        self.P_nc = cp.Parameter(len(self.other_idx))
        self.Q_nc = cp.Parameter(len(self.other_idx))
        self.P_av = cp.Parameter(len(self.gen_idx))
        
        # Voltage: Apply to All Buses
        z = cp.hstack([self.P, self.P_nc, self.Q, self.Q_nc]) # z: (70, )
        constraints = [self.V_lower - self.V0 <= H_new@z,
                       H_new@z <= self.V_upper - self.V0]
        
        ## Power: Only applies to Inverters
        PQ = cp.vstack([self.P0[self.gen_idx] + self.P, 
                       self.Q0[self.gen_idx] + self.Q]) # (2, n)
        constraints += [0 <= self.P0[self.gen_idx] + self.P,
                       self.P0[self.gen_idx] + self.P <= self.P_av,
                       cp.norm(PQ, axis = 0) <= self.S_rating]
        
        #objective = cp.Minimize(cp.sum_squares(P - P_tilde) + cp.sum_squares(Q - Q_tilde))
        objective = cp.Minimize(cp.sum(cp.maximum(self.P_av - self.P, 
                                                  np.zeros(len(self.gen_idx)))))
        self.problem = cp.Problem(objective, constraints)

    def solve(self, Sbus, P_av):
        self.P_nc.value = Sbus.real[self.other_idx]
        self.Q_nc.value = Sbus.imag[self.other_idx]
        self.P_av.value = P_av
        
        #try:
        self.problem.solve()
        #except:
        #    print("Solver failed")
        #    self.P.value = None

        ## Check solution valid
        #if self.P.value is not None:
        #print(self.problem.status)
        #print(self.P.value, self.Q.value)
        return self.P.value, self.Q.value#, self.Problem.status
        
        #else:
        #    return Sbus.real, Sbus.imag

def main():
    torch.manual_seed(args.seed)
    writer = SummaryWriter(comment = args.exp_name)
    
    # Create Simulation Environment
    if args.network_name == 'ieee37':
        env = IEEE37()
    else:
        print("Not implemented")
    
    n_bus = env.n - 1
    n_inverters = len(env.gen_idx) # inverters at PV panels
    
    env_params = {'V0': env.V0[-env.n_pq:],
                  'P0': env.P0[-env.n_pq:],
                  'Q0': env.Q0[-env.n_pq:],
                  'H': np.hstack([env.R, env.B]), # 35 x 70
                  'n_bus':n_bus, # Slack bus is not controllable
                  'gen_idx': env.gen_idx - 1, # Excluded the slack bus
                  'V_upper': env.v_upper, 'V_lower': env.v_lower,
                 'S_rating': env.max_S,
                 }

    controller = QP_solver(**env_params)

    # 1-week data
    num_steps = 900 # 15 minutes
    n_episodes = 7*86400//num_steps

    V_prev = np.zeros(n_bus)
    
    V_record = []
    V_est_record = []
    P_record = []
    Q_record = []
    
    for i in range(n_episodes):
        loss = 0
        violation_count = 0
        
        for k in range(num_steps):
            t = i*num_steps + k
            Sbus, P_av = env.getSbus(t)
            
            P_gen, Q_gen = controller.solve(Sbus, P_av)
            print(f"P_av = {P_av}, P = {P_gen}")

            P = Sbus.real
            Q = Sbus.imag
            P[controller.gen_idx] = P_gen
            Q[controller.gen_idx] = Q_gen

            V, success = env.step(P + 1j*Q)
            V_prev = V[1:]
            
            if np.any(V>env.v_upper) | np.any(V<env.v_lower):
                violation_count += 1
            writer.add_scalar("V/max", max(V[1:]), t)
            writer.add_scalar("V/min", min(V[1:]), t)
            
            cost = np.clip(P_av - P_gen, 0, None)
            loss += cost
            
            V_record.append(V[1:])
            P_record.append(P)
            Q_record.append(Q)
             
        writer.add_scalar("Loss", loss.mean().item(), i)
        writer.add_scalar("violations", violation_count, i)
        
        if (i % 20 ==0) & (i>0):
            np.save(f"results/V_{args.exp_name}.npy", np.array(V_record))
            np.save(f"results/P_{args.exp_name}.npy", np.array(P_record))
            np.save(f"results/Q_{args.exp_name}.npy", np.array(Q_record))
            
    np.save(f"results/V_{args.exp_name}.npy", np.array(V_record))
    np.save(f"results/P_{args.exp_name}.npy", np.array(P_record))
    np.save(f"results/Q_{args.exp_name}.npy", np.array(Q_record))
            
if __name__ == '__main__':
    main()

'''
    # Example Usage of the environment
    t = 10
    Sbus = env.getSbus(t)
    
    # Solve power flow equations
    V, success = env.step(Sbus)
    print(np.abs(V))
    if success == 0:
        print("Something is wrong")
    
    # Estimation using the linearized model
    V_est = env.linear_estimate(Sbus)
    print(V_est)
'''
