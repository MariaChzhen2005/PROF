import os, sys, argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

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
parser.add_argument('--exp_name', type=str, default='inverter',
                    help='save name')
parser.add_argument('--network_name', type=str, default='ieee37',
                    help='')
args = parser.parse_args()


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
    scaler = 1000 # Note: The value for Sbus is really small; Scale up for better learning
    
    mbp_nn = Net(n_bus, n_inverters, [256, 128, 64], [16, 4])
    memory = Replay_Memory()
    mbp_policy = NeuralController(mbp_nn, memory, args.lr, lam = args.lam, scaler = scaler, **env_params)
    mbp_policy = mbp_policy.to(DEVICE)
    
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
            Sbus *= scaler
            state = np.concatenate([V_prev, np.real(Sbus), np.imag(Sbus)])
            mbp_policy.memory.append((state, Sbus, P_av)) ## Everything is np.array!
            
            state = torch.tensor(state).float().unsqueeze(0)
            
            P, Q = mbp_policy(state, Sbus, P_av = P_av)
            V, success = env.step(P.detach().cpu().numpy() + 1j*Q.detach().cpu().numpy())
            V_prev = V[1:]
            
            if np.any(V>env.v_upper) | np.any(V<env.v_lower):
                violation_count += 1
            writer.add_scalar("V/max", max(V[1:]), t)
            writer.add_scalar("V/min", min(V[1:]), t)
            
            cost = torch.clamp(torch.tensor(P_av).float() - P[mbp_policy.gen_idx].cpu(), min =0)
            loss += cost
            
            V_record.append(V[1:])
            P_record.append(P.detach().cpu().numpy())
            Q_record.append(Q.detach().cpu().numpy())
            
            if (k % 900 == 0) & (t>0):
                mbp_policy.update()
             
        writer.add_scalar("Loss", loss.mean().item(), i)
        writer.add_scalar("violations", violation_count, i)
        ## Number of Projection operation during inference time
        writer.add_scalar("proj_count", mbp_policy.proj_count, i)
        mbp_policy.proj_count = 0
        
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
