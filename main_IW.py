import os
import sys

import gym
import eplus_env

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import numpy as np
import pandas as pd
import copy
import pickle
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.distributions import MultivariateNormal, Normal
from torch.utils.tensorboard import SummaryWriter

from algo.ppo import PPO
from agents.nn_policy import NeuralController
from utils.network import LSTM
from utils.ppo_utils import make_dict, R_func, Advantage_func, Replay_Memory

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE

parser = argparse.ArgumentParser(description='Gnu-RL: Online Learning')
parser.add_argument('--gamma', type=float, default=0.9, metavar='G',
                    help='discount factor (default: 0.9)')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 42)')
parser.add_argument('--lr', type=float, default=5e-4, metavar='G',
                    help='Learning Rate')
parser.add_argument('--lam', type=int, default=10, metavar='N',
                   help='random seed (default: 42)')
parser.add_argument('--epsilon', type=float, default=0.2, metavar='G', help='PPO Clip Parameter')
parser.add_argument('--update_episode', type=int, default=4, metavar='N',
                    help='PPO update episode (default: 1); If -1, do not update weights')
parser.add_argument('--T', type=int, default=12, metavar='N',
                    help='Planning Horizon (default: 12)')
parser.add_argument('--step', type=int, default=300*3, metavar='N',
                    help='Time Step in Simulation, Unit in Seconds (default: 900)') # 15 Minutes Now!
parser.add_argument('--exp_name', type=str, default='nn_w_proj',
                    help='save name')
parser.add_argument('--eta', type=int, default=3,
                    help='Hyper Parameter for Balancing Comfort and Energy')
parser.add_argument('--model_no', type = int, default = 1800, help = '')
args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)
    writer = SummaryWriter(comment = args.exp_name)
    
    # Create Simulation Environment
    env = gym.make('Eplus-IW-test-v0')
    
    # Specify variable names for control problem
    obs_name = ["Outdoor Temp.", "Outdoor RH", "Wind Speed", "Wind Direction", "Diff. Solar Rad.", "Direct Solar Rad.", "HW Enable OA Setpoint", "IW Average PPD", "HW Supply Setpoint", "Indoor Air Temp.", "Indoor Temp. Setpoint", "Occupancy Flag", "Heating Demand"]
    state_name = ["Indoor Air Temp."]
    dist_name = ["Outdoor Temp.", "Outdoor RH", "Wind Speed", "Wind Direction", "Diff. Solar Rad.", "Direct Solar Rad.", "Occupancy Flag"]
    ctrl_name = ["HW Enable OA Setpoint", "HW Supply Setpoint"]
    target_name = ["Indoor Temp. Setpoint"]
    dist_name = dist_name + target_name
    
    n_state = len(state_name)
    n_ctrl = 1 #len(ctrl_name)
    n_dist = len(dist_name)
    eta = [0.1, args.eta] # eta: Weight for comfort during unoccupied and occupied mode
    step = args.step # step: Timestep; Unit in seconds
    T = args.T # T: Number of timesteps in the planning horizon
    tol_eps = 91 # tol_eps: Total number of episodes; Each episode is a natural day

    # Read Information on Weather, Occupancy, and Target Setpoint
    obs_2017 = pd.read_pickle("data/data_2017_baseline.pkl")
    disturbance = obs_2017[dist_name]
    # Min-Max Normalization
    obs_TMY3 = pd.read_pickle("data/data_TMY3_baseline.pkl") # For Min-Max Normalization Only
    dist_min = obs_TMY3[dist_name].min()
    dist_max = obs_TMY3[dist_name].max()
    disturbance = (disturbance - dist_min)/(dist_max - dist_min)
    state_min = obs_TMY3[state_name].min().values
    state_max = obs_TMY3[state_name].max().values
    memory = Replay_Memory()
    
    ## Load pretrained LSTM policy weights
    '''
        Expects all states, actions, and disturbances are MinMaxNormalized; (Based on TMY3 data)
        The LSTM also expects "setpoint" as part of the disturbance term.
    '''
    network = LSTM(n_state, n_ctrl, n_dist)
    network.load_state_dict(torch.load("data/param_IW-nn-{}".format(args.model_no)))
    
    ## Load thermodynamics model to construct the polytope
    '''
        New model also expects states, actions, and disturbances to be MinMaxNormalized
    '''
    model_dict ={'a': np.array([0.934899]),
                'bu': np.array([0.024423]),
                'bd': np.array([5.15795080e-02, -6.92141185e-04, -1.21103548e-02,
                2.38717578e-03, -3.52816030e-03,  3.32528746e-03,  7.19267820e-03]),
                'Pm': 1  # Upper bound of u;
                }
    policy = NeuralController(T, step, network, RC_flag = False, **model_dict)
    agent = PPO(policy, memory, lr = args.lr, clip_param = args.epsilon, lam = args.lam)
    
    dir = 'results'
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    multiplier = 1 # Normalize the reward for better training performance
    n_step = 96 #timesteps per day
    
    sigma = 0.1
    sigma_min = 0.01
    sigma_step = (sigma-sigma_min) * args.update_episode/tol_eps
    
    timeStep, obs, isTerminal = env.reset()
    start_time = pd.datetime(year = env.start_year, month = env.start_mon, day = env.start_day)
    cur_time = start_time
    obs_dict = make_dict(obs_name, obs)
    
    # Save for record
    timeStamp = [start_time]
    observations = [obs]
    actions_taken = []

    for i_episode in range(tol_eps):
        ## Save for Parameter Updates
        rewards = []
        real_rewards = []

        for t in range(n_step):
            state = np.array([obs_dict[name] for name in state_name])
            state = (state-state_min)/(state_max-state_min)
            
            x_upper = obs_2017['x_upper'][cur_time : cur_time + pd.Timedelta(seconds = (T-1) * step)].values
            x_lower = obs_2017['x_lower'][cur_time : cur_time + pd.Timedelta(seconds = (T-1) * step)].values
            ## Margin
            #x_lower+=0.025
            #x_upper-=0.025
            
            x_upper = (x_upper-state_min)/(state_max-state_min)
            x_lower = (x_lower-state_min)/(state_max-state_min)
            
            dt = disturbance[cur_time : cur_time + pd.Timedelta(seconds = (T-1) * step)].values  # T x n_dist
            
            ## Update the model in the controller
            # CVXPY expects np.array for parameters
            agent.policy_old.updateState(state, x_lower = x_lower, x_upper = x_upper, d = dt[:, :-1])
            agent.memory.x_lowers.append(torch.tensor(x_lower).float())
            agent.memory.x_uppers.append(torch.tensor(x_upper).float())
            
            state = torch.tensor(state).unsqueeze(0).float() # 1 x n_state
            dt = torch.tensor(dt).float()
            agent.memory.states.append(state)
            agent.memory.disturbance.append(dt)
            
            ## Use policy_old to select action
            mu, sigma_sq, _ = agent.forward(state, dt.unsqueeze(1), current = False) # mu, sigma_sq: T x 1 x Dim.
            sigma_sq = torch.ones_like(mu) * sigma**2
            
            ## Myopic Limit: A hack to make sure the projected actions do not result in tiny violations
            margin = 0.1/(state_max-state_min)
            u_limits = np.array([x_lower[0]+margin.item(), x_upper[0]-margin.item()]) - model_dict['a'] * state.item() -  model_dict['bd'].dot(dt[0, :-1].numpy())
            u_limits /= model_dict['bu']
            u_limits = np.clip(u_limits, 0, 1)
            #pdb.set_trace()
            action, old_logprob = agent.select_action(mu[0], sigma_sq[0], u_limits = u_limits)
            agent.memory.actions.append(action.detach().clone())
            agent.memory.old_logprobs.append(old_logprob.detach().clone())
            
            SWT = 20 + 45 * action.item()
            if (SWT<30):
                HWOEN = -30 # De Facto Off
                action = torch.zeros_like(action)
                SWT = 20
            else:
                HWOEN = 30 # De Facto On
            if np.isnan(SWT):
                SWT = 20
            action4env = (HWOEN, SWT)
            
            # Before step
            print(f'{cur_time}: IAT={obs_dict["Indoor Air Temp."]}, Occupied={obs_dict["Occupancy Flag"]}, Control={SWT}')
            for _ in range(3):
                timeStep, obs, isTerminal = env.step(action4env)

            obs_dict = make_dict(obs_name, obs)
            reward = R_func(obs_dict, SWT-20, eta)
            
            # Per step
            real_rewards.append(reward)
            bl = 0#obs_2017['rewards'][cur_time]
            rewards.append((reward-bl) / 15) # multiplier
            # print(f'Reward={reward}, BL={bl}')
            # Save for record
            cur_time = start_time + pd.Timedelta(seconds = timeStep)
            timeStamp.append(cur_time)
            observations.append(obs)
            actions_taken.append(action4env)
        
        writer.add_scalar('Reward', np.mean(real_rewards), i_episode)
        writer.add_scalar('Reward_Diff', np.mean(rewards), i_episode)
        print("{}, reward: {}".format(cur_time, np.mean(real_rewards)))
        
        advantages = Advantage_func(rewards, args.gamma)
        agent.memory.advantages.append(advantages)
        # if -1, do not update parameters
        if args.update_episode == -1:
            agent.memory.clear_memory()
        elif (i_episode >0) & (i_episode % args.update_episode ==0):
            agent.update_parameters(sigma = sigma, K = 8)
            sigma = max(sigma_min, sigma-sigma_step)
            
        obs_df = pd.DataFrame(np.array(observations), index = np.array(timeStamp), columns = obs_name)
        obs_df = obs_df.drop(columns=ctrl_name)
        action_df = pd.DataFrame(np.array(actions_taken), index = np.array(timeStamp[:-1]), columns = ctrl_name)
        obs_df = obs_df.merge(action_df, how = 'left', right_index = True, left_index = True)
        obs_df.to_pickle("results/obs_"+args.exp_name+".pkl")

if __name__ == '__main__':
    main()
