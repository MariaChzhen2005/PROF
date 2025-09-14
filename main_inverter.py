import os, sys, argparse, pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from env.inverter import IEEE37

from algo.ppo import PPO
from agents.inverter_policy import Net, NeuralController
from utils.inverter_utils import Replay_Memory


import pdb

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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


class StatsLogger:
    def __init__(self, exp_name):
        self.exp_name = exp_name
        self.stats = {'V_max': [], 'V_min': [], 'Loss': [], 'violations': [], 'episodes': [], 'timesteps': [], 'proj_count': [], 'inference_times': [], 'curtailment': [], 'curtailment_episodes': []}
    def add_scalar(self, name, value, step):
        if name.startswith('V/'):
            metric = name.replace('V/', 'V_')
            self.stats[metric].append(value)
            self.stats['timesteps'].append(step)
        elif name in ['Loss', 'violations', 'proj_count']:
            self.stats[name].append(value)
            self.stats['episodes'].append(step)
        elif name == 'curtailment':
            self.stats['curtailment'].append(value)
            self.stats['curtailment_episodes'].append(step)
        elif name == 'inference_time':
            self.stats['inference_times'].append(value)
    def save(self):
        os.makedirs("results", exist_ok=True)
        with open(f"results/stats_{self.exp_name}.pkl", 'wb') as f:
            pickle.dump(self.stats, f)
        np.savez(f"results/stats_{self.exp_name}.npz", **self.stats)
        with open(f"results/stats_{self.exp_name}.txt", 'w') as f:
            f.write(f"Training Statistics for {self.exp_name}\n")
            f.write("=" * 50 + "\n\n")
            if self.stats['Loss']:
                for i, (ep, loss, viol, proj) in enumerate(zip(self.stats['episodes'], self.stats['Loss'], self.stats['violations'], self.stats['proj_count'])):
                    f.write(f"Episode {ep:4d}: Loss={loss:8.4f}, Violations={viol:4d}, Proj={proj:4d}\n")
            if self.stats['V_max']:
                f.write("\nVoltage summary\n")
                f.write(f"Max: {max(self.stats['V_max']):.4f}, Min: {min(self.stats['V_min']):.4f}\n")
            if self.stats['inference_times']:
                f.write("\nInference Time Statistics\n")
                f.write(f"Mean: {np.mean(self.stats['inference_times']):.6f}s, ")
                f.write(f"Std: {np.std(self.stats['inference_times']):.6f}s, ")
                f.write(f"Min: {min(self.stats['inference_times']):.6f}s, ")
                f.write(f"Max: {max(self.stats['inference_times']):.6f}s\n")
            if self.stats['curtailment']:
                f.write(f"\nCurtailment Statistics\n")
                f.write(f"Mean: {np.mean(self.stats['curtailment']):.6f}, ")
                f.write(f"Std: {np.std(self.stats['curtailment']):.6f}, ")
                f.write(f"Min: {min(self.stats['curtailment']):.6f}, ")
                f.write(f"Max: {max(self.stats['curtailment']):.6f}\n")
                f.write(f"Total curtailment: {sum(self.stats['curtailment']):.6f}\n")


def main():
    torch.manual_seed(args.seed)
    writer = SummaryWriter(comment = args.exp_name)
    logger = StatsLogger(args.exp_name)
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
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
    Curtailment_record = []
    
    # Training loop with progress tracking
    for i in tqdm(range(n_episodes), desc="Training Episodes", unit="episode"):
        loss = 0
        violation_count = 0
        episode_inference_times = []
        
        # Inner loop with progress tracking (only show for first few episodes to avoid clutter)
        step_iterator = tqdm(range(num_steps), desc=f"Episode {i+1} Steps", leave=False, disable=(i >= 5)) if i < 5 else range(num_steps)
        for k in step_iterator:
            t = i*num_steps + k
            Sbus, P_av = env.getSbus(t)
            Sbus *= scaler
            state = np.concatenate([V_prev, np.real(Sbus), np.imag(Sbus)])
            mbp_policy.memory.append((state, Sbus, P_av)) ## Everything is np.array!
            
            state = torch.tensor(state).float().unsqueeze(0)
            
            # Track projection count for monitoring
            prev_proj_count = mbp_policy.proj_count
            P, Q = mbp_policy(state, Sbus, P_av = P_av)
            
            # Get inference time from policy
            if hasattr(mbp_policy, 'get_last_inference_time'):
                inference_time = mbp_policy.get_last_inference_time()
                episode_inference_times.append(inference_time)
            #pdb.set_trace()
            
            V, success = env.step(P + 1j*Q)
            V_prev = V[1:]
            
            if np.any(V>env.v_upper) | np.any(V<env.v_lower):
                violation_count += 1
            writer.add_scalar("V/max", max(V[1:]), t)
            writer.add_scalar("V/min", min(V[1:]), t)
            logger.add_scalar("V/max", float(np.max(V[1:])), t)
            logger.add_scalar("V/min", float(np.min(V[1:])), t)
            
            cost = np.clip(P_av - P[mbp_policy.gen_idx], 0, None)
            curtailment = np.mean(cost)
            loss += cost
            
            V_record.append(V[1:])
            P_record.append(P)
            Q_record.append(Q)
            Curtailment_record.append(cost)
            
            if (k % 900 == 0) & (t>0):
                mbp_policy.update()
             
        writer.add_scalar("Loss", loss.mean().item(), i)
        writer.add_scalar("violations", violation_count, i)
        ## Number of Projection operation during inference time
        writer.add_scalar("proj_count", mbp_policy.proj_count, i)
        
        # Add stats to logger
        logger.add_scalar("Loss", loss.mean().item(), i)
        logger.add_scalar("violations", violation_count, i)
        logger.add_scalar("proj_count", mbp_policy.proj_count, i)
        logger.add_scalar("curtailment", loss.mean().item(), i)
        
        # Log inference time statistics for this episode
        if episode_inference_times:
            mean_inference_time = np.mean(episode_inference_times)
            logger.add_scalar("inference_time", mean_inference_time, i)
        
        mbp_policy.proj_count = 0
        
        if (i % 20 ==0) & (i>0):
            # Save results
            np.save(f"results/V_{args.exp_name}.npy", np.array(V_record))
            np.save(f"results/P_{args.exp_name}.npy", np.array(P_record))
            np.save(f"results/Q_{args.exp_name}.npy", np.array(Q_record))
            np.save(f"results/Curtailment_{args.exp_name}.npy", np.array(Curtailment_record))
            
            # Save trained model
            torch.save({
                'episode': i,
                'model_state_dict': mbp_policy.nn.state_dict(),
                'optimizer_state_dict': mbp_policy.optimizer.state_dict(),
                'args': args,
            }, f"results/model_{args.exp_name}_episode_{i}.pt")
            logger.save()
            print(f"Saved model checkpoint at episode {i}")
            
    # Final save of results and model
    np.save(f"results/V_{args.exp_name}.npy", np.array(V_record))
    np.save(f"results/P_{args.exp_name}.npy", np.array(P_record))
    np.save(f"results/Q_{args.exp_name}.npy", np.array(Q_record))
    np.save(f"results/Curtailment_{args.exp_name}.npy", np.array(Curtailment_record))
    
    # Save final trained model
    torch.save({
        'episode': n_episodes,
        'model_state_dict': mbp_policy.nn.state_dict(),
        'optimizer_state_dict': mbp_policy.optimizer.state_dict(),
        'args': args,
    }, f"results/model_{args.exp_name}_final.pt")
    logger.save()
    print(f"Training completed! Final model saved as model_{args.exp_name}_final.pt")

if __name__ == '__main__':
    main()


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

