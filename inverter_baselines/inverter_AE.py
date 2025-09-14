"""
inverter_AE.py: This script trains a neural network with autoencoder projection.
"""


import os, sys, argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

main_path = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.insert(0, main_path)

from env.inverter import IEEE37

from algo.ppo import PPO
from agents.ae_policy import Net, AEController
from utils.inverter_utils import Replay_Memory


import pdb

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='GnuRL Demo: Online Learning - Autoencoder Projection Baseline')
parser.add_argument('--gamma', type=float, default=0.98, metavar='G',
                    help='discount factor (default: 0.98)')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 42)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='G',
                    help='Learning Rate')
parser.add_argument('--epsilon', type=float, default=0.2, metavar='G', help='PPO Clip Parameter')
parser.add_argument('--update_episode', type=int, default=4, metavar='N',
                    help='PPO update episode (default: 1); If -1, do not update weights')
parser.add_argument('--exp_name', type=str, default='AE',
                    help='save name')
parser.add_argument('--ae_model_path', type=str, default='phase2_ieee37bus_1_decoders_72_72_absolute_Adam.pt',
                    help='path to trained autoencoder model')
parser.add_argument('--network_name', type=str, default='ieee37',
                    help='')
args = parser.parse_args()


class StatsLogger:
    """Custom logger to save training statistics to multiple formats"""
    def __init__(self, exp_name):
        self.exp_name = exp_name
        self.stats = {
            'V_max': [],
            'V_min': [],
            'Loss': [],
            'violations': [],
            'episodes': [],
            'timesteps': []
        }
        
    def add_scalar(self, name, value, step):
        """Add a scalar value to the statistics"""
        if name.startswith('V/'):
            metric = name.replace('V/', 'V_')
            self.stats[metric].append(value)
            self.stats['timesteps'].append(step)
        elif name in ['Loss', 'violations']:
            self.stats[name].append(value)
            self.stats['episodes'].append(step)
    
    def save_stats(self):
        """Save statistics to pickle, npz, and txt files"""
        # Save as pickle
        with open(f"results/stats_{self.exp_name}.pkl", 'wb') as f:
            pickle.dump(self.stats, f)
        
        # Save as npz
        np.savez(f"results/stats_{self.exp_name}.npz", **self.stats)
        
        # Save as txt (human readable)
        with open(f"results/stats_{self.exp_name}.txt", 'w') as f:
            f.write(f"Training Statistics for {self.exp_name}\n")
            f.write("=" * 50 + "\n\n")
            
            # Episode-level stats
            if self.stats['Loss']:
                f.write("Episode-level Statistics:\n")
                f.write("-" * 25 + "\n")
                for i, (ep, loss, viol) in enumerate(zip(self.stats['episodes'], 
                                                       self.stats['Loss'], 
                                                       self.stats['violations'])):
                    f.write(f"Episode {ep:4d}: Loss={loss:8.4f}, Violations={viol:4d}\n")
                f.write("\n")
            
            # Summary statistics
            if self.stats['V_max']:
                f.write("Voltage Statistics Summary:\n")
                f.write("-" * 28 + "\n")
                f.write(f"Max voltage observed: {max(self.stats['V_max']):.4f}\n")
                f.write(f"Min voltage observed: {min(self.stats['V_min']):.4f}\n")
                f.write(f"Avg max voltage: {np.mean(self.stats['V_max']):.4f}\n")
                f.write(f"Avg min voltage: {np.mean(self.stats['V_min']):.4f}\n")
                f.write("\n")
            
            if self.stats['Loss']:
                f.write("Training Summary:\n")
                f.write("-" * 17 + "\n")
                f.write(f"Total episodes: {len(self.stats['Loss'])}\n")
                f.write(f"Final loss: {self.stats['Loss'][-1]:.4f}\n")
                f.write(f"Average loss: {np.mean(self.stats['Loss']):.4f}\n")
                f.write(f"Total violations: {sum(self.stats['violations'])}\n")
                f.write(f"Average violations per episode: {np.mean(self.stats['violations']):.2f}\n")

def main():
    torch.manual_seed(args.seed)
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Initialize custom logger
    logger = StatsLogger(args.exp_name)
    print(f"Logging enabled - will save to pickle, npz, and txt formats")
    
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
                  'n_bus':n_bus, # Slack bus is not controllable
                  'gen_idx': env.gen_idx - 1, # Excluded the slack bus
                  'V_upper': env.v_upper, 'V_lower': env.v_lower,
                 'S_rating': env.max_S,
                 }
    scaler = 1000 # Note: The value for Sbus is really small; Scale up for better learning
    
    # Use the neural network with autoencoder projection
    nn_policy = Net(n_bus, n_inverters, [256, 128, 64], [16, 4])
    memory = Replay_Memory()
    policy = AEController(nn_policy, memory, args.lr, args.ae_model_path, scaler = scaler, **env_params)
    policy = policy.to(DEVICE)
    
    # 1-week data
    num_steps = 900 # 15 minutes
    n_episodes = 7*86400//num_steps

    V_prev = np.zeros(n_bus)
    
    V_record = []
    V_est_record = []
    P_record = []
    Q_record = []
    
    # Training loop with progress tracking
    for i in tqdm(range(n_episodes), desc="Training Episodes", unit="episode"):
        loss = 0
        violation_count = 0
        
        # Inner loop with progress tracking (only show for first few episodes to avoid clutter)
        step_iterator = tqdm(range(num_steps), desc=f"Episode {i+1} Steps", leave=False, disable=(i >= 5)) if i < 5 else range(num_steps)
        for k in step_iterator:
            t = i*num_steps + k
            Sbus, P_av = env.getSbus(t)
            Sbus *= scaler
            state = np.concatenate([V_prev, np.real(Sbus), np.imag(Sbus)])
            policy.memory.append((state, Sbus, P_av)) ## Everything is np.array!
            
            state = torch.tensor(state).float().unsqueeze(0)
            
            P, Q = policy(state, Sbus, P_av = P_av)
            
            V, success = env.step(P + 1j*Q)
            V_prev = V[1:]
            
            if np.any(V>env.v_upper) | np.any(V<env.v_lower):
                violation_count += 1
            logger.add_scalar("V/max", max(V[1:]), t)
            logger.add_scalar("V/min", min(V[1:]), t)
            
            cost = np.clip(P_av - P[policy.gen_idx], 0, None)
            loss += cost
            
            V_record.append(V[1:])
            P_record.append(P)
            Q_record.append(Q)
            
            if (k % 900 == 0) & (t>0):
                policy.update()
             
        logger.add_scalar("Loss", loss.mean().item(), i)
        logger.add_scalar("violations", violation_count, i)
        
        if (i % 20 ==0) & (i>0):
            # Save results
            np.save(f"results/V_{args.exp_name}.npy", np.array(V_record))
            np.save(f"results/P_{args.exp_name}.npy", np.array(P_record))
            np.save(f"results/Q_{args.exp_name}.npy", np.array(Q_record))
            
            # Save trained model
            torch.save({
                'episode': i,
                'model_state_dict': policy.nn.state_dict(),
                'optimizer_state_dict': policy.optimizer.state_dict(),
                'args': args,
            }, f"results/model_{args.exp_name}_episode_{i}.pt")
            print(f"Saved model checkpoint at episode {i}")
            
            # Save statistics at checkpoints
            logger.save_stats()
            print(f"Saved training statistics at episode {i}")
            
    # Final save of results and model
    np.save(f"results/V_{args.exp_name}.npy", np.array(V_record))
    np.save(f"results/P_{args.exp_name}.npy", np.array(P_record))
    np.save(f"results/Q_{args.exp_name}.npy", np.array(Q_record))
    
    # Save final trained model
    torch.save({
        'episode': n_episodes,
        'model_state_dict': policy.nn.state_dict(),
        'optimizer_state_dict': policy.optimizer.state_dict(),
        'args': args,
    }, f"results/model_{args.exp_name}_final.pt")
    
    # Final save of statistics
    logger.save_stats()
    print(f"Training completed! Final model saved as model_{args.exp_name}_final.pt")
    print(f"Training statistics saved to:")
    print(f"  - results/stats_{args.exp_name}.pkl")
    print(f"  - results/stats_{args.exp_name}.npz") 
    print(f"  - results/stats_{args.exp_name}.txt")
            
if __name__ == '__main__':
    main()
