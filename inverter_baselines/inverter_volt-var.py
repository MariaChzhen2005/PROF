import os, sys, argparse

import numpy as np
from torch.utils.tensorboard import SummaryWriter

main_path = os.path.abspath(os.path.join(__file__, '..'))
sys.path.insert(0, main_path)

from env.inverter import IEEE37

import pdb

#import torch
#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE

parser = argparse.ArgumentParser(description='GnuRL Demo: Online Learning')
parser.add_argument('--gamma', type=float, default=0.98, metavar='G',
                    help='discount factor (default: 0.98)')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 42)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='G',
                    help='Learning Rate')
parser.add_argument('--epsilon', type=float, default=0.2, metavar='G', help='PPO Clip Parameter')
parser.add_argument('--update_episode', type=int, default=4, metavar='N',
                    help='PPO update episode (default: 1); If -1, do not update weights')
parser.add_argument('--exp_name', type=str, default='volt-var',
                    help='save name')
parser.add_argument('--network_name', type=str, default='ieee37',
                    help='')
args = parser.parse_args()


class VoltVarController():
    def __init__(self, delta, **env_params):
        super(VoltVarController, self).__init__()
        self.V_upper = env_params['V_upper']
        self.V_lower = env_params['V_lower']
        self.delta = delta
        self.gen_idx = env_params['gen_idx']
        self.S_rating = env_params['S_rating']
        self.a = 1/(self.V_upper-1-self.delta/2)

    def forward(self, voltage, P_av):
        Q = (self.S_rating**2-P_av**2)**0.5
        
        # Piece-wise Linear Curve
        voltage = voltage[self.gen_idx]
        out = np.zeros_like(voltage)
        
        out[voltage <= self.V_lower] = 1
        
        idx = (voltage > self.V_lower) & (voltage < 1 - self.delta/2)
        out[idx] = 1 - self.a*(voltage[idx]-self.V_lower)

        idx = (voltage > 1 + self.delta/2) & (voltage < self.V_upper)
        out[idx] = -self.a*(voltage[idx]-1-self.delta/2)
        
        out[voltage >= self.V_upper] = -1
        return out * Q 

def main():
    writer = SummaryWriter(comment = args.exp_name)
    
    # Create Simulation Environment
    if args.network_name == 'ieee37':
        env = IEEE37()
    else:
        print("Not implemented")
    n_bus = env.n
    env_params = {'V0': env.V0[-env.n_pq:],
                  'P0': env.P0[-env.n_pq:],
                  'Q0': env.Q0[-env.n_pq:],
                  'gen_idx': env.gen_idx, # Including the slack bus
                  'V_upper': env.v_upper, 'V_lower': env.v_lower,
                 'S_rating': env.max_S,
                 }

    ## Note: Volt-Var controller considers deviation from 1
    controller = VoltVarController(0.04, **env_params)
    
    # 1-week data
    num_steps = 600 # 10 minutes
    n_episodes = 7*86400//num_steps

    V_prev = np.ones(n_bus)    
    V_record = []
    
    for i in range(n_episodes):
        violation_count = 0
        for k in range(num_steps):
            t = i*num_steps + k
            Sbus, P_av = env.getSbus(t, wrt_reference = False, w_slack = True)
            
            Q = controller.forward(V_prev, P_av = P_av) # at Generation buses
        
            Sbus.imag[env.gen_idx] += Q
            
            V, success = env.step(Sbus)
            V_prev = V
            
            if np.any(V>env.v_upper) | np.any(V<env.v_lower):
                violation_count += 1
            writer.add_scalar("V/max", max(V[1:]), t)
            writer.add_scalar("V/min", min(V[1:]), t)
            
            V_record.append(V[1:])
        
        writer.add_scalar("violations", violation_count, i)
        
        if (i % 20 == 0) & (i>0):
            np.save(f"results/V_{args.exp_name}.npy", np.array(V_record))
<<<<<<< HEAD
=======
            
>>>>>>> 5d88b0ccebcea057216087804a12ef2c880e3345
    np.save(f"results/V_{args.exp_name}.npy", np.array(V_record))
        
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
