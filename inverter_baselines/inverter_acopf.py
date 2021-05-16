import os, sys, argparse

import numpy as np
import cvxpy as cp
import ipopt
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

import ipdb

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
parser.add_argument('--exp_name', type=str, default='inverter_ACOPF',
                    help='save name')
parser.add_argument('--network_name', type=str, default='ieee37',
                    help='')
args = parser.parse_args()

class ACOPFController():
    def __init__(self,  **env_params):
        self.n_bus = env_params['n_bus']
        self.V0 = env_params['V0']
        self.P0 = env_params['P0']
        self.Q0 = env_params['Q0']
        self.V_upper = env_params['V_upper']
        self.V_lower = env_params['V_lower']
        self.S_rating = env_params['S_rating']
        self.gen_idx = env_params['gen_idx']
        self.slack_idx = env_params['slack_idx']
        self.Ybus = env_params['Ybus']

        self.A0 = np.zeros(self.V0.shape)   # initial voltage angle
        self.n_gen = len(self.gen_idx)
        self.n_slack = len(self.slack_idx)
        self.other_idx = [i for i in range(self.n_bus) if i not in self.gen_idx and i not in self.slack_idx]


    def solve(self, Sbus, P_av):
        P_nc = Sbus.real[self.other_idx]
        Q_nc = Sbus.imag[self.other_idx]

        # Decision variables: P and Q at controllable buses, 
        #   Pslack and Qslack at slack bus, V and theta at all buses

        # initial guess for decision variables
        x0 = np.hstack([P_av, self.Q0[self.gen_idx], 
            self.P0[self.slack_idx], self.Q0[self.slack_idx], 
            self.V0, self.A0])

        # upper and lower bounds on decision variables
        #   0 \leq P \leq P_av
        #   no explicit bounds on Q
        #   no explicit bounds on Pslack or Qslack
        #   V and A known at ref bus
        #   Vmin \leq V \leq Vmax at non-ref buses
        #   no explicit bounds on A at non-ref buses
        def get_bound_with_slack(bound, slack_bound):
            values = bound * np.ones(self.n_bus)
            values[self.slack_idx] = slack_bound
            return values

        lb = np.hstack([
            np.zeros(self.n_gen), 
            -np.infty * np.ones(self.n_gen),
            -np.infty * np.ones(2 * self.n_slack),
            get_bound_with_slack(self.V_lower, self.V0[self.slack_idx]), 
            get_bound_with_slack(-np.infty, self.A0[self.slack_idx])])
        ub = np.hstack([
            P_av, 
            np.infty * np.ones(self.n_gen),
            np.infty * np.ones(2 * self.n_slack),
            get_bound_with_slack(self.V_upper, self.V0[self.slack_idx]),
            get_bound_with_slack(np.infty, self.A0[self.slack_idx])])

        # upper and lower bounds on other constraints
        #   power flow constraint: diag(v)conj(Ybus)conj(v) - S = 0      
        #      where v = diag(V*exp(1j*A)) and S is net demand at all nodes
        #      (separate out real and imaginary parts)
        #   P^2 + Q^2 \leq S_rating^2
        cl = np.hstack(
            [np.zeros(2*self.n_bus), np.zeros(self.n_gen)])
        cu = np.hstack(
            [np.zeros(2*self.n_bus), self.S_rating**2])

        problem_obj = ACOPFSolver(P_av, P_nc, Q_nc, self.Ybus, 
            self.n_bus, self.n_gen, self.n_slack, self.gen_idx, self.slack_idx, self.other_idx)
        nlp = ipopt.problem(
                    n=len(x0),    # num decision vars
                    m=len(cl),    # num constraints
                    problem_obj=problem_obj,
                    lb=lb,        # lower bounds on decision vars
                    ub=ub,        # upper bounds on decision vars
                    cl=cl,        # lower bounds on constraints
                    cu=cu         # upper bounds on constraints
                    )

        nlp.addOption('tol', 1e-4)
        nlp.addOption('print_level', 0) # 3)

        x, info = nlp.solve(x0)
        P = x[:self.n_gen]
        Q = x[self.n_gen:2*self.n_gen]
        
        return P, Q


class ACOPFSolver(object):
    def __init__(self, P_av, P_nc, Q_nc, Ybus, n_bus, n_gen, n_slack, gen_idx, slack_idx, other_idx):
        self.P_av = P_av
        self.P_nc = P_nc
        self.Q_nc = Q_nc
        self.Ybus = Ybus
        self.n_bus = n_bus
        self.n_gen = n_gen
        self.n_slack = n_slack
        self.gen_idx = gen_idx
        self.slack_idx = slack_idx
        self.other_idx = other_idx
        self.split_inds = np.cumsum(
            [self.n_gen, self.n_gen, self.n_slack, self.n_slack, self.n_bus, self.n_bus])[:-1]

    # Curtailment objective (will be minimized)
    def objective(self, x):
        return np.maximum(self.P_av - x[:self.n_gen], 0).sum()

    # Gradient of objective
    def gradient(self, x):
        p_grad = -1 * ((self.P_av - x[:self.n_gen]) > 0).astype(int)
        return np.hstack([p_grad, np.zeros(self.n_gen + 2*self.n_slack + 2*self.n_bus)])

    # Constraints (excluding box constraints on decision variables)
    def constraints(self, y):
        P, Q, Pslack, Qslack, V, A = np.split(y, self.split_inds)
        
        # power flow constraint [diag(v)conj(Ybus)conj(v) - S = 0]
        #   separate out real and imaginary parts
        voltage = V * np.exp(1j * A)
        net_power = np.zeros(self.n_bus, dtype=np.complex128)
        net_power[self.gen_idx] = P + 1j*Q
        net_power[self.slack_idx] = Pslack + 1j*Qslack
        net_power[self.other_idx] = self.P_nc + 1j*self.Q_nc
        power_mismatch = np.diag(voltage)@np.conj(self.Ybus)@np.conj(voltage) - net_power

        # apparent power at inverters [P^2 + Q^2 \leq S_rating^2; compute left side here]
        apparent_power = P**2 + Q**2

        return np.hstack([np.real(power_mismatch), np.imag(power_mismatch), apparent_power])

    # Jacobian of constraints (excluding box constraints on decision variables)
    def jacobian(self, y):
        P, Q, _, _, V, A = np.split(y, self.split_inds)

        # Jacobian of power flow constraint
        #  See: http://www.cs.cmu.edu/~zkolter/course/15-884/eps_power_flow.pdf
        vol = V * np.exp(1j * A)
        Y = self.Ybus
        J1 = 1j * np.diag(vol) @ (np.diag(np.conj(Y)@np.conj(vol)) - np.conj(Y)@np.diag(np.conj(vol)))
        J2 = np.diag(vol)@np.conj(Y)@np.diag(np.exp(-1j * A)) + \
            np.diag(np.exp(1j * A))@np.diag(np.conj(Y)@np.conj(vol))
        power_flow_jac = np.vstack([
                np.hstack([-np.eye(self.n_bus)[:, self.gen_idx], np.zeros((self.n_bus, self.n_gen)), 
                    -np.eye(self.n_bus)[:, self.slack_idx], np.zeros((self.n_bus, self.n_slack)),
                    np.real(J2), np.real(J1)]),
                np.hstack([np.zeros((self.n_bus, self.n_gen)), -np.eye(self.n_bus)[:, self.gen_idx], 
                    np.zeros((self.n_bus, self.n_slack)), -np.eye(self.n_bus)[:, self.slack_idx],
                    np.imag(J2), np.imag(J1)])
            ])

        # Jacobian of apparent power constraint
        apparent_power_jac = np.hstack([
            np.diag(2*P), np.diag(2*Q), 
            np.zeros( (self.n_gen, 2*self.n_slack + 2*self.n_bus))])

        return np.concatenate([power_flow_jac.flatten(), apparent_power_jac.flatten()])


def main():
    torch.manual_seed(args.seed)
    writer = SummaryWriter(comment = args.exp_name)
    
    # Create Simulation Environment
    if args.network_name == 'ieee37':
        env = IEEE37()
    else:
        print("Not implemented")
    
    n_bus = env.n
    n_inverters = len(env.gen_idx) # inverters at PV panels
    
    env_params = {'V0': env.V0,
                  'P0': env.P0,
                  'Q0': env.Q0,
                  'n_bus': n_bus,
                  'gen_idx': env.gen_idx,
                  'slack_idx': env.ref,
                  'V_upper': env.v_upper, 'V_lower': env.v_lower,
                  'S_rating': env.max_S,
                  'Ybus': env.Ybus
                 }

    controller = ACOPFController(**env_params)

    # 1-week data
    num_steps = 900 # 15 minutes
    n_episodes = 7*86400//num_steps

    V_prev = np.zeros(n_bus)
    
    V_record = []
    V_est_record = []
    P_record = []
    Q_record = []
    
    start_ep = 600
    for i in range(start_ep, min(n_episodes, start_ep + 100)):
        loss = 0
        violation_count = 0
        
        for k in range(num_steps):
            t = i*num_steps + k
            Sbus, P_av = env.getSbus(t, wrt_reference=False, w_slack=True)
            
            P_gen, Q_gen = controller.solve(Sbus, P_av)
            print(f"P_av = {P_av}, P = {P_gen}")

            P = Sbus.real
            Q = Sbus.imag
            P[controller.gen_idx] = P_gen
            Q[controller.gen_idx] = Q_gen

            V, success = env.step(P + 1j*Q, wrt_reference=False)
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
            np.save(f"results/V_{args.exp_name}_{start_ep}.npy", np.array(V_record))
            np.save(f"results/P_{args.exp_name}_{start_ep}.npy", np.array(P_record))
            np.save(f"results/Q_{args.exp_name}_{start_ep}.npy", np.array(Q_record))
            
    np.save(f"results/V_{args.exp_name}_{start_ep}.npy", np.array(V_record))
    np.save(f"results/P_{args.exp_name}_{start_ep}.npy", np.array(P_record))
    np.save(f"results/Q_{args.exp_name}_{start_ep}.npy", np.array(Q_record))

            
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
