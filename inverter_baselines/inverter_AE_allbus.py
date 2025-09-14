import os, sys, argparse, pickle
import numpy as np
import torch
import time
from tqdm import tqdm

main_path = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.insert(0, main_path)

from env.inverter import IEEE37
from agents.ae_policy_allbus import Net, AEControllerAllBus
from utils.inverter_utils import Replay_Memory

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='AE Projection (no-retrain) with all-bus input')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--exp_name', type=str, default='AE_allbus')
parser.add_argument('--ae_model_path', type=str, help='path to pretrained AE .pt from phase2', default='phase2_ieee37bus_1_decoders_72_72_absolute_Adam.pt')
parser.add_argument('--ae_norm_path', type=str, help='path to normalization_params.npz used in training', default="normalization_params.npz")
parser.add_argument('--use_cvx_fallback', action='store_true', help='enable exact fallback projection like PROF')
parser.add_argument('--network_name', type=str, default='ieee37')
args = parser.parse_args()


class StatsLogger:
    def __init__(self, exp_name):
        self.exp_name = exp_name
        self.stats = {'V_max': [], 'V_min': [], 'Loss': [], 'violations': [], 'episodes': [], 'timesteps': [], 'proj_count': [], 'fallback_count': [], 'inference_times': [], 'curtailment': [], 'curtailment_episodes': []}
    def add_scalar(self, name, value, step):
        if name.startswith('V/'):
            metric = name.replace('V/', 'V_')
            self.stats[metric].append(value)
            self.stats['timesteps'].append(step)
        elif name in ['Loss', 'violations', 'proj_count', 'fallback_count']:
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
                for i, (ep, loss, viol, proj, fallback) in enumerate(zip(self.stats['episodes'], self.stats['Loss'], self.stats['violations'], self.stats['proj_count'], self.stats['fallback_count'])):
                    f.write(f"Episode {ep:4d}: Loss={loss:8.4f}, Violations={viol:4d}, Proj={proj:4d}, Fallback={fallback:4d}\n")
            if self.stats['V_max']:
                f.write("\nVoltage summary\n")
                f.write(f"Max: {max(self.stats['V_max']):.4f}, Min: {min(self.stats['V_min']):.4f}\n")
            if self.stats['inference_times']:
                f.write("\nInference Time Statistics\n")
                f.write(f"Mean: {np.mean(self.stats['inference_times']):.6f}s, ")
                f.write(f"Std: {np.std(self.stats['inference_times']):.6f}s, ")
                f.write(f"Min: {min(self.stats['inference_times']):.6f}s, ")
                f.write(f"Max: {max(self.stats['inference_times']):.6f}s\n")
            if self.stats['fallback_count']:
                total_fallbacks = sum(self.stats['fallback_count'])
                total_episodes = len(self.stats['fallback_count'])
                f.write(f"\nFallback Usage Summary\n")
                f.write(f"Total fallbacks used: {total_fallbacks}\n")
                f.write(f"Average fallbacks per episode: {total_fallbacks/total_episodes:.2f}\n")
            if self.stats['curtailment']:
                f.write(f"\nCurtailment Statistics\n")
                f.write(f"Mean: {np.mean(self.stats['curtailment']):.6f}, ")
                f.write(f"Std: {np.std(self.stats['curtailment']):.6f}, ")
                f.write(f"Min: {min(self.stats['curtailment']):.6f}, ")
                f.write(f"Max: {max(self.stats['curtailment']):.6f}\n")
                f.write(f"Total curtailment: {sum(self.stats['curtailment']):.6f}\n")

def main():
    torch.manual_seed(args.seed)
    os.makedirs("results", exist_ok=True)
    logger = StatsLogger(args.exp_name)

    if args.network_name != 'ieee37':
        raise NotImplementedError

    env = IEEE37()
    n_bus_pq = env.n - 1
    n_inverters = len(env.gen_idx)

    env_params = {
        'V0': env.V0[-env.n_pq:],
        'P0': env.P0[-env.n_pq:],
        'Q0': env.Q0[-env.n_pq:],
        'n_bus': n_bus_pq,
        'gen_idx': env.gen_idx - 1,
        'V_upper': env.v_upper, 'V_lower': env.v_lower,
        'S_rating': env.max_S,
        'H': np.hstack([env.R, env.B])
    }
    scaler = 1000

    nn_policy = Net(n_bus_pq, n_inverters, [256, 128, 64], [16, 4])
    memory = Replay_Memory()
    policy = AEControllerAllBus(
        network=nn_policy,
        memory=memory,
        lr=args.lr,
        ae_model_path=args.ae_model_path,
        ae_norm_path=args.ae_norm_path,
        scaler=scaler,
        use_cvx_fallback=args.use_cvx_fallback,
        **env_params
    ).to(DEVICE)

    # 1-week data
    num_steps = 900
    n_episodes = 7 * 86400 // num_steps

    V_prev = np.zeros(n_bus_pq)
    V_record, P_record, Q_record, Curtailment_record = [], [], [], []

    for i in tqdm(range(n_episodes), desc="Training Episodes"):
        loss = 0.0
        violation_count = 0
        fallback_count = 0
        episode_inference_times = []

        it = tqdm(range(num_steps), desc=f"Episode {i+1} Steps", leave=False, disable=(i >= 5)) if i < 5 else range(num_steps)
        for k in it:
            t = i * num_steps + k
            Sbus, P_av = env.getSbus(t)   # per-unit
            Sbus *= scaler                # scaled for state, will be unscaled inside controller
            state = np.concatenate([V_prev, np.real(Sbus), np.imag(Sbus)])
            policy.memory.append((state, Sbus, P_av))
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            # Track inference time and fallback usage
            prev_proj_count = policy.proj_count
            start_time = time.time()
            P_pu, Q_pu = policy(state_t, Sbus, P_av=P_av)  # per-unit PQ-bus deltas
            inference_time = time.time() - start_time
            
            # Check if fallback was used (proj_count increased)
            if policy.proj_count > prev_proj_count:
                fallback_count += 1
            
            episode_inference_times.append(inference_time)
            V, success = env.step(P_pu + 1j * Q_pu)
            V_prev = V[1:]

            if np.any(V > env.v_upper) or np.any(V < env.v_lower):
                violation_count += 1
            logger.add_scalar("V/max", float(np.max(V[1:])), t)
            logger.add_scalar("V/min", float(np.min(V[1:])), t)

            cost = np.clip(P_av - P_pu[policy.gen_idx_pq], 0, None)
            curtailment = np.mean(cost)
            loss += float(curtailment)

            V_record.append(V[1:])
            P_record.append(P_pu)
            Q_record.append(Q_pu)
            Curtailment_record.append(cost)

            if (k % 900 == 0) and (t > 0):
                policy.update()

        logger.add_scalar("Loss", loss / num_steps, i)
        logger.add_scalar("violations", violation_count, i)
        logger.add_scalar("proj_count", policy.proj_count, i)
        logger.add_scalar("fallback_count", fallback_count, i)
        logger.add_scalar("curtailment", loss / num_steps, i)
        
        # Log inference time statistics for this episode
        if episode_inference_times:
            mean_inference_time = np.mean(episode_inference_times)
            logger.add_scalar("inference_time", mean_inference_time, i)
        
        policy.proj_count = 0

        if (i % 20 == 0) and (i > 0):
            np.save(f"results/V_{args.exp_name}.npy", np.array(V_record))
            np.save(f"results/P_{args.exp_name}.npy", np.array(P_record))
            np.save(f"results/Q_{args.exp_name}.npy", np.array(Q_record))
            np.save(f"results/Curtailment_{args.exp_name}.npy", np.array(Curtailment_record))
            torch.save({
                'episode': i,
                'model_state_dict': policy.nn.state_dict(),
                'optimizer_state_dict': policy.optimizer.state_dict(),
                'args': args,
            }, f"results/model_{args.exp_name}_episode_{i}.pt")
            logger.save()
            print(f"Saved checkpoint at episode {i}")

    np.save(f"results/V_{args.exp_name}.npy", np.array(V_record))
    np.save(f"results/P_{args.exp_name}.npy", np.array(P_record))
    np.save(f"results/Q_{args.exp_name}.npy", np.array(Q_record))
    np.save(f"results/Curtailment_{args.exp_name}.npy", np.array(Curtailment_record))
    torch.save({
        'episode': n_episodes,
        'model_state_dict': policy.nn.state_dict(),
        'optimizer_state_dict': policy.optimizer.state_dict(),
        'args': args,
    }, f"results/model_{args.exp_name}_final.pt")
    logger.save()
    print(f"Training complete. Final model saved as model_{args.exp_name}_final.pt")

if __name__ == '__main__':
    main()
