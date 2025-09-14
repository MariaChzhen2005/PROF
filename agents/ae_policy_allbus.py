import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import operator
from functools import reduce

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _infer_ae_arch(state_dict, input_dim):
    # hidden_dim from first encoder linear weight
    if "encoder.0.weight" not in state_dict:
        raise ValueError("State dict missing key encoder.0.weight; cannot infer hidden_dim.")
    hidden_dim = state_dict["encoder.0.weight"].shape[0]
    # num_decoders from keys like "decoders.{k}.0.weight"
    dec_idxs = []
    for k in state_dict.keys():
        if k.startswith("decoders.") and k.endswith(".0.weight"):
            try:
                d = int(k.split(".")[1])
                dec_idxs.append(d)
            except Exception:
                pass
    num_decoders = (max(dec_idxs) + 1) if dec_idxs else 1
    # latent_dim equals input_dim in your training
    latent_dim = input_dim
    return latent_dim, hidden_dim, num_decoders


class TrainedAutoencoder(nn.Module):
    """
    Matches the training architecture in autoencoder.py:
      - encoder: 5x Linear+ReLU, then Linear->Tanh to latent_dim
      - decoders: ModuleList of decoder MLPs
      - gating network: combines decoder outputs
      - feasibility_predictor_nn: present in state_dict; not used here
    Operates on normalized inputs and produces normalized outputs.
    """
    def __init__(self, input_dim, latent_dim, hidden_dim, num_decoders):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_decoders = num_decoders

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()
        )
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(num_decoders)
        ])
        self.gating_network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_decoders),
            nn.Softmax(dim=-1)
        )
        # Feasibility predictor head (not used in forward)
        self.feasibility_predictor_nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def load_pretrained(self, state_dict_path):
        sd = torch.load(state_dict_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if missing:
            # Typically none or only buffers if any
            print(f"Warning: missing keys when loading AE: {missing}")
        if unexpected:
            print(f"Warning: unexpected keys when loading AE: {unexpected}")

    def encode(self, x_norm):
        return self.encoder(x_norm)

    def decode(self, z):
        if self.num_decoders == 1:
            return self.decoders[0](z)
        gate = self.gating_network(z)                 # [B, D]
        outs = torch.stack([d(z) for d in self.decoders], dim=1)  # [B, D, in_dim]
        gate = gate.unsqueeze(-1)                     # [B, D, 1]
        return (outs * gate).sum(dim=1)              # [B, in_dim]

    def project(self, x_norm, latent_geom="hypersphere", latent_radius=0.5):
        # x_norm: normalized input [B, input_dim]
        z = self.encode(x_norm)
        if latent_geom == "hypersphere":
            z_norm = torch.norm(z, dim=1, keepdim=True)
            z = torch.where(z_norm > latent_radius, z * (latent_radius / (z_norm + 1e-8)), z)
        elif latent_geom == "hypercube":
            z = torch.clamp(z, min=-latent_radius, max=latent_radius)
        x_proj_norm = self.decode(z)
        return x_proj_norm


class Net(nn.Module):
    """
    Policy backbone producing raw P/Q for inverters (scaled units), same as your Net.
    """
    def __init__(self, n_bus, n_inverters, shared_hidden_layer_sizes, indiv_hidden_layer_sizes, n_input=3):
        super().__init__()
        layer_sizes = [n_input * n_bus] + shared_hidden_layer_sizes[:-1]
        layers = reduce(operator.add,
                        [[nn.Linear(a, b), nn.ReLU()] for a, b in zip(layer_sizes[:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], shared_hidden_layer_sizes[-1])]
        self.base_net = nn.Sequential(*layers)

        layer_sizes = [shared_hidden_layer_sizes[-1]] + indiv_hidden_layer_sizes
        layers = reduce(operator.add,
                        [[nn.Linear(a, b), nn.ReLU()] for a, b in zip(layer_sizes[:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], 2)]
        indiv_model = nn.Sequential(*layers)
        self.inverter_nets = nn.ModuleList([deepcopy(indiv_model) for _ in range(n_inverters)])

    def forward(self, state):
        z = self.base_net(state)
        res = [inv(z) for inv in self.inverter_nets]
        Ps = torch.cat([x[:, [0]] for x in res], dim=1)
        Qs = torch.cat([x[:, [1]] for x in res], dim=1)
        return Ps, Qs


class AEControllerAllBus(nn.Module):
    """
    No-retrain AE wrapper:
      - Build full absolute [P_net, Q_net] (all buses) in the same normalization as training.
      - Insert policy proposals at inverter buses; keep non-inverter buses fixed.
      - AE latent projection on full vector.
      - Mask back non-inverters to keep them unchanged; take only inverter entries from AE output.
      - Optionally use exact convex fallback like PROF to guarantee feasibility.
    Outputs per-unit PQ-bus vectors for env.step.
    """
    def __init__(self, network, memory, lr,
                 ae_model_path, ae_norm_path,
                 scaler=1000, use_cvx_fallback=True, lam=10,
                 **env_params):
        super().__init__()
        self.nn = network
        self.optimizer = optim.RMSprop(self.nn.parameters(), lr=lr)
        self.memory = memory
        self.ReLU = nn.ReLU()
        self.lam = lam
        self.scaler = scaler
        self.use_cvx_fallback = use_cvx_fallback
        self.proj_count = 0

        # Env params (PQ-bus frame)
        self.n_bus_pq = env_params['n_bus']  # n-1
        self.gen_idx_pq = env_params['gen_idx']  # indices in PQ-bus frame
        self.other_idx_pq = [i for i in range(self.n_bus_pq) if i not in self.gen_idx_pq]
        H = env_params['H']  # [n_pq, 2*n_pq]
        R = H[:, :self.n_bus_pq]
        B = H[:, self.n_bus_pq:]
        # Reorder H with inverter-first columns as in PROF
        R_new = np.vstack([
            np.hstack([R[self.gen_idx_pq][:, self.gen_idx_pq], R[self.gen_idx_pq][:, self.other_idx_pq]]),
            np.hstack([R[self.other_idx_pq][:, self.gen_idx_pq], R[self.other_idx_pq][:, self.other_idx_pq]])
        ])
        B_new = np.vstack([
            np.hstack([B[self.gen_idx_pq][:, self.gen_idx_pq], B[self.gen_idx_pq][:, self.other_idx_pq]]),
            np.hstack([B[self.other_idx_pq][:, self.gen_idx_pq], B[self.other_idx_pq][:, self.other_idx_pq]])
        ])
        H_new = np.hstack([R_new, B_new])

        self.V0_pq = env_params['V0']          # [n_pq]
        self.P0_pq = env_params['P0']          # [n_pq]
        self.Q0_pq = env_params['Q0']          # [n_pq]
        self.V_upper = env_params['V_upper']
        self.V_lower = env_params['V_lower']
        self.S_rating = env_params['S_rating']  # [n_inv]
        # Torch buffers for feasibility checks (PQ-bus)
        self.register_buffer('V0_torch', torch.tensor(self.V0_pq, dtype=torch.float32))
        self.register_buffer('V_upper_torch', torch.tensor(self.V_upper, dtype=torch.float32))
        self.register_buffer('V_lower_torch', torch.tensor(self.V_lower, dtype=torch.float32))
        self.register_buffer('H_torch', torch.tensor(H_new, dtype=torch.float32))
        self.register_buffer('P0_torch', torch.tensor(self.P0_pq, dtype=torch.float32))
        self.register_buffer('Q0_torch', torch.tensor(self.Q0_pq, dtype=torch.float32))
        self.register_buffer('S_rating_torch', torch.tensor(self.S_rating, dtype=torch.float32))

        # Map between PQ-bus frame (length n-1) and full-bus frame (length n)
        self.n_full = self.n_bus_pq + 1
        self.gen_idx_full = np.array(self.gen_idx_pq) + 1  # shift by 1 (slack at 0)

        # Load AE normalization
        norm = np.load(ae_norm_path)
        mean = norm['mean'].astype(np.float32)
        std = norm['std'].astype(np.float32)
        self.ae_in_dim = int(mean.shape[0])
        if std.shape[0] != self.ae_in_dim:
            raise ValueError(f"Norm std length {std.shape[0]} != mean length {self.ae_in_dim}")
        # Determine whether AE was trained on full buses (2*n_full) or PQ-only (2*n_pq)
        if self.ae_in_dim == 2 * self.n_full:
            self.ae_mode = "full"
        elif self.ae_in_dim == 2 * self.n_bus_pq:
            self.ae_mode = "pq"
        else:
            raise ValueError(f"AE input dim {self.ae_in_dim} doesn't match 2*n ({2*self.n_full}) or 2*n_pq ({2*self.n_bus_pq}).")

        # Build AE, inferring hidden_dim and decoders from checkpoint
        sd = torch.load(ae_model_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd_shallow = sd["state_dict"]
        else:
            sd_shallow = sd
        latent_dim, hidden_dim, num_decoders = _infer_ae_arch(sd_shallow, self.ae_in_dim)
        self.ae = TrainedAutoencoder(self.ae_in_dim, latent_dim, hidden_dim, num_decoders).to(DEVICE)
        self.ae.load_pretrained(ae_model_path)
        for p in self.ae.parameters():
            p.requires_grad = False

        # Store normalization tensors
        self.register_buffer('ae_mean', torch.tensor(mean, dtype=torch.float32).view(1, -1))
        self.register_buffer('ae_std', torch.tensor(std, dtype=torch.float32).view(1, -1))

        # Optional exact convex fallback identical to PROF
        if self.use_cvx_fallback:
            n_inv = len(self.gen_idx_pq)
            P = cp.Variable(n_inv)  # per-unit deltas wrt reference
            Q = cp.Variable(n_inv)
            P_tilde = cp.Parameter(n_inv)
            Q_tilde = cp.Parameter(n_inv)
            P_nc = cp.Parameter(len(self.other_idx_pq))
            Q_nc = cp.Parameter(len(self.other_idx_pq))
            P_av = cp.Parameter(n_inv)
            z = cp.hstack([P, P_nc, Q, Q_nc])
            constraints = [
                self.V_lower - self.V0_pq <= H_new @ z,
                H_new @ z <= self.V_upper - self.V0_pq
            ]
            PQ = cp.vstack([self.P0_pq[self.gen_idx_pq] + P, self.Q0_pq[self.gen_idx_pq] + Q])
            constraints += [
                0 <= self.P0_pq[self.gen_idx_pq] + P,
                self.P0_pq[self.gen_idx_pq] + P <= P_av,
                cp.norm(PQ, axis=0) <= self.S_rating
            ]
            obj = cp.Minimize(cp.sum_squares(P - P_tilde) + cp.sum_squares(Q - Q_tilde))
            prob = cp.Problem(obj, constraints)
            self.proj_layer = CvxpyLayer(prob, variables=[P, Q], parameters=[P_tilde, Q_tilde, P_nc, Q_nc, P_av])
        else:
            self.proj_layer = None

    def _is_feasible(self, P_inv_pq, Q_inv_pq, P_nc_pq, Q_nc_pq, P_av):
        eps = 1e-6
        z = torch.cat([
            P_inv_pq,
            torch.tensor(P_nc_pq, dtype=torch.float32, device=DEVICE),
            Q_inv_pq,
            torch.tensor(Q_nc_pq, dtype=torch.float32, device=DEVICE)
        ], dim=-1)
        v = self.H_torch.matmul(z)
        if torch.any(v < self.V_lower_torch - self.V0_torch - eps) or torch.any(v > self.V_upper_torch - self.V0_torch + eps):
            return False
        P_abs_inv = P_inv_pq + self.P0_torch[self.gen_idx_pq]
        Q_abs_inv = Q_inv_pq + self.Q0_torch[self.gen_idx_pq]
        if torch.any(torch.linalg.norm(torch.stack([P_abs_inv, Q_abs_inv], dim=0), dim=0) > self.S_rating_torch + eps):
            return False
        P_av_t = torch.tensor(P_av, dtype=torch.float32, device=DEVICE)
        if torch.any(P_abs_inv < 0 - eps) or torch.any(P_abs_inv > P_av_t + eps):
            return False
        return True

    def _build_full_abs_from_pq_delta(self, P_delta_pq, Q_delta_pq):
        # Full absolute P/Q (length n_full)
        P_abs_full = np.zeros(self.n_full, dtype=np.float32)
        Q_abs_full = np.zeros(self.n_full, dtype=np.float32)
        # PQ buses are 1..n_full-1
        P_abs_full[1:] = (self.P0_torch.cpu().numpy() + P_delta_pq).astype(np.float32)
        Q_abs_full[1:] = (self.Q0_torch.cpu().numpy() + Q_delta_pq).astype(np.float32)
        # Approx slack to enforce sum to zero (lossless approx)
        P_abs_full[0] = -np.sum(P_abs_full[1:])
        Q_abs_full[0] = -np.sum(Q_abs_full[1:])
        return P_abs_full, Q_abs_full

    def _extract_pq_delta_from_full_abs(self, P_abs_full, Q_abs_full):
        P_delta_pq = (P_abs_full[1:] - self.P0_torch.cpu().numpy()).astype(np.float32)
        Q_delta_pq = (Q_abs_full[1:] - self.Q0_torch.cpu().numpy()).astype(np.float32)
        return P_delta_pq, Q_delta_pq

    def forward(self, state, Sbus_scaled, P_av, inference_flag=True):
        """
        Inputs:
          - state: torch [1, ...] policy input
          - Sbus_scaled: numpy complex (PQ buses) already multiplied by scaler for state construction
          - P_av: numpy array per-unit available power at inverter buses
        Outputs:
          - P_pq_pu, Q_pq_pu: numpy arrays length n_pq (per-unit deltas wrt reference), for env.step
        """
        # Base PQ-bus deltas (per-unit) from env at time t
        P_delta_base = Sbus_scaled.real / self.scaler
        Q_delta_base = Sbus_scaled.imag / self.scaler

        # Policy raw actions (scaled) -> per-unit deltas for inverters
        P_raw_scaled, Q_raw_scaled = self.nn(state.to(DEVICE))
        P_raw_pu = (P_raw_scaled / self.scaler).detach().cpu().numpy()  # [batch_size, n_inv]
        Q_raw_pu = (Q_raw_scaled / self.scaler).detach().cpu().numpy()
        
        # Handle both single inference (batch_size=1) and batch training cases
        batch_size = P_raw_pu.shape[0]
        if batch_size == 1:
            P_raw_pu = P_raw_pu.squeeze(0)  # [n_inv]
            Q_raw_pu = Q_raw_pu.squeeze(0)

        # For training mode, return raw policy outputs without AE projection
        if not inference_flag:
            # Return raw policy outputs with gradients preserved
            P_inv = P_raw_scaled / self.scaler  # Keep gradients
            Q_inv = Q_raw_scaled / self.scaler  # Keep gradients
            if batch_size == 1:
                P_inv = P_inv.squeeze(0).unsqueeze(0)  # [1, n_inv]
                Q_inv = Q_inv.squeeze(0).unsqueeze(0)  # [1, n_inv]
            return P_inv, Q_inv, torch.tensor(0.0, device=DEVICE)

        # Inference mode - single sample case only
        P_delta_candidate = P_delta_base.copy()
        Q_delta_candidate = Q_delta_base.copy()
        P_delta_candidate[self.gen_idx_pq] = P_raw_pu
        Q_delta_candidate[self.gen_idx_pq] = Q_raw_pu

        # Build absolute all-bus vector expected by AE, normalized
        if self.ae_mode == "full":
            P_abs_full_in, Q_abs_full_in = self._build_full_abs_from_pq_delta(P_delta_candidate, Q_delta_candidate)
            x_in = np.concatenate([P_abs_full_in, Q_abs_full_in], axis=0).astype(np.float32)[None, :]  # [1, 2*n_full]
        else:  # pq-only AE
            P_abs_pq = (self.P0_torch.cpu().numpy() + P_delta_candidate).astype(np.float32)
            Q_abs_pq = (self.Q0_torch.cpu().numpy() + Q_delta_candidate).astype(np.float32)
            x_in = np.concatenate([P_abs_pq, Q_abs_pq], axis=0).astype(np.float32)[None, :]  # [1, 2*n_pq]

        x_norm = (torch.tensor(x_in, dtype=torch.float32, device=DEVICE) - self.ae_mean) / (self.ae_std + 1e-8)
        x_proj_norm = self.ae.project(x_norm, latent_geom="hypersphere", latent_radius=0.5)
        x_proj = (x_proj_norm * (self.ae_std + 1e-8) + self.ae_mean).detach().cpu().numpy().squeeze(0)

        # Extract AE-projected absolute values
        if self.ae_mode == "full":
            P_abs_full_out = x_proj[:self.n_full]
            Q_abs_full_out = x_proj[self.n_full:]
            # Mask: keep non-inverter buses unchanged (loads), use AE outputs on inverter buses only
            P_abs_full_masked = P_abs_full_in.copy()
            Q_abs_full_masked = Q_abs_full_in.copy()
            P_abs_full_masked[self.gen_idx_full] = P_abs_full_out[self.gen_idx_full]
            Q_abs_full_masked[self.gen_idx_full] = Q_abs_full_out[self.gen_idx_full]
            # Convert back to PQ deltas (env.step expects deltas wrt reference)
            P_delta_pq, Q_delta_pq = self._extract_pq_delta_from_full_abs(P_abs_full_masked, Q_abs_full_masked)
        else:
            # pq-only AE; mask non-inverter PQ entries
            P_abs_pq_out = x_proj[:self.n_bus_pq]
            Q_abs_pq_out = x_proj[self.n_bus_pq:]
            P_abs_pq_in = (self.P0_torch.cpu().numpy() + P_delta_candidate)
            Q_abs_pq_in = (self.Q0_torch.cpu().numpy() + Q_delta_candidate)
            P_abs_pq_masked = P_abs_pq_in.copy()
            Q_abs_pq_masked = Q_abs_pq_in.copy()
            P_abs_pq_masked[self.gen_idx_pq] = P_abs_pq_out[self.gen_idx_pq]
            Q_abs_pq_masked[self.gen_idx_pq] = Q_abs_pq_out[self.gen_idx_pq]
            P_delta_pq = (P_abs_pq_masked - self.P0_torch.cpu().numpy()).astype(np.float32)
            Q_delta_pq = (Q_abs_pq_masked - self.Q0_torch.cpu().numpy()).astype(np.float32)

        if inference_flag:
            # Feasibility check in PQ-bus space (same as PROF)
            P_inv = torch.tensor(P_delta_pq[self.gen_idx_pq], dtype=torch.float32, device=DEVICE)
            Q_inv = torch.tensor(Q_delta_pq[self.gen_idx_pq], dtype=torch.float32, device=DEVICE)
            P_nc = P_delta_pq[self.other_idx_pq]
            Q_nc = Q_delta_pq[self.other_idx_pq]

            if self._is_feasible(P_inv.detach(), Q_inv.detach(), P_nc, Q_nc, P_av):
                return P_delta_pq, Q_delta_pq

            # Optional exact convex fallback
            if self.use_cvx_fallback and self.proj_layer is not None:
                try:
                    P_sol, Q_sol = self.proj_layer(
                        P_inv.unsqueeze(0),
                        Q_inv.unsqueeze(0),
                        torch.tensor(P_nc, dtype=torch.float32, device=DEVICE).unsqueeze(0),
                        torch.tensor(Q_nc, dtype=torch.float32, device=DEVICE).unsqueeze(0),
                        torch.tensor(P_av, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    )
                    self.proj_count += 1
                    P_delta_pq[self.gen_idx_pq] = P_sol.squeeze(0).detach().cpu().numpy()
                    Q_delta_pq[self.gen_idx_pq] = Q_sol.squeeze(0).detach().cpu().numpy()
                except Exception:
                    # Rare solver failure: zero inverter action
                    P_delta_pq[self.gen_idx_pq] = 0.0
                    Q_delta_pq[self.gen_idx_pq] = 0.0
                return P_delta_pq, Q_delta_pq
            else:
                return P_delta_pq, Q_delta_pq
        # If we reach here, it's inference mode (batch_size == 1)
        return P_delta_pq, Q_delta_pq

    def update(self, batch_size=64, n_batch=16):
        for _ in range(n_batch):
            state, Sbus, P_av = self.memory.sample_batch(batch_size=batch_size)
            P_inv, Q_inv, _ = self.forward(state, Sbus, P_av, inference_flag=False)
            # Simple curtailment loss (per-unit)
            curtail = self.ReLU(torch.tensor(P_av, dtype=torch.float32, device=DEVICE) - P_inv)
            loss = curtail.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()