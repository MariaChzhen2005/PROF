"""
ae_policy.py: This file contains the neural network with autoencoder projection.
"""

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


class Net(nn.Module):
    def __init__(self, n_bus, n_inverters, shared_hidden_layer_sizes, indiv_hidden_layer_sizes, n_input=3):
        super(Net, self).__init__()
        # Shared trunk
        layer_sizes = [n_input * n_bus] + shared_hidden_layer_sizes[:-1]
        layers = reduce(
            operator.add,
            [[nn.Linear(a, b), nn.ReLU()] for a, b in zip(layer_sizes[:-1], layer_sizes[1:])]
        )
        layers += [nn.Linear(layer_sizes[-1], shared_hidden_layer_sizes[-1])]
        self.base_net = nn.Sequential(*layers)

        # Heads per inverter
        layer_sizes = [shared_hidden_layer_sizes[-1]] + indiv_hidden_layer_sizes
        layers = reduce(
            operator.add,
            [[nn.Linear(a, b), nn.ReLU()] for a, b in zip(layer_sizes[:-1], layer_sizes[1:])]
        )
        layers += [nn.Linear(layer_sizes[-1], 2)]  # output p_delta, q_delta (relative to reference) for each inverter
        indiv_model = nn.Sequential(*layers)
        self.inverter_nets = nn.ModuleList([deepcopy(indiv_model) for _ in range(n_inverters)])

    def forward(self, state):
        z = self.base_net(state)
        res = [inv(z) for inv in self.inverter_nets]
        Ps = torch.cat([x[:, [0]] for x in res], dim=1)
        Qs = torch.cat([x[:, [1]] for x in res], dim=1)
        return Ps, Qs

class ConstraintAwareAutoencoder(nn.Module):
    """Constraint-aware autoencoder with hypersphere projection in latent space"""
    def __init__(self, ae_model_path, input_dim=72, latent_dim=72, hidden_dim=64, 
                 num_decoders=1, latent_geom="hypersphere", latent_radius=0.5):
        super(ConstraintAwareAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_decoders = num_decoders
        self.latent_geom = latent_geom
        self.latent_radius = latent_radius
        
        # Build encoder architecture matching the trained model
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()  # Constrains latent space to [-1, 1] hypercube
        )
        
        # Build decoders (mixture of experts)
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(num_decoders)
        ])
        
        # Gating network - determines weights for each decoder
        self.gating_network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_decoders),
            nn.Softmax(dim=-1)  # Ensures weights sum to 1
        )
        
        # Feasibility predictor
        self.feasibility_predictor_nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Load the trained weights
        self._load_ae_weights(ae_model_path)
        
        # Freeze the autoencoder weights (don't train them)
        for param in self.parameters():
            param.requires_grad = False
    
    def _load_ae_weights(self, ae_model_path):
        """Load weights from the trained autoencoder model"""
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        try:
            self.load_state_dict(sd, strict=strict)
        except Exception as e:
            print(f"[AE] Warning: load_state_dict(strict={strict}) failed with: {e}")
            print("[AE] Retrying with strict=False.")
            self.load_state_dict(sd, strict=False)
    
    def encode(self, x):
        """Encode input to latent space"""
        return self.encoder(x)
    
    def decode(self, z):
        """
        Decode latent representations using mixture of experts.
        Args:
            z: Latent representations (batch_size, latent_dim)
        Returns:
            Decoded outputs (batch_size, input_dim)
        """
        gate_weights = self.gating_network(z)
        decoder_outputs = []
        for decoder in self.decoders:
            output = decoder(z)
            decoder_outputs.append(output)
        decoder_outputs = torch.stack(decoder_outputs, dim=0)
        gate_weights = gate_weights.t().unsqueeze(-1)
        # weighted sum of decoder outputs
        mixed_output = (decoder_outputs * gate_weights).sum(dim=0) 
        return mixed_output
    
    def project_to_feasible(self, x):
        """
        Project points to feasible set using the trained latent space structure.
        Points with ||z|| > 0.5 are projected/clipped onto the 0.5-radius sphere.
        """
        z = self.encode(x)
        z_norm = torch.norm(z, dim=1, keepdim=True)
        if self.latent_geom == "hypersphere":
            z_projected = torch.where(z_norm > self.latent_radius, z * (self.latent_radius / z_norm), z)
        elif self.latent_geom == "hypercube":
            z_projected = torch.clamp(z, min=-self.latent_radius, max=self.latent_radius)
        return self.decode(z_projected)
    
    def forward(self, x):
        """
        Forward pass: encode -> project to feasible latent space -> decode
        """
        return self.project_to_feasible(x)


### Neural Network with Autoencoder Projection
class Net(nn.Module):
    def __init__(self, n_bus, n_inverters, shared_hidden_layer_sizes, indiv_hidden_layer_sizes, n_input = 3):
        super(Net, self).__init__()
        #### Multi-headed architecture
        # "Shared" model
        # Set up non-linear network of Linear -> BatchNorm -> ReLU
        layer_sizes = [n_input * n_bus] + shared_hidden_layer_sizes[:-1]
        layers = reduce(operator.add, 
            [[nn.Linear(a,b), nn.ReLU(), ] # nn.BatchNorm1d(b), nn.Dropout(p=0.2)]
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], shared_hidden_layer_sizes[-1])]
        self.base_net = nn.Sequential(*layers)
        
        # Individual inverter model
        layer_sizes = [shared_hidden_layer_sizes[-1]] + indiv_hidden_layer_sizes
        layers = reduce(operator.add, 
            [[nn.Linear(a,b),  nn.ReLU(), ] # nn.BatchNorm1d(b), nn.Dropout(p=0.2)]
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], 2)]  # output p and q
        indiv_model = nn.Sequential(*layers)
        self.inverter_nets = nn.ModuleList(
                [deepcopy(indiv_model) for _ in range(n_inverters)]
                )

    def forward(self, state):
        '''
            Input: Vector of voltage magnitudes and angles, real and reactive power demand
            Output: Vector of inverter P setpoints, vector of inverter Q setpoints
        '''
        # Multi-headed architecture
        z = self.base_net(state)
        res = [inverter(z) for inverter in self.inverter_nets]
        Ps = torch.cat([x[:, [0]] for x in res], dim=1)
        Qs = torch.cat([x[:, [1]] for x in res], dim=1)
        return Ps, Qs


class AEController(nn.Module):
    """
    AE-regularized controller with exact cvxpylayer fallback to ensure fairness.
    Pipeline:
      - Compose full PQ-bus absolute net injection vector from current Sbus (other buses) and
        policy raw outputs for generator buses (converted to absolute).
      - Normalize with training mean/std and pass through AE projector.
      - Extract generator entries from AE output, convert back to deltas.
      - Enforce fast local constraints (0 <= P <= P_av; apparent power limit).
      - If linearized voltage constraints violated, project with cvxpylayer (same as PROF).
      - Return final deltas for all PQ buses.
    """
    def __init__(self, network, memory, lr, ae_model_path, ae_norm_path=None,
                 scaler=1000, latent_geom="hypersphere", latent_radius=0.5,
                 hidden_dim=64, num_decoders=1, lam=10, **env_params):
        super().__init__()
        self.nn = network
        self.optimizer = optim.RMSprop(self.nn.parameters(), lr=lr)
        self.memory = memory
        self.ReLU = nn.ReLU()
        self.lam = lam

        # Environment parameters (all for PQ buses only)
        self.n_bus = env_params['n_bus']              # n_pq
        self.gen_idx = env_params['gen_idx']          # indices within PQ-only indexing
        self.other_idx = [i for i in range(self.n_bus) if i not in self.gen_idx]
        self.V0 = env_params['V0']
        self.P0 = env_params['P0']
        self.Q0 = env_params['Q0']
        self.V_upper = env_params['V_upper']
        self.V_lower = env_params['V_lower']
        self.S_rating = env_params['S_rating']
        H = env_params['H']                           # shape: [n_pq, 2*n_pq]
        R = H[:, :self.n_bus]
        B = H[:, self.n_bus:]

        # Reorder H to [gen, other] for both P and Q blocks to match our z concat
        R_new = np.vstack([np.hstack([R[self.gen_idx][:, self.gen_idx], R[self.gen_idx][:, self.other_idx]]),
                           np.hstack([R[self.other_idx][:, self.gen_idx], R[self.other_idx][:, self.other_idx]])])
        B_new = np.vstack([np.hstack([B[self.gen_idx][:, self.gen_idx], B[self.gen_idx][:, self.other_idx]]),
                           np.hstack([B[self.other_idx][:, self.gen_idx], B[self.other_idx][:, self.other_idx]])])
        H_new = np.hstack([R_new, B_new])

        # Torch buffers/params for GPU move
        self.scaler = scaler
        self.register_buffer("V0_t", torch.as_tensor(self.V0, dtype=torch.float32))
        self.register_buffer("V_upper_t", torch.as_tensor(self.V_upper, dtype=torch.float32))
        self.register_buffer("V_lower_t", torch.as_tensor(self.V_lower, dtype=torch.float32))
        self.register_buffer("H_t", torch.as_tensor(H_new, dtype=torch.float32))
        self.register_buffer("P0_t", torch.as_tensor(self.P0, dtype=torch.float32))
        self.register_buffer("Q0_t", torch.as_tensor(self.Q0, dtype=torch.float32))
        self.register_buffer("S_rating_t", torch.as_tensor(self.S_rating, dtype=torch.float32))

        # Load normalization stats if provided (must match training)
        norm_mean = None
        norm_std = None
        if ae_norm_path is not None:
            try:
                stats = np.load(ae_norm_path)
                norm_mean = stats["mean"]
                norm_std = stats["std"]
                # Accept either 2*n_pq or 2*n form; here we expect 2*n_pq
                if norm_mean.shape[0] != 2 * self.n_bus:
                    print(f"[AE] Warning: normalization mean dim {norm_mean.shape[0]} != 2*n_pq={2*self.n_bus}. "
                          f"Attempting to slice if larger or fail gracefully.")
                    if norm_mean.shape[0] > 2 * self.n_bus:
                        norm_mean = norm_mean[-2*self.n_bus:]
                        norm_std = norm_std[-2*self.n_bus:]
            except Exception as e:
                print(f"[AE] Warning: failed to load ae_norm_path={ae_norm_path}: {e}")

        # AE projector expects 2*n_pq absolute net injections
        self.ae_input_dim = 2 * self.n_bus
        self.projector = AEProjector(
            ae_model_path=ae_model_path,
            input_dim=self.ae_input_dim,
            latent_dim=self.ae_input_dim,      # training used latent_dim = input_dim
            hidden_dim=hidden_dim,
            num_decoders=num_decoders,
            latent_geom=latent_geom,
            latent_radius=latent_radius,
            norm_mean=norm_mean,
            norm_std=norm_std,
            freeze=True
        ).to(DEVICE)

        # Exact projection cvx layer (same as PROF)
        P = cp.Variable(len(self.gen_idx))
        Q = cp.Variable(len(self.gen_idx))
        P_tilde = cp.Parameter(len(self.gen_idx))
        Q_tilde = cp.Parameter(len(self.gen_idx))
        P_nc = cp.Parameter(len(self.other_idx))
        Q_nc = cp.Parameter(len(self.other_idx))
        P_av = cp.Parameter(len(self.gen_idx))

        z = cp.hstack([P, P_nc, Q, Q_nc])  # deltas on PQ buses [gen, other] for P and Q
        constraints = [
            self.V_lower - self.V0 <= H_new @ z,
            H_new @ z <= self.V_upper - self.V0,
        ]
        PQ_abs = cp.vstack([self.P0[self.gen_idx] + P, self.Q0[self.gen_idx] + Q])
        constraints += [
            0 <= self.P0[self.gen_idx] + P,
            self.P0[self.gen_idx] + P <= P_av,
            cp.norm(PQ_abs, axis=0) <= self.S_rating
        ]
        objective = cp.Minimize(cp.sum_squares(P - P_tilde) + cp.sum_squares(Q - Q_tilde))
        problem = cp.Problem(objective, constraints)
        self.proj_layer = CvxpyLayer(
            problem,
            variables=[P, Q],
            parameters=[P_tilde, Q_tilde, P_nc, Q_nc, P_av]
        )

        self.proj_count = 0

    def _is_feasible(self, P_delta_gen, Q_delta_gen, P_nc_delta, Q_nc_delta, P_av):
        # Assemble z=[P_delta_gen,P_nc_delta,Q_delta_gen,Q_nc_delta]
        z = torch.cat([P_delta_gen, P_nc_delta, Q_delta_gen, Q_nc_delta], dim=-1)  # [B, 2*n_pq]
        v = z @ self.H_t.T  # [B, n_pq]
        below = v < (self.V_lower_t - self.V0_t - 1e-6)
        above = v > (self.V_upper_t - self.V0_t + 1e-6)
        if torch.any(below) or torch.any(above):
            return False

        # Apparent power per inverter (absolute)
        P_abs = P_delta_gen + self.P0_t[self.gen_idx]
        Q_abs = Q_delta_gen + self.Q0_t[self.gen_idx]
        S = torch.sqrt(P_abs**2 + Q_abs**2)
        if torch.any(S > self.S_rating_t + 1e-6):
            return False

        # Active power box
        Pav_t = torch.as_tensor(P_av, dtype=torch.float32, device=P_abs.device)
        if torch.any(P_abs < -1e-6) or torch.any(P_abs > Pav_t + 1e-6):
            return False

        return True

    def _box_soc_clamp(self, P_delta_gen, Q_delta_gen, P_av):
        """
        Cheap local clamp:
          1) clamp P_abs to [0, P_av]
          2) project [P_abs, Q_abs] onto circle of radius S_rating if needed
        return deltas again.
        """
        P_abs = P_delta_gen + self.P0_t[self.gen_idx]
        Q_abs = Q_delta_gen + self.Q0_t[self.gen_idx]
        P_av_t = torch.as_tensor(P_av, dtype=torch.float32, device=P_abs.device)

        P_abs = torch.clamp(P_abs, min=0.0, max=P_av_t)

        S = torch.sqrt(P_abs**2 + Q_abs**2) + 1e-12
        over = S > self.S_rating_t
        if torch.any(over):
            scale = (self.S_rating_t / S).where(over, torch.ones_like(S))
            P_abs = P_abs * scale
            Q_abs = Q_abs * scale

        # return deltas
        return P_abs - self.P0_t[self.gen_idx], Q_abs - self.Q0_t[self.gen_idx]

    def forward(self, state, Sbus, P_av, inference_flag=True):
        """
        Inputs:
          state: [B, 3*n_pq] torch.float32
          Sbus:  np.ndarray complex of shape [n_pq], deltas w.r.t reference, scaled by 'scaler'
          P_av:  np.ndarray [n_inverters], per-unit available power at generator buses
        Returns:
          P_all_delta, Q_all_delta: np.ndarray [n_pq], deltas for all PQ buses
        """
        # Build current deltas for non-controllables from Sbus (convert back from 'scaler')
        P_nc_delta = torch.as_tensor(np.real(Sbus)[self.other_idx] / self.scaler, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        Q_nc_delta = torch.as_tensor(np.imag(Sbus)[self.other_idx] / self.scaler, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # Base NN raw deltas for generators
        P_raw_delta, Q_raw_delta = self.nn(state.to(DEVICE))  # [B, n_inverters]
        # Convert to absolute per unit for generators
        P_abs_gen = P_raw_delta + self.P0_t[self.gen_idx].unsqueeze(0)
        Q_abs_gen = Q_raw_delta + self.Q0_t[self.gen_idx].unsqueeze(0)

        # Build full PQ absolute vector [P_abs_pq, Q_abs_pq] for AE input
        B = P_raw_delta.shape[0]
        P_abs_pq = torch.empty(B, self.n_bus, device=DEVICE, dtype=torch.float32)
        Q_abs_pq = torch.empty(B, self.n_bus, device=DEVICE, dtype=torch.float32)
        # Fill other (non-control) buses from current measurements
        P_abs_pq[:, self.other_idx] = P_nc_delta + self.P0_t[self.other_idx].unsqueeze(0)
        Q_abs_pq[:, self.other_idx] = Q_nc_delta + self.Q0_t[self.other_idx].unsqueeze(0)
        # Fill generators with current proposal
        P_abs_pq[:, self.gen_idx] = P_abs_gen
        Q_abs_pq[:, self.gen_idx] = Q_abs_gen

        # AE projection in absolute, normalized space
        x_abs = torch.cat([P_abs_pq, Q_abs_pq], dim=-1)  # [B, 2*n_pq]
        x_proj_abs = self.projector.project(x_abs)

        # Extract generator entries back, convert to deltas
        P_proj_abs_gen = x_proj_abs[:, :self.n_bus][:, self.gen_idx]
        Q_proj_abs_gen = x_proj_abs[:, self.n_bus:][:, self.gen_idx]
        P_delta_gen = P_proj_abs_gen - self.P0_t[self.gen_idx].unsqueeze(0)
        Q_delta_gen = Q_proj_abs_gen - self.Q0_t[self.gen_idx].unsqueeze(0)

        # Fast local clamp for per-inverter constraints
        P_delta_gen, Q_delta_gen = self._box_soc_clamp(P_delta_gen, Q_delta_gen, P_av)

        # Check feasibility (linearized voltage + per-inverter) and fallback if needed
        feasible = self._is_feasible(
            P_delta_gen.squeeze(0), Q_delta_gen.squeeze(0),
            P_nc_delta.squeeze(0), Q_nc_delta.squeeze(0), P_av
        )

        if not feasible:
            try:
                P_star, Q_star = self.proj_layer(
                    P_delta_gen.squeeze(0),
                    Q_delta_gen.squeeze(0),
                    P_nc_delta.squeeze(0),
                    Q_nc_delta.squeeze(0),
                    torch.as_tensor(P_av, dtype=torch.float32, device=DEVICE)
                )
                self.proj_count += 1
                P_delta_gen = P_star.unsqueeze(0)
                Q_delta_gen = Q_star.unsqueeze(0)
            except Exception as e:
                # Extremely rare solver failure fallback: zero controls
                print(f"[AE] cvxpylayer projection failed, zeroing control. Error: {e}")
                P_delta_gen = torch.zeros_like(P_delta_gen)
                Q_delta_gen = torch.zeros_like(Q_delta_gen)

        # Assemble final deltas for all PQ buses
        P_all_delta = torch.empty(B, self.n_bus, device=DEVICE, dtype=torch.float32)
        Q_all_delta = torch.empty(B, self.n_bus, device=DEVICE, dtype=torch.float32)
        P_all_delta[:, self.other_idx] = P_nc_delta
        Q_all_delta[:, self.other_idx] = Q_nc_delta
        P_all_delta[:, self.gen_idx] = P_delta_gen
        Q_all_delta[:, self.gen_idx] = Q_delta_gen

        # Return numpy arrays for env.step (first batch element)
        P_np = P_all_delta[0].detach().cpu().numpy()
        Q_np = Q_all_delta[0].detach().cpu().numpy()
        return P_np, Q_np

    def update(self, batch_size=64, n_batch=16):
        """
        Train only the policy network to reduce curtailment, using AE as fixed projector + fallback.
        """
        for _ in range(n_batch):
            state, Sbus, P_av = self.memory.sample_batch(batch_size=batch_size)
            # Forward in training mode (same as inference)
            P, Q = self.forward(state, Sbus, P_av, inference_flag=False)

            # P here are deltas for all PQ buses; extract generators, convert to absolute
            P_gen_abs = torch.as_tensor(P, dtype=torch.float32, device=DEVICE)[..., self.gen_idx] + self.P0_t[self.gen_idx]
            P_av_t = torch.as_tensor(P_av, dtype=torch.float32, device=DEVICE)
            curtail = self.ReLU(P_av_t - P_gen_abs)

            loss = curtail.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()