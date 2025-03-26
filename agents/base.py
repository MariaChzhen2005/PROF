import numpy as np
import cvxpy as cp
import pdb
import torch

# Helper to force conversion to a NumPy array of type float.
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(float)
    return np.asarray(x, dtype=float)

class Controller():
    def __init__(self, T, dt, RC_flag=True, **kwargs):
        # dt: planning timestep; T: planning horizon; RC_flag: RC model flag; **kwargs: model parameters.
        self.T = T
        self.RC_flag = RC_flag
        self.err_count = 0
        
        if RC_flag:
            # RC model parameters.
            self.R = kwargs["R"]
            self.C = kwargs["C"]
            self.Pm = kwargs["Pm"]
            self.eta = kwargs["eta"]
            self.T_sp = kwargs["theta"]
            self.Delta = kwargs["Delta"]
            self.sign = kwargs["sign"]  # (+) for heating; (-) for cooling.
        else:
            # ARX model parameters.
            self.ap = kwargs["a"]
            self.bu = kwargs["bu"]
            self.bd = kwargs["bd"]
            self.p = len(self.ap)
            self.m = len(self.bu)
            self.n_dist = len(self.bd)
            self.Pm = kwargs["Pm"]
            self.T_sp = 75
            self.Delta = 1.8
            
        # Decision variable.
        self.u = cp.Variable(T)
        
        # Parameters for previous control differences and objective.
        self.u_diff = cp.Parameter(T)
        self.v_bar = cp.Parameter(T)
        self.w_bar = cp.Parameter(T)
        self.objective = cp.sum_squares(self.u - self.u_diff - self.v_bar + self.w_bar)

        # Parameters for constraints.
        if RC_flag:
            self.x0 = cp.Parameter()  # scalar in RC case.
            self.d = cp.Parameter(T)
        else:
            self.x0 = cp.Parameter(self.p)
            self.d = cp.Parameter((T, self.n_dist))
            
        # Default constraint bounds.
        self.u_lower = cp.Parameter(T)
        self.u_lower.value = np.tile(0, T)
        self.u_upper = cp.Parameter(T)
        self.u_upper.value = np.tile(self.Pm, T)
        self.x_lower = cp.Parameter(T)
        self.x_lower.value = np.tile(self.T_sp - self.Delta, T)
        self.x_upper = cp.Parameter(T)
        self.x_upper.value = np.tile(self.T_sp + self.Delta, T)
        
        if RC_flag:
            a = np.exp(-dt / (self.R * self.C))
            b = self.eta * self.R

            lam = np.logspace(1, T, num=T, base=a)
            Lam = np.zeros((T, T))
            for i in range(T):
                for j in range(i+1):
                    Lam[i, j] = a**(i-j)
            B = np.eye(T) * b * (1-a) * self.Pm
            self.d.value = (1-a) * np.tile(32, T)
        else:
            A = np.eye(T)
            for i in range(T-1):
                A[i+1, max(0, i+1-self.p):i+1] = -np.flip(self.ap)[-(i+1):]
            Lam = np.linalg.inv(A)
    
            lam = np.zeros((T, self.p))
            for i in range(self.p):
                lam[i, i:] = np.flip(self.ap)[:self.p-i]
        
            B = np.zeros((T, T))
            for i in range(self.m):
                B += np.diag(np.ones(T-i), -i) * self.bu[i] / self.Pm
            
            self.d.value = np.zeros((T, self.n_dist))
            
        self.constraints = [
            -self.u <= -self.u_lower,
            self.u <= self.u_upper
        ]
        if RC_flag:
            self.constraints += [
                -Lam @ (self.sign * (1-a)*b*self.u + self.d) <= -self.x_lower + lam*self.x0,
                Lam @ (self.sign * (1-a)*b*self.u + self.d) <= self.x_upper - lam*self.x0,
            ]
        else:
            self.constraints += [
                -Lam @ (B @ self.u + self.d @ self.bd + lam @ self.x0) <= -self.x_lower,
                Lam @ (B @ self.u + self.d @ self.bd + lam @ self.x0) <= self.x_upper,
            ]

        self.Problem = cp.Problem(cp.Minimize(self.objective), self.constraints)
        self.scaling_factor = 1.0

    def scale_problem(self):
        # Force conversion of parameters.
        x0_val = to_numpy(self.x0.value)
        d_val = to_numpy(self.d.value)
        x_lower_val = to_numpy(self.x_lower.value)
        x_upper_val = to_numpy(self.x_upper.value)
        
        state_magnitude = max(np.abs(x0_val).max() if np.ndim(x0_val)>0 else np.abs(x0_val), 1.0)
        dist_magnitude = max(np.abs(d_val).max() if np.ndim(d_val)>0 else np.abs(d_val), 1.0)
        bounds_magnitude = max(np.abs(x_lower_val).max(), np.abs(x_upper_val).max(), 1.0)
        self.scaling_factor = 1.0 / max(state_magnitude, dist_magnitude, bounds_magnitude)
        
        self.x0.value = to_numpy(self.x0.value) * self.scaling_factor
        self.d.value = to_numpy(self.d.value) * self.scaling_factor
        self.x_lower.value = to_numpy(self.x_lower.value) * self.scaling_factor
        self.x_upper.value = to_numpy(self.x_upper.value) * self.scaling_factor
        print("Applied scaling factor:", self.scaling_factor)
        # Use np.any to check element-wise
        if np.any(self.x0.value < self.x_lower.value) or np.any(self.x0.value > self.x_upper.value):
            print("Warning: scaled x0 is outside the bounds. Consider adjusting x0 or the bounds.")

    def u_update(self, v_bar, w_bar):
        self.v_bar.value = to_numpy(v_bar)
        self.w_bar.value = to_numpy(w_bar)
        print("DEBUG: Before solving, x0.value =", self.x0.value, type(self.x0.value))
        try:
            self.Problem.solve()
        except Exception as e:
            print("Solver failed:", e)
            self.u.value = None

        if self.u.value is not None:
            return self.u.value, self.Problem.status
        else:
            u = (self.x0.value - self.T_sp) / self.Delta
            self.err_count += 1
            return np.ones(self.T) * np.clip(u, 0, 1) * self.Pm, self.Problem.status

    def updateState(self, x, u_lower=None, u_upper=None, x_lower=None, x_upper=None, d=None):
        # Convert x to a NumPy array.
        x = to_numpy(x)
        original_x = x.copy()
        if x_lower is not None:
            self.x_lower.value = to_numpy(x_lower)
        if x_upper is not None:
            self.x_upper.value = to_numpy(x_upper)
            self.T_sp = (self.x_upper.value[0] + self.x_lower.value[0]) / 2
            self.Delta = (self.x_upper.value[0] - self.x_lower.value[0]) / 2
        # Clip x to the defined bounds.
        x = np.clip(x, np.min(self.x_lower.value), np.max(self.x_upper.value))
        if not np.array_equal(original_x, x):
            print("Warning: Initial state was outside bounds. Clipped from", original_x, "to", x)
        # Ensure x has at least one dimension.
        self.x0.value = np.atleast_1d(to_numpy(x))
        print("DEBUG: After updateState, x0.value =", self.x0.value, type(self.x0.value))
        if u_lower is not None:
            self.u_lower.value = np.tile(u_lower, self.T) if np.isscalar(u_lower) else to_numpy(u_lower)
        if u_upper is not None:
            self.u_upper.value = np.tile(u_upper, self.T) if np.isscalar(u_upper) else to_numpy(u_upper)
        if d is not None:
            self.d.value = to_numpy(d)
        self.scale_problem()

class ControllerGroup():
    def __init__(self, T, dt, parameters, RC_flag=True):
        self.n_agent = len(parameters)
        self.T = T
        self.dt = dt
        self.RC_flag = RC_flag
        self.controller_list = self._init_agents(parameters)

    def _init_agents(self, parameters):
        controllers = []
        for param in parameters:
            controllers.append(Controller(T=self.T, dt=self.dt, RC_flag=self.RC_flag, **param))
        return controllers

    def updateState(self, x_list, u_lower_list=None, u_upper_list=None,
                    x_lower_list=None, x_upper_list=None, d_list=None):
        for i, controller in enumerate(self.controller_list):
            controller.updateState(
                x_list[i],
                u_lower = u_lower_list[i] if u_lower_list is not None else None,
                u_upper = u_upper_list[i] if u_upper_list is not None else None,
                x_lower = x_lower_list[i] if x_lower_list is not None else None,
                x_upper = x_upper_list[i] if x_upper_list is not None else None,
                d = d_list[i] if d_list is not None else None,
            )
            controller.u_diff.value = np.zeros(controller.T)

    def u_update(self, v_bar, w_bar):
        u_list = []
        statuses = []
        for i, controller in enumerate(self.controller_list):
            u_i, status = controller.u_update(v_bar, w_bar)
            statuses.append(status)
            if status in ["infeasible", "unbounded"]:
                print(f"Controller {i} status: {status}")
            u_list.append(u_i)
        u_bar = np.mean(u_list, axis=0)
        for i, controller in enumerate(self.controller_list):
            controller.u_diff.value = u_list[i] - u_bar
        print("QP statuses:", statuses)
        return u_bar, np.array(u_list)
