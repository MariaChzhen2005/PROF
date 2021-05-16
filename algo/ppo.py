import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.distributions import MultivariateNormal, Normal

import pdb
from copy import deepcopy

from utils.ppo_utils import Dataset

class PPO():
    def __init__(self, policy, memory, clip_param = 0.2, lam = 10, lr = 5e-4, n_ctrl = 1):
        self.memory = memory

        self.policy = policy
        self.policy_old = deepcopy(policy)

        self.clip_param = clip_param
        self.optimizer = optim.RMSprop(self.policy.nn.parameters(), lr=lr)
        self.lam = lam
        
        self.n_ctrl = n_ctrl
        
    # Use the "current" flag to indicate which set of parameters to use
    def forward(self, state, disturbance, x_lowers = None, x_uppers = None, current = True):
        T, n_batch, n_dist = disturbance.shape
        if current == True:
            mu, sigma_sq, proj_loss = self.policy.forward(state, disturbance, x_lowers = x_lowers, x_uppers = x_uppers)
        else:
            mu, sigma_sq, proj_loss = self.policy_old.forward(state, disturbance)
        return mu, sigma_sq, proj_loss

    def select_action(self, mu, sigma_sq, u_limits = None):
        if self.n_ctrl > 1:
            m = MultivariateNormal(mu, torch.diag(sigma_sq.squeeze()).unsqueeze(0))
        else:
            m = Normal(mu, sigma_sq**0.5)
        action = m.sample()
        if u_limits is not None:
            action = torch.clamp(action, min = u_limits[0], max = u_limits[1])
        log_prob = m.log_prob(action)
        return action, log_prob

    def evaluate_action(self, mu, actions, sigma_sq):
        n_batch = len(mu)
        if self.n_ctrl > 1:
            cov = torch.diag_embed(sigma_sq)
            m = MultivariateNormal(mu, cov)
        else:
            m = Normal(mu, sigma_sq**0.5)
        log_prob = m.log_prob(actions)
        entropy = m.entropy()
        return log_prob, entropy
    
    def _get_training_samples(self):
        states, actions, disturbance, advantages, old_logprobs, x_uppers, x_lowers = self.memory.sample()
        batch_set = Dataset(states, actions, disturbance, advantages, old_logprobs, x_uppers, x_lowers)
        batch_loader = data.DataLoader(batch_set, batch_size=32, shuffle=True, num_workers=2)
        return batch_loader
    
    def update_parameters(self, sigma=0.1, K = 4):
        loader = self._get_training_samples()
        for i in range(K):
            for states, actions, disturbance, advantages, old_logprobs, x_uppers, x_lowers in loader:
                n_batch = states.shape[0]
                # pdb.set_trace()
                mus, sigma_sqs, proj_loss = self.policy.forward(states, disturbance.transpose(0, 1),  x_lowers = x_lowers, x_uppers = x_uppers) # x, u: T x N x Dim.
                sigma_sqs = torch.ones_like(mus) * sigma**2
                log_probs, entropies = self.evaluate_action(mus[0], actions, sigma_sqs)
        
                ratio = torch.exp(log_probs.squeeze()-old_logprobs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantages
                loss  = -torch.min(surr1, surr2).mean()
                self.optimizer.zero_grad()
                ## Auxiliary losses
                loss -= torch.mean(entropies) * 0.01
                loss += self.lam * proj_loss
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.nn.parameters(), 100)
                self.optimizer.step()
                print("Post Step")
        self.policy_old.nn.load_state_dict(self.policy.nn.state_dict())
    
    ##TODO: Move the update_policy to a Trainer class
    def behavior_cloning(self, batch_size):
        u_hat, u_star, u_nns = self._get_training_samples(batch_size)

        loss = self.criterion(u_hat, u_star)
        loss += self.lam * self.criterion(u_nns, u_hat) # Auxiliary loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.predictions = []
        self.targets = []
        return loss.detach()
            


