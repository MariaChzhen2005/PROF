import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
    
# Implement a vanilla MLP here
class MLP(nn.Module):
    def __init__(self, input_size, hiddens, output_size):
        super(MLP, self).__init__()
        self.n_layers = len(hiddens)
        self.layers = []
        tmp = [input_size] + hiddens
        
        for i in range(self.n_layers):
            self.layers.append(nn.Linear(tmp[i], tmp[i+1]))
            self.layers.append(nn.ReLU())
            # self.layers.append(nn.BatchNorm1d(tmp[i+1]))
        self.layers.append(nn.Linear(tmp[-1], output_size))
        self.layers = nn.ModuleList(self.layers)
    
    def forward(self,x):
        out = x
        for i, l in enumerate(self.layers):
            out = l(out)
        return out


class LSTM(nn.Module):
    def __init__(self, n_state, n_action, n_dist, lstm_hidden = 8, hiddens = [4], lstm_layer = 2, bi = False):
        super(LSTM, self).__init__()
        
        self.rnn = nn.LSTM(n_dist, lstm_hidden, lstm_layer, dropout = 0, bidirectional = bi)
        if bi:
            self.n_direction = 2
        else:
            self.n_direction = 1
            
        self.lstm_hidden = lstm_hidden
        self.lstm_layer = lstm_layer
        
        self.encoder1 = nn.Sequential(
            nn.Linear(n_state, 4),
            nn.ReLU(),
            #nn.BatchNorm1d(32),
            nn.Linear(4, lstm_hidden*self.n_direction*self.lstm_layer),
            nn.ReLU())
        
        self.encoder2 = nn.Sequential(
            nn.Linear(n_state, 4),
            nn.ReLU(),
            #nn.BatchNorm1d(32),
            nn.Linear(4, lstm_hidden * self.n_direction*self.lstm_layer),
            nn.ReLU())
            
        n_layers = len(hiddens) + 1
        tmp = [self.n_direction * lstm_hidden] + hiddens #+ [n_action]
        
        self.decoder = []
        for i in range(n_layers-1):
            self.decoder.append(nn.Linear(tmp[i], tmp[i+1]))
            self.decoder.append(nn.ReLU())
        self.decoder = nn.ModuleList(self.decoder)
        
        # mu and sigma2 are learned separately
        self.final_layer = nn.Linear(tmp[-1], n_action)
        self.final_layer_ = nn.Linear(tmp[-1], n_action)
    
    def forward(self, state, disturbance):
        # state: n x dim
        # disturbance: T x n x dist
        n = state.shape[0]
        T = disturbance.shape[0]
        
        h0 = self.encoder1(state).reshape(n, self.n_direction*self.lstm_layer, self.lstm_hidden).transpose(0, 1) # (layer x direction) x n x Dim.
        c0 = self.encoder2(state).reshape(n, self.n_direction*self.lstm_layer, self.lstm_hidden).transpose(0, 1)

        out, (hn, cn) = self.rnn(disturbance, (h0, c0)) # out:  T x n x (lstm_hidden x n_direction)
        #print("line 176")
        out = out.reshape(T * n, self.lstm_hidden * self.n_direction)
        for layer in self.decoder:
            out = layer(out)
        mu = self.final_layer(out).reshape(T, n, -1)
        sigma_sq = self.final_layer_(out).reshape(T, n, -1)
        # out: (T x n) x n_action
        return mu, sigma_sq

'''
class Replay_Memory():
    def __init__(self, memory_size=288, burn_in=32):
        self.memory_size = memory_size
        self.burn_in = burn_in
        # the memory is as a list of transitions (S,A,R,S,D).
        self.storage = []

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        rand_idx = np.random.choice(len(self.storage), batch_size)
        return [self.storage[i] for i in rand_idx]

    def append(self, transition):
        # appends transition to the memory.
        self.storage.append(transition)
        # only keeps the latest memory_size transitions
        if len(self.storage) > self.memory_size:
            self.storage = self.storage[-self.memory_size:]
'''
