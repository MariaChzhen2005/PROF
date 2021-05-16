# Helper Functions
import numpy as np
import torch
import torch.utils.data as data
import pdb

class Replay_Memory():
    def __init__(self, memory_size=86400):
        self.memory_size = memory_size
        self.storage = []

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        rand_idx = np.random.choice(len(self.storage), batch_size)
        batch = [self.storage[i] for i in rand_idx]
        
        state = [transition[0] for transition in batch]
        Sbus = [transition[1] for transition in batch]
        P_av = [transition[2] for transition in batch]
        return torch.tensor(np.stack(state)).float(), np.stack(Sbus), np.stack(P_av)
         
    def append(self, transition):
        # appends transition to the memory.
        self.storage.append(transition)
        # only keeps the latest memory_size transitions
        if len(self.storage) > self.memory_size:
            self.storage = self.storage[-self.memory_size:]

    
