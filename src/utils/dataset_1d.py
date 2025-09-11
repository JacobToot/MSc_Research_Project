import numpy as np
import torch
from torch.utils.data import Dataset

import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(repo_root, "src"))

from utils.generate_irrational import generate_irrational

class DistanceDataset(Dataset):
    def __init__(self, n_samples=3000, seq_len=200):
        self.n_samples = n_samples
        self.seq_len = seq_len

        # balanced labels
        labels = [0] * 0.9 * n_samples + [1] * 0.05 * n_samples + [2] * 0.05 * n_samples
        self.labels = np.array(labels)
        np.random.shuffle(self.labels)  

        from simulations.quasi_crystal_1d import quasi_crystal_1d
        from simulations.random_step_1d import random_step_1d
        from simulations.gaussian_step_1d import gaussian_step_1d
        self.generators = [quasi_crystal_1d, random_step_1d, gaussian_step_1d]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        label = self.labels[idx]
        generator = self.generators[label]

        # Quasi-Crystal 
        if label == 0:  
            acceptance_window = np.random.randint(10, 50)
            seq = generator(lattice_spacing = 1, slope = generate_irrational(upper_limit = 2), acceptance_window = acceptance_window, number_of_points = self.seq_len)[0]

        # Random Step
        elif label == 1:  
            step_size = np.random.uniform(0.1, 1.5)
            positive_probability = np.random.uniform(0.1, 0.9)
            seq = generator(step_size = step_size, positive_probability = positive_probability, number_of_points = self.seq_len)
        
        # Gaussian Step
        else:
            step_size = np.random.uniform(0.5, 1.5)
            std_dev = np.random.uniform(0.1, 0.3)
            seq = generator(step_size = step_size, std_dev = std_dev, number_of_points = self.seq_len)

        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)

        return x, y

