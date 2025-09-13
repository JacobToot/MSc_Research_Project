import numpy as np
import torch

import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(repo_root, "src"))

from utils.generate_irrational import generate_irrational
from simulations.quasi_crystal_1d import quasi_crystal_1d
from simulations.random_step_1d import random_step_1d
from simulations.gaussian_step_1d import gaussian_step_1d
from utils.permutation import permutation_1d

class DataSet1D():
  def __init__(self):
    self.data = []
    self.labels =[]
    self.next_label = 0

  def _get_value(self, param):
        return param() if callable(param) else param
    
  def gaussian(self, number_of_sequences, number_of_points, step_size, std_dev):
    """Appends a number of gaussian sequences to the dataset using the helper function gaussian_step_1d"""

    for i in range(number_of_sequences):
      step_size_val = self._get_value(step_size)
      std_dev_val = self._get_value(std_dev)
      sequence = gaussian_step_1d(step_size_val, std_dev_val, number_of_points)
      avg = np.mean(sequence)
      self.data.append((1 / avg) * sequence)
      self.labels.append(self.next_label)
    
    self.next_label += 1

    return self

  def quasi_crystal(self, number_of_sequences, lattice_spacing, slope, acceptance_window, number_of_points):
    """Appends a number of quasi crystal sequences to the dataset using the helper function quasi_crystal_1d"""

    for i in range(number_of_sequences):
      slope_val = self._get_value(slope)
      acceptance_window_val = self._get_value(acceptance_window)
      sequence = quasi_crystal_1d(lattice_spacing, slope_val, acceptance_window_val, number_of_points)[0]
      avg = np.mean(sequence)
      self.data.append((1 / avg) * sequence)
      self.labels.append(self.next_label)
      print(f"Generated quasi crystal sequence {i+1}/{number_of_sequences}")
    
    self.next_label += 1

    return self

  def random_step(self, number_of_sequences, step_size, positive_probability, number_of_points):
    """Appends a number of random steps to dataset"""

    for i in range(number_of_sequences):
      step_size_val = self._get_value(step_size)
      positive_probability_val = self._get_value(positive_probability)
      sequence = random_step_1d(step_size_val, positive_probability_val, number_of_points)
      avg = np.mean(sequence)
      self.data.append((1 / avg) * sequence)
      self.labels.append(self.next_label)
    
    self.next_label += 1

    return self

  def permutation(self, number_of_pairs, permute_label=1):
    """creates a permuted version of the previously added dataset and appends it to the dataset
    Args:
        number_of_pairs (int): number of pairs to permute
        permute_label (int, optional): which label to permute. 1 = most recent, 2 = second most recent etc.. Defaults to 1."""

    last_label = self.next_label - permute_label

    indices = [i for i, lab in enumerate(self.labels) if lab == last_label]

    for i in indices:
      self.data.append(permutation_1d(self.data[i], number_of_pairs))
      self.labels.append(self.next_label)
    
    self.next_label += 1

    return self

  def __getitem__(self, index):
    return torch.tensor(self.data[index], dtype=torch.float32).unsqueeze(0), torch.tensor(self.labels[index], dtype=torch.long)

  def __len__(self):
    return len(self.data)

