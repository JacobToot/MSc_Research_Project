import numpy as np
import random
import math

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import sys
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(repo_root, "src"))

results_path = os.path.join(repo_root, "results", "output", "1D_Step_Classification")
os.makedirs(results_path, exist_ok=True)

weights_path = os.path.join(repo_root, "results", "weights", "1D_Step_Classification")
os.makedirs(weights_path, exist_ok=True)

from utils.dataset_1d_ext import DataSet1D
from models.classifier_1d import Conv1DClassifier
from simulations.quasi_crystal_1d import quasi_crystal_1d
from simulations.random_step_1d import random_step_1d
from simulations.gaussian_step_1d import gaussian_step_1d
from utils.permutation import permutation_1d
from utils.generate_irrational import generate_irrational

crystal = quasi_crystal_1d(lattice_spacing = 1, acceptance_window_lower = 2, acceptance_window_upper = 20,  upper_limit_slope = 10, number_of_points = 2000, number_of_sequences = 3000)