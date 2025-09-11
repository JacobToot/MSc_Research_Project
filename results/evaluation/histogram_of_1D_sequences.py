import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import random

# Adding src directory to path to import functions
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(repo_root, "src"))
results_path = os.path.join(repo_root, "results", "output", "Histogram_of_1D_sequences")
os.makedirs(results_path, exist_ok=True)

# Importing functions to be used in this script
from simulations.quasi_crystal_1d import quasi_crystal_1d
from simulations.random_step_1d import random_step_1d
from simulations.gaussian_step_1d import gaussian_step_1d
from utils.histogram_1d import histogram
from utils.generate_irrational import generate_irrational

# Parameters

quasi_crystal, slope, _ = quasi_crystal_1d(lattice_spacing = 1, slope = np.sqrt(13), acceptance_window = 40, number_of_points = 100000)
random_step = random_step_1d(step_size = 1, positive_probability = 0.5, number_of_points = 10000)
gaussian_step = gaussian_step_1d(step_size = 1, std_dev = 0.1, number_of_points = 10000)

# Generating histograms
histogram(quasi_crystal, title = f"1D Quasi Crystal Histogram, slope = {slope}", lattice_spacing = 1, bins = 200, log = True)
plt.savefig(os.path.join(results_path, "quasi_crystal_histogram3.png"))
plt.clf()

np.set_printoptions(threshold=np.inf)
print(quasi_crystal)

histogram(random_step, title = "1D Random Step Histogram", lattice_spacing = 1, bins = 20, log = False)
plt.savefig(os.path.join(results_path, "random_step_histogram2.png"))
plt.clf()

histogram(gaussian_step, title = "1D Gaussian Step Histogram", lattice_spacing = 1, bins = 20, log = False)
plt.savefig(os.path.join(results_path, "gaussian_step_histogram2.png"))
plt.clf()