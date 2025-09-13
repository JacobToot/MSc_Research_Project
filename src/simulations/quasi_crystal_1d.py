import numpy as np
from utils.generate_irrational import generate_irrational
def quasi_crystal_1d(lattice_spacing = 1, acceptance_window_lower = 2, acceptance_window_upper = 20,  upper_limit_slope = 10, number_of_points = 100, number_of_sequences = 1):

  """Function to compute a sequence of points that correspond to a 1D quasi crystal. Using cut-and-project method.

  Args:
      lattice_spacing (float, optional): spacing of the 2D lattice. Defaults to 1.
      acceptance_window_lower (int, optional): lower bound on acceptance window size. Defaults to 2.
      acceptance_window_upper (int, optional): upper bound on acceptance window size. Defaults to 20.
      upper_limit_slope (int, optional): upper limit on slope of line used in cut-and-project method. Defaults to 10.
      number_of_points (int, optional): number of points in the sequence. Defaults to 100.
      number_of_sequences (int, optional): number of sequences to generate. Defaults to 1.

  Returns:
      distances (np.ndarray): array of distances of point i to point i + 1. If number_of_sequences > 1, returns a 2D array where each row is a sequence.
      slope (float or np.ndarray): slope of line used in cut-and-project method. If number_of_sequences > 1, returns an array of slopes.
      acceptance_window (int or np.ndarray): size of acceptance window used in cut-and-project method. If number_of_sequences > 1, returns an array of acceptance window sizes.
      points (np.ndarray): array of projected points in 2D space used to generate the sequence. If number_of_sequences > 1, returns the points from the last generated sequence.
      
  """

  upper_limit = 2 * number_of_points * lattice_spacing
  coords = np.arange(np.floor(-acceptance_window_upper / 2), upper_limit + lattice_spacing, lattice_spacing)
  X, Y = np.meshgrid(coords, coords, indexing='ij')
  
  if number_of_sequences == 1:
    slope = generate_irrational(upper_limit=upper_limit_slope)
    acceptance_window = np.random.randint(acceptance_window_lower, acceptance_window_upper)
    
    # computing perpendicular distance from each point to the line y = slope * x
    s = np.abs(slope * X - Y) / np.sqrt(1 + slope ** 2)
    mask = s <= acceptance_window / 2
    X_masked = X[mask]
    Y_masked = Y[mask]
    
    # projecting points onto the line y = slope * x
    proj_x = (X_masked + slope * Y_masked) / (1 + slope**2)
    proj_y = slope * proj_x
    valid = proj_x >= 0
    proj_x = proj_x[valid]
    proj_y = proj_y[valid]
    
    # final touch up of data
    points = np.vstack((proj_x, proj_y)).T
    distances = np.sqrt(proj_x ** 2 + proj_y ** 2)
    distances = np.sort(distances)
    distances = np.diff(distances)
    distances = distances[:number_of_points]

  elif number_of_sequences > 1:
    distances = []
    points = []
    slopes = []
    acceptance_windows = []
    for i in range(number_of_sequences):
      slope = generate_irrational(upper_limit=upper_limit_slope)
      acceptance_window = np.random.randint(acceptance_window_lower, acceptance_window_upper)
      slopes.append(slope)
      acceptance_windows.append(acceptance_window)
      
      # computing perpendicular distance from each point to the line y = slope * x
      s = np.abs(slope * X - Y) / np.sqrt(1 + slope ** 2)
      mask = s <= acceptance_window / 2
      X_masked = X[mask]
      Y_masked = Y[mask]
      
      # projecting points onto the line y = slope * x
      proj_x = (X_masked + slope * Y_masked) / (1 + slope**2)
      proj_y = slope * proj_x
      valid = proj_x >= 0
      proj_x = proj_x[valid]
      proj_y = proj_y[valid]
      
      # final touch up of data
      distance_set = np.sqrt(proj_x ** 2 + proj_y ** 2)
      distance_set = np.sort(distance_set)
      distance_set = np.diff(distance_set)
      distance_set = distance_set[:number_of_points]
      distances.append(distance_set)
      print(f"Generated quasi crystal sequence {i+1}/{number_of_sequences}")
    
    distances = np.array(distances)
    slope = np.array(slopes)
    acceptance_window = np.array(acceptance_windows)
  
  return distances, slope, acceptance_window, points

