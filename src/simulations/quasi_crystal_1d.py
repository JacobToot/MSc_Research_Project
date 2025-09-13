import numpy as np

def quasi_crystal_1d(lattice_spacing = 1, slope = np.sqrt(2), acceptance_window = 10, number_of_points = 100):

  """Function to compute a sequence of points that correspond to a 1D quasi crystal. Using cut-and-project method.

  Args:

    lattice_spacing(float, optional): Spacing between points. Defaults to 1.
    slope(float, optional): Slope of line that 2D crystal points are projected onto. Must be an irrational number. Defaults to np.sqrt(2).
    acceptance_window(float, optional): Defines the maximum perpendicular direction from the line that points are projected onto. Defaults to 10.
    number_of_points(int): Number of points to generate. Defaults to 100.

  Returns:
    distances: a sorted np.ndarray object that stores a list of sorted distances of quasi crystal points. Distance is radial from point i to point i + 1.
    points: a 2D np.ndarray object that stores the coordinates of the quasi crystal points in the original coordinate system.
      
  """

  upper_limit = 2 * number_of_points / (lattice_spacing * acceptance_window)
  coords = np.arange(-acceptance_window / 2, upper_limit + lattice_spacing, lattice_spacing)
  X, Y = np.meshgrid(coords, coords, indexing='ij')

  # perpendicular projection and masking X, Y grid to find all lattice points that need to be projected
  s = np.abs(slope * X - Y) / np.sqrt(1 + slope ** 2)

  mask = s <= acceptance_window / 2
  X_masked = X[mask]
  Y_masked = Y[mask]

  # project all points onto y = slope * x
  proj_x = (X_masked + slope * Y_masked) / (1 + slope**2)
  proj_y = slope * proj_x

  # keep only positive x
  valid = proj_x >= 0
  proj_x = proj_x[valid]
  proj_y = proj_y[valid]
  
  # final touch up of data
  points = np.vstack((proj_x, proj_y)).T
  distances = np.sqrt(proj_x ** 2 + proj_y ** 2)
  distances = np.sort(distances)
  distances = np.diff(distances)
  distances = distances[:number_of_points]

  return distances, slope, points