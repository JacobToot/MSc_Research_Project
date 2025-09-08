import numpy as np
import matplotlib.pyplot as plt

def histogram(sequence, title, lattice_spacing = 1, bins = 20, log = False):
    """Generates a histogram of the quasi-crystal distance to the next point
    
    Inputs:
      sequence (1D np.array object): A sequence of points in 1D
      title(string): title of graph
      lattice_spacing (float, optional): Takes the spacing used in the sequence generation. Set to 1 by default to not normalise datapoints.
      bins(integer, optional): number of bins. defaults to 20.
      log(boolean, optional): use log scale in histogram. defaults to False
      """
    
    sequence = sequence / lattice_spacing
    
    plt.figure()
    plt.hist(sequence, bins = bins, log = log) 
    plt.title(title)
    plt.xlabel("Distance to next point")
    plt.ylabel("Number of points")