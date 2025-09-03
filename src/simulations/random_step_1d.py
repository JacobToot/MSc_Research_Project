def random_step_1d(step_size, positive_probability, number_of_points = 100):
    """Generates a 1D step pattern where each step is of fixed length step_size but the direction is random.
    
    Imports: 
        numpy as np.

    Args:
        step_size (float): fixed length of each step.
        positive_probability (float): probability of taking a positive step (between 0 and 1).
        number_of_points (int, optional): number of points to generate. Defaults to 100.
    
    Returns:
        sequence (np.ndarray): array of distances of points from origin."""
    
    sequence = np.zeros(number_of_points)

    for i in range(number_of_points - 1):
        if np.random.rand() < positive_probability:
            sequence[i + 1] = sequence[i] + step_size
        else:
            sequence[i + 1] = sequence[i] - step_size
    
    return sequence