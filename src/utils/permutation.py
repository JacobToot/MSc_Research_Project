import numpy as np

def permutation_1d(sequence, number_of_pairs=100):
    """Takes in an input sequence and permutes a number of specified pairs by swapping their indices.

    Args:
        sequence (array-like): The input sequence to be permuted.
        number_of_pairs (int, optional): The number of permuted sequences to generate. Defaults to 100.

    Returns:
        list: A list containing the permuted sequences.
    """

    n = len(sequence)
    vals = np.random.choice(n, size=(number_of_pairs, 2), replace=False)

    for k in range(number_of_pairs):
        i, j = vals[k]

        attempts = 0
        while sequence[i] == sequence[j] and attempts < n:
            attempts += 1
            j = (j + 1) % (n-1)
        
        sequence[i], sequence[j] = sequence[j], sequence[i]

    return sequence