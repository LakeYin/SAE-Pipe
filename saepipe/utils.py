from saepipe import DifferenceMatrix

import numpy as np

def root_mean_embedded_error(diff_func, X, embedding, y=None, scale=None):
    M = DifferenceMatrix(diff_func, X, labels=y, scale=scale).get_matrix()
    N = DifferenceMatrix(lambda a, b: np.linalg.norm(a - b), embedding).get_matrix()

    rmse = np.sqrt(np.sum(np.square(M - N)))

    return rmse / len(X)
