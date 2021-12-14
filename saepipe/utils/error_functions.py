from saepipe import DifferenceMatrix

import numpy as np

def mean_square_embedded_error(diff_func, embedding, X, y=None, scale=None, dist_func=None):
    return base_embedded_error(np.square, diff_func, embedding, X, y=y, scale=scale, dist_func=dist_func)

def mean_abs_embedded_error(diff_func, embedding, X, y=None, scale=None, dist_func=None):
    return base_embedded_error(np.absolute, diff_func, embedding, X, y=y, scale=scale, dist_func=dist_func)

def base_embedded_error(error_f, diff_func, embedding, X, y=None, scale=None, dist_func=None):
    if y is None:
        M = DifferenceMatrix(diff_func, X, scale=scale).get_matrix()
    else:
        M = DifferenceMatrix(diff_func, y, scale=scale).get_matrix()

    if dist_func is None:
        dist_func = lambda a, b: np.linalg.norm(a - b)

    N = DifferenceMatrix(dist_func, embedding).get_matrix()

    return np.sum(error_f(M - N)) / (len(X) ** 2)
