import numpy as np

class DifferenceMatrix:
    
    def __init__(self, func, data, labels=None, scale=None):
        self._func = func
        self._matrix = self._fill_matrix(data, labels, scale)

    def __len__(self):
        return len(self._matrix)

    def __iter__(self):
        for i in range(len(self)):
            for j in range(len(self)):
                yield i, j, self._matrix[i, j]

    def __repr__(self):
        return str(self._matrix)

    def _fill_matrix(self, data: dict, y: dict, scale):
        s = len(data)
        M = np.zeros((s, s))
        
        for i in range(s):
            for j in range(i + 1):
                if y is None:
                    M[i][j] = self._func(data[i], data[j])
                else:
                    M[i][j] = self._func(y[i], y[j])

        M = M + M.T - np.diag(np.diag(M))

        if np.any(M.diagonal() != 0):
            raise ArithmeticError(f"{type(self).__name__} diagonal contains nonzero values.")

        if scale is not None:
            M = scale(M, axis=1)

        return M

    def get_index(self, i, j):
        return self._matrix[i, j]

    def get_entry(self, index):
        return self._matrix[index]

    def get_func(self):
        return self._func

    def get_matrix(self):
        return self._matrix.copy()
