import pandas as pd
import numpy as np

class DifferenceMatrix:
    
    def __init__(self, func, data, labels=None, scale=None):
        if (labels is not None) and (isinstance(labels, dict) is not isinstance(data, dict)):
            raise TypeError("Cannot match labels to data.")

        if (labels is not None) and (not isinstance(labels, dict) and not isinstance(data, dict)):
            labels = {i: y for i, y in zip(range(len(labels)), labels)}

        if not isinstance(data, dict):
            data = {i: x for i, x in zip(range(len(data)), data)}

        self._func = func
        self._table = self._fill_table(data, labels, scale)

    def __len__(self):
        return len(self._table)

    def __iter__(self):
        for i in range(len(self)):
            for j in range(len(self)):
                df = self._table.iloc[[i], [j]]
                yield df.index[0], df.columns[0], df.squeeze()

    def __repr__(self):
        return str(self._table)

    def _fill_table(self, data: dict, y: dict, scale):
        sorted_keys = sorted(data.keys())
        s = len(sorted_keys)
        M = np.zeros((s, s))
        
        for i in range(s):
            for j in range(i + 1):
                if y is None:
                    M[i][j] = self._func(data[sorted_keys[i]], data[sorted_keys[j]])
                else:
                    M[i][j] = self._func(y[sorted_keys[i]], y[sorted_keys[j]])

        M = M + M.T - np.diag(np.diag(M))

        if np.any(M.diagonal() != 0):
            raise ArithmeticError(f"{type(self).__name__} diagonal contains nonzero values.")

        if scale is not None:
            M = scale(M, axis=1)

        return pd.DataFrame(M, columns=sorted_keys, index=sorted_keys)    

    def get_index(self, i, j):
        return self._table.iloc[i, j]

    def get_key(self, key):
        return self._table.loc[key]

    def get_func(self):
        return self._func
