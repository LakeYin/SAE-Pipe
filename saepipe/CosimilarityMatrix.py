import pandas as pd
import numpy as np

class CosimilarityMatrix:
    
    def __init__(self, func, data: dict, scale=None):
        self._similarity = func
        self._table = self._fill_table(func, data, scale)

    def __len__(self):
        return len(self._table)

    def __iter__(self):
        for i in range(len(self)):
            for j in range(len(self)):
                df = self._table.iloc[[i], [j]]
                yield df.index[0], df.columns[0], df.squeeze()

    def __repr__(self):
        return str(self._table)

    def _fill_table(self, func, data: dict, scale):
        sorted_keys = sorted(data.keys())
        s = len(sorted_keys)
        M = np.zeros((s, s))

        for i in range(s):
            for j in range(i + 1):
                M[i][j] = func(data[sorted_keys[i]], data[sorted_keys[j]])

        M = M + M.T - np.diag(np.diag(M))

        if scale is not None:
            M = scale(M, axis=1)

        return pd.DataFrame(M, columns=sorted_keys, index=sorted_keys)    

    def get_index(self, i, j):
        return self._table.iloc[i, j]

    def get_label(self, label):
        return self._table.loc[label]

    def get_func(self):
        return self._similarity
