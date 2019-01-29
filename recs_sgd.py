from tqdm import *
import numpy as np

TOTAL = 90000
k = 20
L = 0.1
niters = 40
lr = .01
m, n = (1683, 944)


def calc_err(path, Q, P):
    seen_u = set()
    seen_m = set()
    uperror = []
    for row in tqdm_notebook(open(path), total=TOTAL):
        u,i, rating = [int(x) for x in row.split()]
        seen_u.add(u)
        seen_m.add(i)
        uperror.append((rating - Q[i]).dot(P[u].T))
    E = np.sum(uperror) + L * np.linalg.norm(P[list(seen_u)], ord=2) +  L * np.linalg.norm(Q[list(seen_m)], ord=2)
    return E
