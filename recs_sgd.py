from tqdm import *
import numpy as np

TOTAL = 90000
k = 20
L = 0.1
niters = 40
lr = .01
m, n = (1683, 944)

def l2_norm(x): return np.linalg.norm(x, ord=2)

def calc_err(path, Q, P):
    seen_u = set()
    seen_m = set()
    uperror = []
    for row in open(path):
        u,i, rating = [int(x) for x in row.split()]
        seen_u.add(u)
        seen_m.add(i)
        uperror.append((rating - Q[i].dot(P[u].T))**2)
    E = np.sum(uperror) + L * l2_norm(P[list(seen_u)]) +  L * l2_norm(Q[list(seen_m)])
    return E


Q = np.random.uniform(high=np.sqrt(5/k), size=(m, k))
P = np.random.uniform(high=np.sqrt(5/k), size=(n, k))
path = '/Users/shleifer/Dropbox/stanford/cs246/hw2-bundle/q3/data/ratings.train.txt'

def train(Q, P):
    errs = []
    epoch = 0
    while True:
        epoch +=1
        q_ckpt = Q.copy()
        p_ckpt = P.copy()
        for row in open(path):
            u,i, rating  = [int(x) for x in row.split()]
            uperror = (rating - Q[i].dot(P[u].T))
            dqi = -2 * uperror * P[u] + 2 * L * Q[i]
            dpu = -2 * uperror * Q[i] + 2 * L * P[u]
            assert dpu.shape == P[u].shape
            P[u] -= dpu * lr
            Q[i] -= dqi * lr
            assert not np.isnan(Q).any()
        err = calc_err(path, Q, P)
        errs.append(err)
        print(f'{epoch}: {err:.2f}')
        if np.allclose(Q, q_ckpt) and np.allclose(P, p_ckpt):
            print(f'Converged after {i} iterations')
            break

    return Q, P, errs
