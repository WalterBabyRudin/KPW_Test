import numpy as np
import torch
from Optimization.utils import MatConvert, MMDu, TST_MMD_u, Pdist2


def MMD_test(Data_XY, n_X):

    X = Data_XY[0:n_X, :]
    Y = Data_XY[n_X:, :]


    Dxx = Pdist2(X,X)
    Dyy = Pdist2(Y,Y)
    Dxy = Pdist2(X,Y)

    
    sigma = torch.median(Dxy)
    Kx = torch.exp(-Dxx / (2*sigma))
    Ky = torch.exp(-Dyy / (2*sigma))
    Kxy = torch.exp(-Dxy / (2*sigma))

    Kxxy = torch.cat((Kx,Kxy),1)
    Kyxy = torch.cat((Kxy.transpose(0,1),Ky),1)
    Kxyxy = torch.cat((Kxxy,Kyxy),0)


    _, p_val, _ = mmd2_permutations(Kxyxy, n_X, permutations=100)

    return p_val


def mmd2_permutations(K, n_X, permutations=500):
    """
        Fast implementation of permutations using kernel matrix.
    """
    K = torch.as_tensor(K)
    n = K.shape[0]
    assert K.shape[0] == K.shape[1]
    n_Y = n_X
    assert n == n_X + n_Y
    w_X = 1
    w_Y = -1
    ws = torch.full((permutations + 1, n), w_Y, dtype=K.dtype, device=K.device)
    ws[-1, :n_X] = w_X
    for i in range(permutations):
        ws[i, torch.randperm(n)[:n_X].numpy()] = w_X
    biased_ests = torch.einsum("pi,ij,pj->p", ws, K, ws)
    if True:  # u-stat estimator
        # need to subtract \sum_i k(X_i, X_i) + k(Y_i, Y_i) + 2 k(X_i, Y_i)
        # first two are just trace, but last is harder:
        is_X = ws > 0
        X_inds = is_X.nonzero()[:, 1].view(permutations + 1, n_X)
        Y_inds = (~is_X).nonzero()[:, 1].view(permutations + 1, n_Y)
        del is_X, ws
        cross_terms = K.take(Y_inds * n + X_inds).sum(1)
        del X_inds, Y_inds
        ests = (biased_ests - K.trace() + 2 * cross_terms) / (n_X * (n_X - 1))
    est = ests[-1]
    rest = ests[:-1]
    p_val = (rest > est).float().mean()
    return est.item(), p_val.item(), rest