import numpy as np
import time
from scipy.stats import entropy
import Optimization.sinkhorn as sinkhorn
#import ot
np.random.seed(0)

def Pdist2(x, y):
    # compute the paired distance between x and y
    x_norm = (x ** 2).sum(1).reshape([-1,1])
    y_norm = (y ** 2).sum(1).reshape([1,-1])
    Pdist = x_norm + y_norm - 2.0 * x @ y.T
    Pdist[Pdist<0]=0
    return Pdist

def S_test(X_Te, Y_Te, num_perm = 100, reg=1):
    nX_Te, D = np.shape(X_Te)
    nY_Te, _ = np.shape(Y_Te)
    data_Te = np.concatenate((X_Te, Y_Te), axis=0)
    n_Te = nX_Te + nY_Te

    eta_hist = np.float32(np.zeros(num_perm))
    a = np.float64(np.ones(nX_Te)/nX_Te)
    b = np.float64(np.ones(nY_Te)/nY_Te)

    M0_0 = Pdist2(X_Te, Y_Te)
    PI0_0 = sinkhorn.sinkhorn_knopp(a,b,M0_0,reg=reg,numItermax=100, stopThr=1e-6)
    eta0_0 = np.sum(PI0_0*M0_0)

    M0_1 = Pdist2(X_Te, X_Te)
    PI0_1 = sinkhorn.sinkhorn_knopp(a,b,M0_1,reg=reg,numItermax=100, stopThr=1e-6)
    eta0_1 = np.sum(PI0_1*M0_1)

    M0_2 = Pdist2(Y_Te, Y_Te)
    PI0_2 = sinkhorn.sinkhorn_knopp(a,b,M0_2,reg=reg,numItermax=100, stopThr=1e-6)
    eta0_2 = np.sum(PI0_2*M0_2)

    eta = eta0_0 - 0.5*(eta0_1 + eta0_2)


    for iboot in range(num_perm):
        #tem_X = np.random.permutation(nX_Te)
        #tem_Y = np.random.permutation(nY_Te)
        tmp = np.random.permutation(n_Te)
        idx1_perm, idx2_perm = tmp[0:nX_Te], tmp[nX_Te:n_Te]

        X_Te_perm = data_Te[idx1_perm]
        Y_Te_perm = data_Te[idx2_perm]

        M_0 = Pdist2(X_Te_perm, Y_Te_perm)
        PI_0 = sinkhorn.sinkhorn_knopp(a,b,M_0,reg=reg,numItermax=100, stopThr=1e-6)
        eta_0 = np.sum(PI_0*M_0)

        M_1 = Pdist2(X_Te_perm, X_Te_perm)
        PI_1 = sinkhorn.sinkhorn_knopp(a,b,M_1,reg=reg,numItermax=100, stopThr=1e-6)
        eta_1 = np.sum(PI_1 * M_1)

        M_2 = Pdist2(Y_Te_perm, Y_Te_perm)
        PI_2 = sinkhorn.sinkhorn_knopp(a,b,M_2,reg=reg,numItermax=100, stopThr=1e-6)
        eta_2 = np.sum(PI_2 * M_2)


        eta_hist[iboot] = eta_0 - 0.5*(eta_1+eta_2)



    t_alpha = np.quantile(eta_hist, 0.95)
    if eta > t_alpha:
        decision = 1
    else:
        decision = 0
    return decision, eta, t_alpha



