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

class RBCD_PW:

    def __init__(self, eta, tau, maxiter, tol, silence=False):
        """
        eta: Entropic regularization parameter
        tau: step size for Projected Gradient Ascent Update
        maxiter: Maximum number of iterations
        tol : stopping threshold
        silence: 'True' for printing additional messages, 'False' otherwise
        """

        self.eta = eta
        self.tau = tau
        self.maxiter = maxiter
        self.tol = tol
        self.silence = silence
    
    def initial_Stiefel(self,D,d):
        U = np.random.randn(D, d)
        q, r = np.linalg.qr(U)
        return q
    
    def Stiefel_retraction(self,U,G):
        q, r = np.linalg.qr(U+G)
        return q
    
    def Stiefel_tan_proj(self, G, U):
        # projet G onto the tangent space of the sphere at U
        temp = G.T @ U
        PG = G - U @ (temp + temp.T)/2
        return PG
    
    def Vpi(self,X,Y,a,b,PI):
        # return the second order matrix of the displacements:
        # sum_{i,j} (PI)_{i,j} (X_i-Y_j)*(X_i-Y_j).T
        A = X.T @ PI @ Y
        return X.T.dot(np.diag(a)).dot(X) + Y.T.dot(np.diag(b)).dot(Y) - A - A.T
    
    def run_RBCD(self, X, Y, d):
        # Riemannian Block Coordinate Descent

        # initialization
        n_x, D = np.shape(X)
        n_y, _ = np.shape(Y)
        a = np.float64(np.ones(n_x)/n_x)
        b = np.float64(np.ones(n_y)/n_y)
        u = np.ones(n_x)/n_x
        v = np.ones(n_y)/n_y

        eta = self.eta
        tau = self.tau
        ones = np.ones((n_x,n_y))
        iter = 0
        U = self.initial_Stiefel(D,d)
        UUT = U @ U.T
        M = np.diag(np.diag(X.dot(UUT.dot(X.T)))).dot(ones) + ones.dot(np.diag(np.diag(Y.dot(UUT.dot(Y.T))))) - 2*X.dot(UUT.dot(Y.T))

        # Next 3 lines equivalent to A= np.exp(-M/reg), but faster to compute
        A = np.empty(M.shape, dtype=M.dtype)
        np.divide(M, -eta, out=A)
        np.exp(A, out=A)

        Ap = (1 / a).reshape(-1, 1) * A
        AtransposeU = np.dot(A.T, u)
        v = np.divide(b, AtransposeU)
        u = 1. / np.dot(Ap, v)

        PI = u.reshape((-1, 1)) * A * v.reshape((1, -1))
        V = self.Vpi(X, Y, a, b, PI)

        G = 2/eta  * V.dot(U)
        xi = self.Stiefel_tan_proj(G, U)
        grad_norm = np.linalg.norm(xi)

        if D == d:
            grad_norm = 1000
        
        grad_iter = np.zeros(self.maxiter + 1)
        grad_iter[0] = np.linalg.norm(xi)

        while eta*grad_norm > self.tol and iter < self.maxiter:

            UUT = U @ U.T
            M = np.diag(np.diag(X.dot(UUT.dot(X.T)))).dot(ones) + ones.dot(np.diag(np.diag(Y.dot(UUT.dot(Y.T))))) - 2*X.dot(UUT.dot(Y.T))

            #One step Sinkhorn
            A = np.empty(M.shape, dtype=M.dtype)
            np.divide(M, -eta, out=A)
            np.exp(A, out=A)

            Ap = (1 / a).reshape(-1, 1) * A
            AtransposeU = np.dot(A.T, u)
            v = np.divide(b, AtransposeU)
            u = 1. / np.dot(Ap, v)
            PI = u.reshape((-1, 1)) * A * v.reshape((1, -1))
            V = self.Vpi(X, Y, a, b, PI)

            G = 2/eta  * V.dot(U)
            xi = self.Stiefel_tan_proj(G, U)
            U = self.Stiefel_retraction(U, tau * xi)
            grad_norm = np.linalg.norm(xi)

            grad_iter[iter + 1] = np.linalg.norm(xi)
            iter = iter + 1
        
        f_val = np.trace(U.T.dot(V.dot(U)))
        if not(self.silence):
            print('Iter: ', iter, ' grad', eta*grad_norm , '\t fval: ', f_val)
        
        return PI, U, f_val, iter

def PW_test(X_Te, Y_Te, U, num_perm = 100, reg=1):
    nX_Te, D = np.shape(X_Te)
    nY_Te, _ = np.shape(Y_Te)
    data_Te = np.concatenate((X_Te, Y_Te), axis=0)
    n_Te = nX_Te + nY_Te

    eta_hist = np.float32(np.zeros(num_perm))

    a = np.float64(np.ones(nX_Te)/nX_Te)
    b = np.float64(np.ones(nY_Te)/nY_Te)

    X_Te_proj = X_Te @ U
    Y_Te_proj = Y_Te @ U
    M = Pdist2(X_Te_proj, Y_Te_proj)
    PI = sinkhorn.sinkhorn_knopp(a,b,M,reg=reg)
    #PI = ot.sinkhorn(a,b,M,reg)
    eta = np.sum(PI*M)
    for iboot in range(num_perm):
        tmp = np.random.permutation(n_Te)
        idx1_perm, idx2_perm = tmp[0:nX_Te], tmp[nX_Te:n_Te]

        X_Te_proj_perm = data_Te[idx1_perm] @ U
        Y_Te_proj_perm = data_Te[idx2_perm] @ U

        M = Pdist2(X_Te_proj_perm, Y_Te_proj_perm)
        PI = sinkhorn.sinkhorn_knopp(a,b,M,reg=reg)
        #PI = ot.sinkhorn(a,b,M,reg)

        eta_hist[iboot] = np.sum(PI*M)
    
    t_alpha = np.quantile(eta_hist, 0.95)
    
    if eta > t_alpha:
        decision = 1
    else:
        decision = 0
    return decision, eta, t_alpha