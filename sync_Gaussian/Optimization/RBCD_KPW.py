import numpy as np
import time
from scipy.stats import entropy
import Optimization.sinkhorn as sinkhorn

np.random.seed(10)
def Pdist2(x, y):
    # compute the paired distance between x and y
    x_norm = (x ** 2).sum(1).reshape([-1,1])
    y_norm = (y ** 2).sum(1).reshape([1,-1])
    Pdist = x_norm + y_norm - 2.0 * x @ y.T
    Pdist[Pdist<0]=0
    return Pdist

# Riemannian Block Coordinate Descent for computing KPW

class RBCD_kernel:

    def __init__(self, eta, sigma, tau, maxiter, tol, silence=False, orho=0.5):
        """
        eta : Entropic regularization parameter
        sigma: bandwidth for kernel projector
        tau : step size for Projected Gradient Ascent Update
        maxiter: Maximum number of iterations
        tol : stopping threshold
        silence: 'True' for printing additional messages, 'False' otherwise
        """

        self.eta = eta
        self.sigma = sigma
        self.tau = tau
        self.maxiter = maxiter
        self.tol = tol
        self.silence = silence
        self.orho = orho
    
    def initial_sphere(self,dn_XY):
        s = np.random.randn(dn_XY)
        s = s/np.sqrt(s.T @ s)
        return s

    def sphere_retraction(self,s, z):
        q = s + z
        q = q / np.sqrt(q.T @ q)
        return q

    def sphere_tan_proj(self, z, s):
        # projet z onto the tangent space of the sphere at s
        Proj_z = z - s*(s@z)
        return Proj_z

    def Gram_formulate(self,X,Y,d):
        Dxx = Pdist2(X, X)
        Dyy = Pdist2(Y, Y)
        Dxy = Pdist2(X, Y)
        sigma = self.sigma
        orho = self.orho

        Kx = np.empty(Dxx.shape, dtype=Dxx.dtype)
        np.divide(Dxx, -sigma, out = Kx)
        np.exp(Kx, out = Kx)

        Ky = np.empty(Dyy.shape, dtype=Dyy.dtype)
        np.divide(Dyy, -sigma, out=Ky)
        np.exp(Ky, out = Ky)

        Kxy = np.empty(Dxy.shape, dtype=Dxy.dtype)
        np.divide(Dxy, -sigma, out=Kxy)
        np.exp(Kxy, out = Kxy)


        P = orho*np.ones([d,1])*np.ones([1,d]) + (1-orho)*np.eye(d)
        Kx = np.kron(Kx,P)
        Ky = np.kron(Ky,P)
        Kxy = np.kron(Kxy,P)
        Kxxy = np.concatenate((Kx, -Kxy), 1)
        Kyxy = np.concatenate((-Kxy.T, Ky), 1)
        G = np.concatenate((Kxxy, Kyxy), 0)
        return G
    
    def formulate_projection(self,s,G,U,n_x,n_y,d):
        Us = U @ s
        G_Us = G @ Us
        f_x = G_Us[:d*n_x].reshape([n_x,d])
        f_y = G_Us[d*n_x:].reshape([n_y,d])
        return f_x, f_y

    def V_Kpi(self,a,b,PI,G,d,n_x,n_y):
        # return the second-order matrix sum_ij {(OT_plan)_ij (X_i-Y_j)(X_i-Y_j)T}
        # where X_i = [K_{x_i}(x^n); -K_{x_i}(y^m)]
        # and   Y_j = [K_{y_j}(x^n); -K_{y_j}(y^m)]

        G_x = G[n_x*d:,:]
        G_y = G[:n_x*d,:]
        ones = np.ones([d,1])
        
        Grad_xx = G_x.T @ np.diag(np.kron(b, ones).reshape([-1,])) @ G_x
        Grad_yy = G_y.T @ np.diag(np.kron(a, ones).reshape([-1,])) @ G_y
        Grad_xy = G_x.T @ np.kron(PI, np.ones([d,d])) @ G_y
        # GG = np.kron(np.sum(PI,1), np.ones([d,1])).reshape([-1,])

        # Grad_xx = G_x.T@np.diag(np.kron(np.sum(PI,1), np.ones([d,1])).reshape([-1,]))@G_x
        # Grad_yy = G_y.T@np.diag(np.kron(np.sum(PI,0).T,np.ones([d,1])).reshape([-1,]))@G_y
        # Grad_xy = G_x.T@np.kron(PI, np.eye(d))@G_y
        Grad_hist = Grad_xx + Grad_yy + Grad_xy + Grad_xy.T
        return Grad_hist

    def run_RBCD(self, X, Y, d):
        # Riemannian Block Coordinate Descent
        silence = self.silence

        # initialization
        n_x, D = np.shape(X)
        n_y, _ = np.shape(Y)
        a = np.float64(np.ones(n_x)/n_x)
        b = np.float64(np.ones(n_y)/n_y)
        n_xy = n_x + n_y
        dn_xy = d * n_xy
        u = np.ones(n_x) / n_x
        v = np.ones(n_y) / n_y
        s = self.initial_sphere(dn_xy)

        G = self.Gram_formulate(X,Y,d)
        R = np.linalg.cholesky(G + 1e-4 * np.eye(dn_xy))
        #R = np.linalg.cholesky(G)
        U = np.linalg.inv(R)

        eta = self.eta
        tau = self.tau

        f_x,f_y = self.formulate_projection(s, G, U, n_x,n_y,d)
        C = Pdist2(f_x,f_y)

        A = np.empty(C.shape, dtype=C.dtype)
        np.divide(A, -eta, out = A)
        np.exp(A, out = A)

        Ap = (1 / a).reshape(-1, 1) * A
        ATu = A.T @ u 
        v = np.divide(b, ATu)
        u = 1. /(np.dot(Ap, v))
        # u = np.divide(a, A@v)
        # v = np.divide(b, A.T@u)


        PI = u.reshape((-1,1)) * A * v.reshape((1,-1))
        V = self.V_Kpi(a,b,PI,G,d,n_x,n_y)

        grad_Euclidean = 1/eta * U.T@(V@(U@s))
        xi = self.sphere_tan_proj(grad_Euclidean, s)
        grad_norm = np.linalg.norm(xi)

        grad_iter = np.zeros(self.maxiter + 1)
        grad_iter[0] = np.linalg.norm(xi)
        #print(grad_iter[0])

        iter = 0
        while (grad_norm > self.tol) and (iter < self.maxiter) and (D >= 10):

            f_x,f_y = self.formulate_projection(s, G, U, n_x,n_y,d)
            #print([np.mean(f_x), np.mean(f_y)])
            C = Pdist2(f_x,f_y)

            #f_val = np.sum(C * PI) - eta * entropy(PI.reshape([-1,]))
            PI = sinkhorn.sinkhorn_knopp(a,b,C,reg=10)
            f_val = np.sum(C * PI)
            #f_val = s.T @ U.T @ V @ U @ s
            #print([f_val, np.sum(PI)])
            if not(silence):
                print('Iter: ',iter, '\t Norm: ',grad_norm, '\t fval: ',f_val)

            A = np.empty(C.shape, dtype=C.dtype)
            np.divide(A, -eta, out = A)
            np.exp(A, out = A)

            Ap = (1 / a).reshape(-1, 1) * A
            ATu = A.T @ u
            v = np.divide(b, ATu)
            u = 1. /(np.dot(Ap, v))
            # u = np.divide(a, A@v)
            # v = np.divide(b, A.T@u)
            PI = u.reshape((-1,1)) * A * v.reshape((1,-1))

            V = self.V_Kpi(a,b,PI,G,d,n_x,n_y)

            grad_Euclidean = 1/eta * U.T@(V@(U@s))
            xi = self.sphere_tan_proj(grad_Euclidean, s)
            s = self.sphere_retraction(s, tau*xi)
            grad_norm = np.linalg.norm(xi)

            grad_iter[iter + 1] = np.linalg.norm(xi)
            iter = iter + 1

        if not(silence):
            print('Iter: ',iter, '\t Norm: ',grad_norm)

        Us = U @ s
        return Us
    
    def run_Alternating(self, X,Y,d):
        # Alternating Optimization
        silence = self.silence

        # initialization
        n_x, D = np.shape(X)
        n_y, _ = np.shape(Y)
        a = np.float64(np.ones(n_x)/n_x)
        b = np.float64(np.ones(n_y)/n_y)
        n_xy = n_x + n_y
        dn_xy = d * n_xy
        s = self.initial_sphere(dn_xy)

        G = self.Gram_formulate(X,Y,d)
        R = np.linalg.cholesky(G + 1e-4 * np.eye(dn_xy))
        U = np.linalg.inv(R)
        tau = self.tau
        f_x,f_y = self.formulate_projection(s, G, U, n_x,n_y,d)
        C = Pdist2(f_x,f_y)

        PI = sinkhorn.sinkhorn_knopp(a,b,C,reg=10)
        V = self.V_Kpi(a,b,PI,G,d,n_x,n_y)
        grad_Euclidean = U.T@(V@(U@s))
        xi = self.sphere_tan_proj(grad_Euclidean, s)
        grad_norm = np.linalg.norm(xi)
        grad_iter = np.zeros(self.maxiter + 1)
        grad_iter[0] = np.linalg.norm(xi)

        iter = 0

        while (grad_norm > self.tol) and (iter < self.maxiter) and (D >= 10):
            f_x,f_y = self.formulate_projection(s, G, U, n_x,n_y,d)
            C = Pdist2(f_x,f_y)
            PI = sinkhorn.sinkhorn_knopp(a,b,C,reg=10)
            V = self.V_Kpi(a,b,PI,G,d,n_x,n_y)
            grad_Euclidean = U.T@(V@(U@s))
            xi = self.sphere_tan_proj(grad_Euclidean, s)
            s = self.sphere_retraction(s, tau*xi)
            grad_norm = np.linalg.norm(xi)
            grad_iter[iter + 1] = np.linalg.norm(xi)
            iter = iter + 1
        if not(silence):
            print('Iter: ',iter, '\t Norm: ',grad_norm)
        Us = U @ s
        return Us


def KPW_Test(X_Te, Y_Te, X_Tr, Y_Tr, Us, sigma2, d, num_perm = 100, reg=10, orho=0.5):
    nX_Te, D = np.shape(X_Te)
    nY_Te, _ = np.shape(Y_Te)
    n_Te = nX_Te + nY_Te
    a = np.float64(np.ones(nX_Te)/nX_Te)
    b = np.float64(np.ones(nY_Te)/nY_Te)
    P = orho*np.ones([d,1])*np.ones([1,d]) + (1-orho)*np.eye(d)

    if D <= 5: 
        X_Te_proj = X_Te
        Y_Te_proj = Y_Te
    else:
        Dxx = Pdist2(X_Te, X_Tr)
        Kx = np.empty(Dxx.shape, dtype=Dxx.dtype)
        np.divide(Dxx, -sigma2, out = Kx)
        np.exp(Kx, out = Kx)
        Kx = np.kron(Kx, P)

        Dxy = Pdist2(X_Te, Y_Tr)
        Kxy = np.empty(Dxy.shape, dtype=Dxy.dtype)
        np.divide(Dxy, -sigma2, out = Kxy)
        np.exp(Kxy, out = Kxy)
        Kxy = np.kron(Kxy, P)

        Dyx = Pdist2(Y_Te, X_Tr)
        Kyx = np.empty(Dyx.shape, dtype=Dyx.dtype)
        np.divide(Dyx, -sigma2, out = Kyx)
        np.exp(Kyx, out = Kyx)
        Kyx = np.kron(Kyx, P)

        Dyy = Pdist2(Y_Te, Y_Tr)
        Ky = np.empty(Dyy.shape, dtype=Dyy.dtype)
        np.divide(Dyy, -sigma2, out = Ky)
        np.exp(Ky, out = Ky)
        Ky = np.kron(Ky, P)

        Mx = np.concatenate((Kx, -Kxy), axis=1)
        My = np.concatenate((Kyx, -Ky), axis=1)

        X_Te_proj = (Mx @ Us).reshape([nX_Te, d])
        Y_Te_proj = (My @ Us).reshape([nY_Te, d])
    

    data_Te_proj = np.concatenate((X_Te_proj, Y_Te_proj), axis=0)
    M = Pdist2(X_Te_proj, Y_Te_proj)
    PI = sinkhorn.sinkhorn_knopp(a,b,M,reg=reg)
    eta = np.sum(PI*M)
    eta_hist = np.float32(np.zeros(num_perm))

    for iboot in range(num_perm):
        tmp = np.random.permutation(n_Te)
        idx1_perm, idx2_perm = tmp[0:nX_Te], tmp[nX_Te:n_Te]

        X_Te_proj_perm = data_Te_proj[idx1_perm]
        Y_Te_proj_perm = data_Te_proj[idx2_perm]

        M = Pdist2(X_Te_proj_perm, Y_Te_proj_perm)
        PI = sinkhorn.sinkhorn_knopp(a,b,M,reg=reg)
        #PI = ot.sinkhorn(a,b,M,reg)

        eta_hist[iboot] = np.sum(PI*M)

    t_alpha = np.quantile(eta_hist, 0.95)
    if eta > t_alpha:
        decision = 1
    else:
        decision = 0
    return decision, eta, eta_hist