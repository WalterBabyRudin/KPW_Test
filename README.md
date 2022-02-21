### Kernel Projected Wasserstein Distance
Codes for "Two-sample Test with Kernel Projected Wasserstein Distance"


### Demo
Here we presents a demo for using the KPW Test:
<pre><code>import numpy as np
import Optimization.RBCD_KPW as RBCD_KPW
''' generating data set '''
mean_1 = np.zeros(8)
mean_2 = np.zeros(8)
Cov_1 = np.eye(8)
Cov_2 = np.eye(8)
Cov_2[0,0], Cov_2[1,1], Cov_2[2,2] = 4,4,4
X_Tr = np.random.multivariate_normal(mean_1, Cov_1, size=50)
Y_Tr = np.random.multivariate_normal(mean_2, Cov_2, size=50)
''' Training projector '''
alg_KPW = RBCD_KPW.RBCD_kernel(eta=10,sigma=20,tau=1e-3,maxiter=1000,tol=1e-1)
Us = alg_KPW.run_RBCD(X_Tr,Y_Tr,3)
''' Testing '''
X_Te = np.random.multivariate_normal(mean_1, Cov_1, size=50)
Y_Te = np.random.multivariate_normal(mean_2, Cov_2, size=50)
decision, _, _ = RBCD_KPW.KPW_Test(X_Te, Y_Te, X_Tr, Y_Tr, Us, 20, 3)
print(decision)</code></pre>
