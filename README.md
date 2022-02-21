### Kernel Projected Wasserstein Distance
Codes for "Two-sample Test with Kernel Projected Wasserstein Distance"


### Demo
Here we presents a demo for using the KPW Test:
<pre><code>import numpy as np
import Optimization.RBCD_KPW as RBCD_KPW
''' generating data set '''
mean\_1 = np.zeros(8)
mean\_2 = np.zeros(8)
Cov\_1 = np.eye(8)
Cov\_2 = np.eye(8)
Cov\_2[0,0], Cov\_2[1,1], Cov\_2[2,2] = 4,4,4
X\_Tr = np.random.multivariate\_normal(mean\_1, Cov\_1, size=50)
Y\_Tr = np.random.multivariate\_normal(mean\_2, Cov\_2, size=50)
''' Training projector '''
alg\_KPW = RBCD\_KPW.RBCD\_kernel(eta=10,sigma=20,tau=1e-3,maxiter=1000,tol=1e-1)
Us = alg\_KPW.run\_RBCD(X\_Tr,Y\_Tr,3)
''' Testing '''
X\_Te = np.random.multivariate\_normal(mean\_1, Cov\_1, size=50)
Y\_Te = np.random.multivariate\_normal(mean\_2, Cov\_2, size=50)
decision, _, _ = RBCD\_KPW.KPW\_Test(X\_Te, Y\_Te, X\_Tr, Y\_Tr, Us, 20, 3)
print(decision)</code></pre>
