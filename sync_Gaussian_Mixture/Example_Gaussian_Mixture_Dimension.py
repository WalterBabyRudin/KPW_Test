import numpy as np
from numpy.lib import utils
import Optimization.RBCD_KPW as RBCD_KPW
import Optimization.RBCD_PW as RBCD_PW
import torch
import freqopttest.tst as tst
import freqopttest.data as tst_data
import Optimization.NTK_MMD as NTK_MMD
import Optimization.MMD_O as MMD_O
from Optimization.utils import MatConvert, MMDu, TST_MMD_u
from scipy.stats import multivariate_normal

torch.manual_seed(1102)
torch.cuda.manual_seed(1102)
np.random.seed(1)
print('---------------------------')
input_method = input("Enter Testing Method: ")
dtype = torch.float
device = torch.device("cpu")

def kernelwidthPair(x1, x2):
    '''Implementation of the median heuristic. See Gretton 2012
       Pick sigma such that the exponent of exp(- ||x-y|| / (2*sigma2)),
       in other words ||x-y|| / (2*sigma2),  equals 1 for the median distance x
       and y of all distances between points from both data sets X and Y.
    '''
    n, nfeatures = x1.shape
    m, mfeatures = x2.shape
    
    k1 = np.sum((x1*x1), 1)
    q = np.tile(k1, (m, 1)).transpose()
    del k1
    
    k2 = np.sum((x2*x2), 1)
    r = np.tile(k2, (n, 1))
    del k2
    
    h= q + r
    del q,r
    
    # The norm
    h = h - 2*np.dot(x1,x2.transpose())
    h = np.array(h, dtype=float)
    
    mdist = np.median([i for i in h.flat if i])

    return mdist


n_run = 10
n_test = 100
num_perm = 100
d = 3
def generate_Gaussian_mixture(N,D,seedNum):
    # input:
    #     N: sample size
    #     D: dimension of target distributions
    mu_1_hist = np.zeros([2,D])
    mu_1_hist[1] = mu_1_hist[1] + 1
    
    mu_2_hist = np.zeros([2,D])
    mu_2_hist[1] = mu_2_hist[1] + 1 + 0.8/np.sqrt(D)
    
    sigma_1 = np.identity(D)
    sigma_2_hist = [np.identity(D), np.identity(D)]
    sigma_2_hist[0][0,0] = 4
    sigma_2_hist[0][1,1] = 4
    sigma_2_hist[0][0,1] = -0.9
    sigma_2_hist[0][1,0] = -0.9
    sigma_2_hist[1][0,1] = 0.9
    sigma_2_hist[1][1,0] = 0.9
    
    X = np.zeros([N,D])
    Y = np.zeros([N,D])
    
    rv_X1 = multivariate_normal(mu_1_hist[0], sigma_1)
    rv_Y1 = multivariate_normal(mu_2_hist[0], sigma_2_hist[0])
    rv_X2 = multivariate_normal(mu_1_hist[1], sigma_1)
    rv_Y2 = multivariate_normal(mu_2_hist[1], sigma_2_hist[1])
    for i in range(N):
        uni_rand = np.random.rand()
        if uni_rand < 0.5:
            X[i, :] = rv_X1.rvs(1)#, random_state=seedNum)
            Y[i, :] = rv_Y1.rvs(1)#, random_state=seedNum+1)
        else:
            X[i, :] = rv_X2.rvs(1)#, random_state=seedNum)
            Y[i, :] = rv_Y2.rvs(1)#, random_state=seedNum+1)
    return X, Y
    


D_hist = [20,40,60,80,100,120,140,160]
L_D_hist = len(D_hist)
nX_Tr = 100
nY_Tr = 100
nX_Te = 100
nY_Te = 100

LD_hist = int(len(D_hist))
ME_decision_hist = np.zeros([LD_hist, n_test, n_run])
PW_decision_hist = np.zeros([LD_hist, n_test, n_run])
NTK_decision_hist = np.zeros([LD_hist, n_test, n_run])
MMDO_decision_hist = np.zeros([LD_hist, n_test, n_run])
KPW_decision_hist = np.zeros([LD_hist, n_test, n_run])

for id_D in range(LD_hist):
    D = D_hist[id_D]
    if D == 20:
        hsigma, orho = 1, 0.5
    elif D == 40:
        hsigma, orho = 0.5, 0.25
    elif D == 60:
        hsigma, orho = 1, 0.5
    elif D == 80:
        hsigma, orho = 2, 0.25
    elif D == 100:
        hsigma, orho = 1, 0.25
    elif D == 120:
        hsigma, orho = 1, 0.25
    elif D == 140:
        hsigma, orho = 2, 0.75
    elif D == 160:
        hsigma, orho = 0.5, 0.25

    for trial in range(n_run):
        randomNum = 5 + id_D*11 + trial * 1111 + n_run * 1741
        np.random.seed(randomNum)
        X_Tr, Y_Tr = generate_Gaussian_mixture(nX_Tr, D, seedNum = randomNum)
        alpha = 0.05
        
        if input_method == "ME":
        ### ME Test
            ME_opt = {
                'n_test_locs': 3, # number of test locations to optimize
                'max_iter': 200, # maximum number of gradient ascent iterations
                'locs_step_size': 1, # step size for the test locations (features)
                'gwidth_step_size': 0.5, # step size for the Gaussian width
                'tol_fun': 1e-4, # stop if the objective does not increase more than this.
                'seed': 10 + id_D*11 + trial * 1111 + n_run * 1741,  # random seed
                    }
            Data_Tr = tst_data.TSTData(X_Tr, Y_Tr, 'Tr_')
            test_locs, gwidth, info = tst.MeanEmbeddingTest.optimize_locs_width(Data_Tr, alpha, **ME_opt)
            met_opt = tst.MeanEmbeddingTest(test_locs, gwidth, alpha)

        elif input_method == "PW":
        ### PW Test
            alg_PW = RBCD_PW.RBCD_PW(eta=10,tau=0.1,maxiter=4000,tol=0.01,silence=True)
            _, U, _, _ = alg_PW.run_RBCD(X_Tr, Y_Tr, d)

        elif input_method == "NTK":
        ### NTK Test
            model = NTK_MMD.NTK_Train(X_Tr, Y_Tr, learning_rate=1e-3)

        elif input_method == "MMDO":
        ### MMD_O Test
            ep, sigma, sigma0_u, model_u = MMD_O.MMD_O(X_Tr, Y_Tr, silence=True)

        elif input_method == "KPW":
        #### KPW Test
            sigma2 = kernelwidthPair(X_Tr,Y_Tr)
            alg_KPW = RBCD_KPW.RBCD_kernel(eta=10,sigma=hsigma*sigma2,tau=1e-2,maxiter=200,tol=5e-2,orho=orho)
            Us = alg_KPW.run_RBCD(X_Tr,Y_Tr,d)
        
        for test_idx in range(n_test):
            randomNum = 7 + id_D*11 + trial * 1111 + n_run * 1741 + test_idx * 15125125
            np.random.seed(randomNum)
            
            X_Te, Y_Te = generate_Gaussian_mixture(nX_Te, D, seedNum = randomNum)
            
            if input_method == "ME":
            ## ME Test
                Data_Te = tst_data.TSTData(X_Te, Y_Te, 'Te_')
                test_result = met_opt.perform_test(Data_Te)
                ME_decision_hist[id_D, test_idx, trial] = np.float32(test_result['h0_rejected'])


            elif input_method == "PW": 
            ## PW Test
                decision, _, _ = RBCD_PW.PW_test(X_Te, Y_Te, U, num_perm = 100, reg=10)
                print('Idx: ',test_idx, '\t D: ',D, '\t Decision: ',decision)
                PW_decision_hist[id_D, test_idx, trial] = decision

            elif input_method == "NTK":
            ## NTK Test
                decision = NTK_MMD.NTK_test(X_Te, Y_Te, model)
                print('Idx: ',test_idx, '\t D: ',D, '\t Decision: ',decision)
                NTK_decision_hist[id_D, test_idx, trial] = decision

            elif input_method == "MMDO":
            ### MMD_O Test
                Data_Te = np.concatenate((X_Te, Y_Te), axis=0)
                Data_Te = MatConvert(Data_Te, device=device, dtype=dtype)
                h_u, _, _ = TST_MMD_u(model_u(Data_Te), num_perm, nX_Te, Data_Te, sigma, sigma0_u, alpha, device, dtype, ep)
                print('Idx: ',test_idx, '\t D: ',D, '\t Decision: ',h_u)
                MMDO_decision_hist[id_D, test_idx, trial] = h_u

            elif input_method == "KPW":
            #### KPW Test
                decision, _, _ = RBCD_KPW.KPW_Test(X_Te, Y_Te, X_Tr, Y_Tr, Us, sigma2*hsigma, d, orho=orho)
                print('Idx: ',test_idx, '\t D: ',D, '\t Decision: ',decision)
                KPW_decision_hist[id_D, test_idx, trial] = decision

if input_method == "ME":
    np.save('ME_decision_hist_GaussMixture_Dimension.npy', ME_decision_hist)
    print(np.mean(np.mean(ME_decision_hist, axis=1), axis=1))

elif input_method == "PW":
    np.save('PW_decision_hist_GaussMixture_Dimension.npy', PW_decision_hist)
    print(np.mean(np.mean(PW_decision_hist, axis=1), axis=1))

elif input_method == "NTK":
    np.save('NTK_decision_hist_GaussMixture_Dimension.npy', NTK_decision_hist)
    print(np.mean(np.mean(NTK_decision_hist, axis=1), axis=1))

elif input_method == "MMDO":
    np.save('MMDO_decision_hist_GaussMixture_Dimension.npy', MMDO_decision_hist)
    print(np.mean(np.mean(MMDO_decision_hist, axis=1), axis=1))

elif input_method == "KPW":      
    np.save('KPW_decision_hist_GaussMixture_Dimension.npy', KPW_decision_hist)
    print(np.mean(np.mean(KPW_decision_hist, axis=1), axis=1))