from mlxtend.data import loadlocal_mnist
import platform
import numpy as np
from numpy.lib import utils
import Optimization.RBCD_KPW as RBCD_KPW
import Optimization.RBCD_PW as RBCD_PW
import torch
import freqopttest.tst as tst
import freqopttest.data as tst_data
import Optimization.NTK_MMD as NTK_MMD
import Optimization.MMD_O as MMD_O
from Optimization.utils import MatConvert, MMDu, TST_MMD_u, TST_MMD_adaptive_bandwidth


torch.manual_seed(1102)
torch.cuda.manual_seed(1102)
np.random.seed(1)
print('---------------------------')
input_method = input("Enter Testing Method: ")
dtype = torch.float
device = torch.device("cpu")

np.random.seed(1)
Feature_Tr, label_Tr = loadlocal_mnist(
            images_path='MNIST/train-images.idx3-ubyte', 
            labels_path='MNIST/train-labels.idx1-ubyte')
Feature_Te, label_Te = loadlocal_mnist(
            images_path='MNIST/t10k-images.idx3-ubyte', 
            labels_path='MNIST/t10k-labels.idx1-ubyte')

N, D = np.shape(Feature_Tr)
id_digit_1 = np.where(label_Te == 1)[0]
id_all = np.arange(len(label_Tr))

def sample_mnist_Tr_H0(n):
    # sample mnist data to format training set
    # n: number of trainig samples
    # fract: fraction of digit 1
    
    N, D = np.shape(Feature_Tr)
    id_Tr = np.random.choice(N, 2*n, replace=False)
    id_X = id_Tr[:n]
    id_Y = id_Tr[n:]
    
    X = Feature_Tr[id_X]
    Y = Feature_Tr[id_Y]
    return np.float32(X), np.float32(Y), id_Tr

def sample_mnist_Te_H0(n,ind_Te):

    # sample mnist data to format testing set
    # n: number of testing samples
    # id_Tr: index for training set (without digit 1)
    # id_Y_1: index for training set with digit 1
    # fract: fraction of digit 1
    id_Te = np.random.choice(len(ind_Te), n*2, replace=False)
    id_Te = ind_Te[id_Te]
    id_X = id_Te[:n]
    id_Y = id_Te[n:]

    X = Feature_Tr[id_X,:]
    Y = Feature_Tr[id_Y,:]

    return np.float32(X), np.float32(Y)

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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

n_run = 10
n_test = 100
num_perm = 100
d = 3

N_hist = [200,250,300,400,500]
L_N_hist = len(N_hist)


ME_decision_hist = np.zeros([L_N_hist, n_test, n_run])
PW_decision_hist = np.zeros([L_N_hist, n_test, n_run])
NTK_decision_hist = np.zeros([L_N_hist, n_test, n_run])
MMDO_decision_hist = np.zeros([L_N_hist, n_test, n_run])
KPW_decision_hist = np.zeros([L_N_hist, n_test, n_run])


for id_N in range(L_N_hist):
    N = N_hist[id_N]
    if N == 200:
        hsigma, orho = 1,0.75
    elif N == 250:
        hsigma, orho = 0.5, 0.25
    elif N == 300:
        hsigma, orho = 1,0.25
    elif N == 400:
        hsigma, orho = 1, 0.5
    elif N == 500:
        hsigma, orho = 1, 0.5

    nX_Tr, nY_Tr, nX_Te, nY_Te = N, N, N, N
    for trial in range(n_run):
        randomNum = 5 + id_N*11 + trial * 1111
        np.random.seed(randomNum)
        
        X_Tr, Y_Tr, id_Tr = sample_mnist_Tr_H0(nX_Tr)
        
        X_Tr = sigmoid(X_Tr)
        Y_Tr = sigmoid(Y_Tr)
        alpha = 0.05
        
        if input_method == "ME":
        ### ME Test
            ME_opt = {
                'n_test_locs': 3, # number of test locations to optimize
                'max_iter': 200, # maximum number of gradient ascent iterations
                'locs_step_size': 1, # step size for the test locations (features)
                'gwidth_step_size': 0.5, # step size for the Gaussian width
                'tol_fun': 1e-4, # stop if the objective does not increase more than this.
                'seed': 10 + id_N*11 + trial * 1111 + n_run * 1741,  # random seed
                    }
            Data_Tr = tst_data.TSTData(X_Tr, Y_Tr, 'Tr_')
            test_locs, gwidth, info = tst.MeanEmbeddingTest.optimize_locs_width(Data_Tr, alpha, **ME_opt)
            met_opt = tst.MeanEmbeddingTest(test_locs, gwidth, alpha)
        
        elif input_method == "PW":
        ### PW Test
            alg_PW = RBCD_PW.RBCD_PW(eta=40,tau=0.01,maxiter=100,tol=0.1,silence=False)
            _, U, _, _ = alg_PW.run_RBCD(X_Tr, Y_Tr, d)
        
        elif input_method == "NTK":
        ### NTK Test
            model = NTK_MMD.NTK_Train(X_Tr, Y_Tr, learning_rate=4e-3)

        elif input_method == "MMDO":        
        ### MMD_O Test
            sigma, sigma0 = MMD_O.MMD_O_MNIST(X_Tr, Y_Tr, learning_rate=0.0001,silence=True)

        elif input_method == "KPW":
        #### KPW Test
            sigma2 = kernelwidthPair(X_Tr,Y_Tr)
            alg_KPW = RBCD_KPW.RBCD_kernel(eta=100,sigma=hsigma*sigma2,tau=1e-4,maxiter=100,tol=5e-1,orho=orho)
            Us = alg_KPW.run_RBCD(X_Tr,Y_Tr,d)
        
        ind_Te = np.delete(id_all, id_Tr)
        
        for test_idx in range(n_test):
            randomNum = 7 + id_N*11 + trial * 1111 + test_idx * 125125
            np.random.seed(randomNum)

            X_Te, Y_Te = sample_mnist_Te_H0(nX_Te,ind_Te)
            X_Te = sigmoid(X_Te)
            Y_Te = sigmoid(Y_Te)
            
            if input_method == "ME":
            ## ME Test
                Data_Te = tst_data.TSTData(X_Te, Y_Te, 'Te_')
                test_result = met_opt.perform_test(Data_Te)
                ME_decision_hist[id_N, test_idx, trial] = np.float32(test_result['h0_rejected'])
                print('Idx: ',test_idx, '\t N: ',N, '\t Decision: ',np.float32(test_result['h0_rejected']))
            
            elif input_method == "PW":
            ## PW Test
                decision, _, _ = RBCD_PW.PW_test(X_Te, Y_Te, U, num_perm = 100, reg=60)
                print('Idx: ',test_idx, '\t N: ',N, '\t Decision: ',decision)
                PW_decision_hist[id_N, test_idx, trial] = decision

            elif input_method == "NTK":
            ## NTK Test
                decision = NTK_MMD.NTK_test(X_Te, Y_Te, model)
                print('Idx: ',test_idx, '\t N: ',N, '\t Decision: ',decision)
                NTK_decision_hist[id_N, test_idx, trial] = decision

            elif input_method == "MMDO":
            ### MMD_O Test
                Data_Te = np.concatenate((X_Te, Y_Te), axis=0)
                Data_Te = MatConvert(Data_Te, device=device, dtype=dtype)
                #h_u, _, _ = TST_MMD_u(model_u(Data_Te), num_perm, nX_Te, Data_Te, sigma, sigma0_u, alpha, device, dtype, ep)
                h_u, _, _ = TST_MMD_adaptive_bandwidth(Data_Te, num_perm, nX_Te, Data_Te, sigma, sigma0, alpha, device, dtype)
                print('Idx: ',test_idx, '\t N: ',N, '\t Decision: ',h_u)
                MMDO_decision_hist[id_N, test_idx, trial] = h_u

            elif input_method == "KPW":
            ### KPW Test
                decision, _, _ = RBCD_KPW.KPW_Test(X_Te, Y_Te, X_Tr, Y_Tr, Us, sigma2, d,reg=100)
                print('Idx: ',test_idx, '\t N: ',N, '\t Decision: ',decision)
                KPW_decision_hist[id_N, test_idx, trial] = decision



            
            
if input_method == "ME":
    #np.save('ME_decision_hist_MNIST.npy', ME_decision_hist)
    print(np.mean(np.mean(ME_decision_hist, axis=1), axis=1))
    print(np.var(np.mean(ME_decision_hist, axis=1), axis=1))

elif input_method == "PW":
    #np.save('PW_decision_hist_MNIST.npy', PW_decision_hist)
    print(np.mean(np.mean(PW_decision_hist, axis=1), axis=1))
    print(np.var(np.mean(PW_decision_hist, axis=1), axis=1))

elif input_method == "NTK":
    # np.save('NTK_decision_hist_MNIST.npy', NTK_decision_hist)
    print(np.mean(np.mean(NTK_decision_hist, axis=1), axis=1))
    print(np.var(np.mean(NTK_decision_hist, axis=1), axis=1)) 

elif input_method == "MMDO":
    #np.save('MMDO_decision_hist_MNIST.npy', MMDO_decision_hist)
    print(np.mean(np.mean(MMDO_decision_hist, axis=1), axis=1))
    print(np.var(np.mean(MMDO_decision_hist, axis=1), axis=1))

elif input_method == "KPW":
    #np.save('KPW_decision_hist_MNIST.npy', PW_decision_hist)
    print(np.mean(np.mean(KPW_decision_hist, axis=1), axis=1))
    print(np.var(np.mean(KPW_decision_hist, axis=1), axis=1))