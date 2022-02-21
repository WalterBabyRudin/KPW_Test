import numpy as np
from numpy.lib import utils
from torch.nn.functional import threshold
import Optimization.RBCD_KPW as RBCD_KPW
import Optimization.RBCD_PW as RBCD_PW
import torch
import freqopttest.tst as tst
import freqopttest.data as tst_data
import Optimization.NTK_MMD as NTK_MMD
import Optimization.MMD_O as MMD_O
from Optimization.utils import MatConvert, MMDu, TST_MMD_u
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt 
import Optimization.sinkhorn as sinkhorn

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

torch.manual_seed(1102)
torch.cuda.manual_seed(1102)
np.random.seed(1)
print('---------------------------')
input_method = input("Enter Testing Method: ")
dtype = torch.float
device = torch.device("cpu")
data_bending = np.load('Guest/data_bending_4.npy')
data_throwing = np.load('Guest/data_throwing_4.npy')


N_throwing, D = np.shape(data_throwing)
data_total = np.concatenate((data_bending, data_throwing), axis=0)
W = 100

X_Tr = data_total[0:W,:]
Y_Tr = data_total[W:2*W,:]
alpha = 0.05


eta_reg = 100

if input_method == "ME":
# ### ME Test
    ME_opt = {
        'n_test_locs': 1, # number of test locations to optimize
        'max_iter': 200, # maximum number of gradient ascent iterations
        'locs_step_size': 1, # step size for the test locations (features)
        'gwidth_step_size': 0.5, # step size for the Gaussian width
        'tol_fun': 1e-4, # stop if the objective does not increase more than this.
        'seed': 10,  # random seed
            }
    Data_Tr = tst_data.TSTData(X_Tr, Y_Tr, 'Tr_')
    test_locs, gwidth, info = tst.MeanEmbeddingTest.optimize_locs_width(Data_Tr, alpha, **ME_opt)
    met_opt = tst.MeanEmbeddingTest(test_locs, gwidth, alpha)

elif input_method == "PW":
### PW Test
    np.random.seed(4)
    idx_XY = np.sort(np.random.choice(300, 2*W, replace=False))
    X_Tr = data_total[idx_XY[:W], :]
    Y_Tr = data_total[idx_XY[W:], :]
    d = 3
    alg_PW = RBCD_PW.RBCD_PW(eta=1,tau=0.1,maxiter=200,tol=0.01,silence=True)
    _, U, _, _ = alg_PW.run_RBCD(X_Tr, Y_Tr, d)

elif input_method == "NTK":
### NTK Test
    model = NTK_MMD.NTK_Train(X_Tr, Y_Tr, learning_rate=1e-3)

elif input_method == "MMDO":
# MMD-O Test
    ep, sigma, sigma0_u, model_u = MMD_O.MMD_O(X_Tr, Y_Tr, silence=True)

elif input_method == "KPW":
### KPW Test
    np.random.seed(4)
    idx_XY = np.sort(np.random.choice(300, 2*W, replace=False))
    X_Tr = data_total[idx_XY[:W], :]
    Y_Tr = data_total[idx_XY[W:], :]
    d = 3
    sigma2 = kernelwidthPair(X_Tr,Y_Tr)
    alg_KPW = RBCD_KPW.RBCD_kernel(eta=eta_reg,sigma=sigma2,tau=1e-3,maxiter=500,tol=5e-2)
    Us = alg_KPW.run_RBCD(X_Tr,Y_Tr,d)

num_trial = 1000
Stat_hist = np.zeros(num_trial)
if input_method == "ME":
## ME Test
    for trial in range(num_trial):
        np.random.seed(trial*44 + 5)
        idx_XY = np.sort(np.random.choice(300, 2*W, replace=False))
        X_Te = data_total[idx_XY[:W], :]
        Y_Te = data_total[idx_XY[W:], :]
        Data_Te = tst_data.TSTData(X_Te, Y_Te, 'Te_')
        stat = met_opt.compute_stat(Data_Te)
        print('Trial: ',trial, '\t Stat: ',stat)
        Stat_hist[trial] = stat
    np.save('ME_Guest_stat_hist_4',Stat_hist)
    Threshold = np.quantile(Stat_hist, 0.95)
    print(Threshold)

elif input_method == "PW":
# ## PW Test
    a = np.float64(np.ones(W)/W)
    b = np.float64(np.ones(W)/W)
    for trial in range(num_trial):
        np.random.seed(trial*44 + 5)
        idx_XY = np.sort(np.random.choice(400, 2*W, replace=False))
        X_Te = data_total[idx_XY[:W], :] @ U
        Y_Te = data_total[idx_XY[W:], :] @ U
        M = RBCD_PW.Pdist2(X_Te, Y_Te)
        PI = sinkhorn.sinkhorn_knopp(a,b,M,reg=1)
        stat = np.sum(PI*M)
        print('Trial: ',trial, '\t Stat: ',stat)
        Stat_hist[trial] = stat
    np.save('PW_Guest_stat_hist_4',Stat_hist)
    Threshold = np.quantile(Stat_hist, 0.95)#Threshold = np.max(Stat_hist)
    print(Threshold)

elif input_method == "NTK":
# NTK Test
    for trial in range(num_trial):
        np.random.seed(trial*44 + 5)
        idx_XY = np.sort(np.random.choice(400, 2*W, replace=False))
        X_Te = data_total[idx_XY[:W], :]
        Y_Te = data_total[idx_XY[W:], :]
        stat = NTK_MMD.NTK_test(X_Te, Y_Te, model, compute_stat=True)
        print('Trial: ',trial, '\t Stat: ',stat)
        Stat_hist[trial] = np.abs(stat)
    np.save('NTK_Guest_stat_hist_4',Stat_hist)
    Threshold = np.quantile(Stat_hist, 0.95)
    print(Threshold)

elif input_method == "MMDO":
# MMD-O Test
    for trial in range(num_trial):
        np.random.seed(trial*44 + 5)
        idx_XY = np.sort(np.random.choice(400, 2*W, replace=False))
        X_Te = data_total[idx_XY[:W], :]
        Y_Te = data_total[idx_XY[W:], :]
        #idx_X = np.sort(np.random.choice(3*W, W, replace=False))
        #X_Te = data_total[idx_X, :]
        Data_Te = np.concatenate((X_Te, Y_Te), axis=0)
        Data_Te = MatConvert(Data_Te, device=device, dtype=dtype)
        _,_,Stat = TST_MMD_u(model_u(Data_Te), 10, W, Data_Te, sigma, sigma0_u, alpha, device, dtype, ep)
        print('Trial: ',trial, '\t Stat: ',Stat)
        Stat_hist[trial] = Stat
    np.save('MMDO_Guest_stat_hist_4',Stat_hist)
    Threshold = np.quantile(Stat_hist, 0.95)
    print(Threshold)

elif input_method == "KPW":
    ## KPW Test
    a = np.float64(np.ones(W)/W)
    b = np.float64(np.ones(W)/W)
    for trial in range(num_trial):
        np.random.seed(trial*44 + 5)
        idx_XY = np.sort(np.random.choice(300, 2*W, replace=False))
        X_Te = data_total[idx_XY[:W], :]
        Y_Te = data_total[idx_XY[W:], :]
        _, stat, _ = RBCD_KPW.KPW_Test(X_Te, Y_Te, X_Tr, Y_Tr, Us, sigma2, d,num_perm=1,reg=eta_reg)
        print('Trial: ',trial, '\t Stat: ',stat)
        Stat_hist[trial] = stat
    np.save('KPW_Guest_stat_hist_4',Stat_hist)
    # Stat_hist = np.load('KPW_Guest_stat_hist.npy')
    Threshold = np.quantile(Stat_hist, 0.95)
    print(Threshold)

Time_length =100 + N_throwing
ME_Test_Stat_hist = np.zeros(Time_length)
PW_Test_Stat_hist = np.zeros(Time_length)
NTK_Test_Stat_hist = np.zeros(Time_length)
MMDO_Test_Stat_hist = np.zeros(Time_length)
KPW_Test_Stat_hist = np.zeros(Time_length)


np.random.seed(100)
idx_X = np.sort(np.random.choice(3*W, W, replace=False))
X_Te = data_total[idx_X, :]
if input_method == "PW":
    X_Te = X_Te @ U
#X_Te = data_total[:W,:] @ U

for t in range(Time_length):
    Time_idx = 400+t
    Y_Te = data_total[Time_idx-W:Time_idx,:]
    if input_method == "PW":
        Y_Te = Y_Te @ U
    
    if input_method == "ME":
    ### ME Test
        Data_Te = tst_data.TSTData(X_Te, Y_Te, 'Te_')
        stat = met_opt.compute_stat(Data_Te)
        if stat > Threshold:
            decision = 1
        else:
            decision = 0
        print('Time Index: ',Time_idx,'\t Decision: ',decision, '\t Stat: ',stat)
        ME_Test_Stat_hist[t] = stat

    elif input_method == "PW":
        # PW Test
        M = RBCD_PW.Pdist2(X_Te, Y_Te)
        PI = sinkhorn.sinkhorn_knopp(a,b,M,reg=1)
        stat = np.sum(PI*M)
        if stat > Threshold:
            decision = 1
        else:
            decision = 0
        print('Time Index: ',Time_idx,'\t Decision: ',decision, '\t Stat: ',stat)
        PW_Test_Stat_hist[t] = stat

    elif input_method == "NTK":
        # NTK Test
        stat = NTK_MMD.NTK_test(X_Te, Y_Te, model, compute_stat=True)
        stat = np.abs(stat)
        if stat > Threshold:
            decision = 1
        else:
            decision = 0
        print('Time Index: ',Time_idx,'\t Decision: ',decision, '\t Stat: ',stat)
        NTK_Test_Stat_hist[t] = stat

    elif input_method == "MMDO":
    # MMD-O Test
        Data_Te = np.concatenate((X_Te, Y_Te), axis=0)
        Data_Te = MatConvert(Data_Te, device=device, dtype=dtype)
        _,_,Stat = TST_MMD_u(model_u(Data_Te), 10, W, Data_Te, sigma, sigma0_u, alpha, device, dtype, ep)
        if Stat > Threshold:
            decision = 1
        else:
            decision = 0
        print('Time Index: ',Time_idx,'\t Decision: ',decision, '\t Stat: ',Stat)
        MMDO_Test_Stat_hist[t] = Stat

    elif input_method == "KPW":
        # KPW Test
        _, stat, _ = RBCD_KPW.KPW_Test(X_Te, Y_Te, X_Tr, Y_Tr, Us, sigma2, d,num_perm=1,reg=eta_reg)
        if stat > Threshold:
            decision = 1
        else:
            decision = 0
        print('Time Index: ',Time_idx,'\t Decision: ',decision, '\t Stat: ',stat)
        KPW_Test_Stat_hist[t] = stat

if input_method == "ME":
    np.save('ME_Guest_Test_Stat_hist_4',ME_Test_Stat_hist)

    fig = plt.figure(figsize= (8,3))
    plt.plot(ME_Test_Stat_hist,'.-')
    plt.title('ME Test Stat')
    plt.xlabel('sample index')
    plt.show()
elif input_method == "PW":
    np.save('PW_Guest_Test_Stat_hist_4',PW_Test_Stat_hist)

    fig = plt.figure(figsize= (8,3))
    plt.plot(PW_Test_Stat_hist,'.-')
    plt.title('PW Test Stat')
    plt.xlabel('sample index')
    plt.show()
elif input_method == "NTK":
    np.save('NTK_Test_Stat_hist_4',NTK_Test_Stat_hist)

    fig = plt.figure(figsize= (8,3))
    plt.plot(NTK_Test_Stat_hist,'.-')
    plt.title('NTK Test Stat')
    plt.xlabel('sample index')
    plt.show()
elif input_method == "MMDO":
    np.save('MMDO_Test_Stat_hist_4',MMDO_Test_Stat_hist)

    fig = plt.figure(figsize= (8,3))
    plt.plot(MMDO_Test_Stat_hist,'.-')
    plt.title('MMDO Test Stat')
    plt.xlabel('sample index')
    plt.show()
elif input_method == "KPW":
    np.save('KPW_Guest_Test_Stat_hist_4',KPW_Test_Stat_hist)

    fig = plt.figure(figsize= (8,3))
    plt.plot(KPW_Test_Stat_hist,'.-')
    plt.title('KPW Test Stat')
    plt.xlabel('sample index')
    plt.show()