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
dtype = torch.float
device = torch.device("cpu")
torch.manual_seed(1102)
torch.cuda.manual_seed(1102)
np.random.seed(1)
if __name__ == "__main__":
    np.random.seed(1)
    print('---------------------------')
    input_method = input("Enter Testing Method: ")


    D_hist = [4,8,16,32,64,128,256]
    nX_Tr = 50
    nY_Tr = 50
    nX_Te = 50
    nY_Te = 50
    n_run = 10
    n_test = 100
    num_perm = 100
    d = 3

    ######################################################
    ######### Exam the power for Var Shited Gauss ########
    ######################################################
    LD_hist = int(len(D_hist))
    ME_decision_hist = np.zeros([LD_hist, n_test, n_run])
    PW_decision_hist = np.zeros([LD_hist, n_test, n_run])
    NTK_decision_hist = np.zeros([LD_hist, n_test, n_run])
    MMDO_decision_hist = np.zeros([LD_hist, n_test, n_run])
    KPW_decision_hist = np.zeros([LD_hist, n_test, n_run])

    for id_D in range(LD_hist):
        D = D_hist[id_D]

        if D == 4:
            hsigma, orho = 0.5, 0.5
        elif D == 8:
            hsigma, orho = 1, 0.25
        elif D == 16:
            hsigma, orho = 1, 0.25
        elif D == 32:
            hsigma, orho = 1, 0.5
        elif D == 64:
            hsigma, orho = 2, 0.5
        elif D == 128:
            hsigma, orho = 1, 0.25
        elif D == 256:
            hsigma, orho = 0.5, 0.5

        mean_1 = np.zeros(D)
        mean_2 = np.zeros(D)

        Cov_1 = np.eye(D)
        Cov_2 = np.eye(D)
        Cov_2[0,0], Cov_2[1,1], Cov_2[2,2] = 4, 4, 4
        d = 3

        for trial in range(n_run):
            np.random.seed(5 + id_D*11 + trial * 1111 + n_run * 1741)
            X_Tr = np.random.multivariate_normal(mean_1, Cov_1, size=nX_Tr)
            Y_Tr = np.random.multivariate_normal(mean_2, Cov_2, size=nY_Tr)
            alpha = 0.05

            #### ME Test
            if input_method == "ME":
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
            #### PW Test
                alg_PW = RBCD_PW.RBCD_PW(eta=10,tau=0.1,maxiter=4000,tol=0.01,silence=True)
                _, U, _, _ = alg_PW.run_RBCD(X_Tr, Y_Tr, d)

            elif input_method == "NTK":
            #### NTK Test
                model = NTK_MMD.NTK_Train(X_Tr, Y_Tr, learning_rate=1e-3)

            elif input_method == "MMDO":
            #### MMD_O Test
                ep, sigma, sigma0_u, model_u = MMD_O.MMD_O(X_Tr, Y_Tr, silence=False)

            elif input_method == "KPW":
            #### KPW Test
                sigma2 = kernelwidthPair(X_Tr,Y_Tr)
                alg_KPW = RBCD_KPW.RBCD_kernel(eta=10,sigma=hsigma*sigma2,tau=1e-2,maxiter=1000,tol=1e-1,orho=orho)
                Us = alg_KPW.run_RBCD(X_Tr,Y_Tr,d)

            for test_idx in range(n_test):
                np.random.seed(5 + id_D*11 + trial * 1111 + n_run * 1741 + 17467*test_idx)

                X_Te = np.random.multivariate_normal(mean_1, Cov_1, size=nX_Te)
                Y_Te = np.random.multivariate_normal(mean_2, Cov_2, size=nY_Te)

                if input_method == "ME":
                ## ME Test
                    Data_Te = tst_data.TSTData(X_Te, Y_Te, 'Te_')
                    test_result = met_opt.perform_test(Data_Te)
                    ME_decision_hist[id_D, test_idx, trial] = np.float32(test_result['h0_rejected'])
                
                elif input_method == "PW":                
                ### PW Test
                    decision, _, _ = RBCD_PW.PW_test(X_Te, Y_Te, U, num_perm = 100, reg=10)
                    print('Idx: ',test_idx, '\t D: ',D, '\t Decision: ',decision)
                    PW_decision_hist[id_D, test_idx, trial] = decision
                
                elif input_method == "NTK":
                ### NTK Test
                    decision = NTK_MMD.NTK_test(X_Te, Y_Te, model)
                    print('Idx: ',test_idx, '\t D: ',D, '\t Decision: ',decision)
                    NTK_decision_hist[id_D, test_idx, trial] = decision

                elif input_method == "MMDO":
                #### MMD_O Test
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
        np.save('ME_decision_hist_Gauss_Var_shifted.npy', ME_decision_hist)
        print(np.mean(np.mean(ME_decision_hist, axis=1), axis=1))

    elif input_method == "PW":
        np.save('PW_decision_hist_Gauss_Var_shifted.npy', PW_decision_hist)
        print(np.mean(np.mean(PW_decision_hist, axis=1), axis=1))

    elif input_method == "NTK":
        np.save('NTK_decision_hist_Gauss_Var_shifted.npy', NTK_decision_hist)
        print(np.mean(np.mean(NTK_decision_hist, axis=1), axis=1))

    elif input_method == "MMDO":
        np.save('MMDO_decision_hist_Gauss_Var_shifted.npy', MMDO_decision_hist)
        print(np.mean(np.mean(MMDO_decision_hist, axis=1), axis=1))

    elif input_method == "KPW":
        np.save('KPW_decision_hist_Gauss_Var_shifted.npy', KPW_decision_hist)
        print(np.mean(np.mean(KPW_decision_hist, axis=1), axis=1))



    # # ######################################################
    # # ######### Exam the level for Var Shited Gauss ########
    # # ######################################################
    # LD_hist = int(len(D_hist))
    # ME_decision_hist = np.zeros([LD_hist, n_test, n_run])
    # PW_decision_hist = np.zeros([LD_hist, n_test, n_run])
    # NTK_decision_hist = np.zeros([LD_hist, n_test, n_run])
    # MMDO_decision_hist = np.zeros([LD_hist, n_test, n_run])
    # KPW_decision_hist = np.zeros([LD_hist, n_test, n_run])

    # for id_D in range(LD_hist):
    #     D = D_hist[id_D]

    #     if D == 4:
    #         hsigma, orho = 0.5, 0.5
    #     elif D == 8:
    #         hsigma, orho = 1, 0.25
    #     elif D == 16:
    #         hsigma, orho = 1, 0.25
    #     elif D == 32:
    #         hsigma, orho = 1, 0.5
    #     elif D == 64:
    #         hsigma, orho = 2, 0.5
    #     elif D == 128:
    #         hsigma, orho = 1, 0.25
    #     elif D == 256:
    #         hsigma, orho = 0.5, 0.5

    #     mean_1 = np.zeros(D)
    #     mean_2 = np.zeros(D)

    #     Cov_1 = np.eye(D)
    #     Cov_2 = np.eye(D)

    #     for trial in range(n_run):
    #         np.random.seed(5 + id_D*11 + trial * 1111 + n_run * 1741)
    #         X_Tr = np.random.multivariate_normal(mean_1, Cov_1, size=nX_Tr)
    #         Y_Tr = np.random.multivariate_normal(mean_2, Cov_2, size=nY_Tr)
    #         alpha = 0.05

    #         if input_method == "ME":
    #         ### ME Test
    #             ME_opt = {
    #                 'n_test_locs': 3, # number of test locations to optimize
    #                 'max_iter': 200, # maximum number of gradient ascent iterations
    #                 'locs_step_size': 1, # step size for the test locations (features)
    #                 'gwidth_step_size': 0.5, # step size for the Gaussian width
    #                 'tol_fun': 1e-4, # stop if the objective does not increase more than this.
    #                 'seed': 10 + id_D*11 + trial * 1111 + n_run * 1741,  # random seed
    #                     }
    #             Data_Tr = tst_data.TSTData(X_Tr, Y_Tr, 'Tr_')
    #             test_locs, gwidth, info = tst.MeanEmbeddingTest.optimize_locs_width(Data_Tr, alpha, **ME_opt)
    #             met_opt = tst.MeanEmbeddingTest(test_locs, gwidth, alpha)

    #         elif input_method == "PW":
    #         ### PW Test
    #             alg_PW = RBCD_PW.RBCD_PW(eta=10,tau=0.1,maxiter=4000,tol=0.01,silence=True)
    #             _, U, _, _ = alg_PW.run_RBCD(X_Tr, Y_Tr, d)

    #         elif input_method == "NTK":
    #         ### NTK Test
    #             model = NTK_MMD.NTK_Train(X_Tr, Y_Tr, learning_rate=1e-3)

    #         elif input_method == "MMDO":
    #         ### MMD_O Test
    #             ep, sigma, sigma0_u, model_u = MMD_O.MMD_O(X_Tr, Y_Tr, silence=True)

    #         elif input_method == "KPW":
    #         #### KPW Test
    #             sigma2 = kernelwidthPair(X_Tr,Y_Tr)
    #             alg_KPW = RBCD_KPW.RBCD_kernel(eta=10,sigma=hsigma*sigma2,tau=1e-2,maxiter=1000,tol=1e-1,orho=orho)
    #             Us = alg_KPW.run_RBCD(X_Tr,Y_Tr,d)


    #         for test_idx in range(n_test):
    #             np.random.seed(5 + id_D*11 + trial * 1111 + n_run * 1741 + 17467*test_idx)

    #             X_Te = np.random.multivariate_normal(mean_1, Cov_1, size=nX_Te)
    #             Y_Te = np.random.multivariate_normal(mean_2, Cov_2, size=nY_Te)

    #             if input_method == "ME":
    #             ### ME Test
    #                 Data_Te = tst_data.TSTData(X_Te, Y_Te, 'Te_')
    #                 test_result = met_opt.perform_test(Data_Te)
    #                 ME_decision_hist[id_D, test_idx, trial] = np.float32(test_result['h0_rejected'])

    #             elif input_method == "PW":
    #             ### PW Test
    #                 decision, _, _ = RBCD_PW.PW_test(X_Te, Y_Te, U, num_perm = 100, reg=10)
    #                 print('Idx: ',test_idx, '\t D: ',D, '\t Decision: ',decision)
    #                 PW_decision_hist[id_D, test_idx, trial] = decision

    #             elif input_method == "NTK":
    #             ### NTK Test
    #                 decision = NTK_MMD.NTK_test(X_Te, Y_Te, model)
    #                 print('Idx: ',test_idx, '\t D: ',D, '\t Decision: ',decision)
    #                 NTK_decision_hist[id_D, test_idx, trial] = decision

    #             elif input_method == "MMDO":
    #             ### MMD_O Test
    #                 Data_Te = np.concatenate((X_Te, Y_Te), axis=0)
    #                 Data_Te = MatConvert(Data_Te, device=device, dtype=dtype)
    #                 h_u, _, _ = TST_MMD_u(model_u(Data_Te), num_perm, nX_Te, Data_Te, sigma, sigma0_u, alpha, device, dtype, ep)
    #                 print('Idx: ',test_idx, '\t D: ',D, '\t Decision: ',h_u)
    #                 MMDO_decision_hist[id_D, test_idx, trial] = h_u

    #             elif input_method == "KPW":
    #             #### KPW Test
    #                 decision, _, _ = RBCD_KPW.KPW_Test(X_Te, Y_Te, X_Tr, Y_Tr, Us, sigma2*hsigma, d, orho=orho)
    #                 print('Idx: ',test_idx, '\t D: ',D, '\t Decision: ',decision)
    #                 KPW_decision_hist[id_D, test_idx, trial] = decision


    # if input_method == "ME":
    #     np.save('ME_decision_hist_Gauss_Var_shifted_level.npy', ME_decision_hist)
    #     print(np.mean(np.mean(ME_decision_hist, axis=1), axis=1))

    # elif input_method == "PW":
    #     np.save('PW_decision_hist_Gauss_Var_shifted_level.npy', PW_decision_hist)
    #     print(np.mean(np.mean(PW_decision_hist, axis=1), axis=1))

    # elif input_method == "NTK":
    #     np.save('NTK_decision_hist_Gauss_Var_shifted_level.npy', NTK_decision_hist)
    #     print(np.mean(np.mean(NTK_decision_hist, axis=1), axis=1))

    # elif input_method == "MMDO":
    #     np.save('MMDO_decision_hist_Gauss_Var_shifted_level.npy', MMDO_decision_hist)
    #     print(np.mean(np.mean(MMDO_decision_hist, axis=1), axis=1))

    # elif input_method == "KPW":
    #     np.save('KPW_decision_hist_Gauss_Var_shifted_level.npy', KPW_decision_hist)
    #     print(np.mean(np.mean(KPW_decision_hist, axis=1), axis=1))
