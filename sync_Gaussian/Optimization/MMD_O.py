import numpy as np
import torch
from Optimization.utils import MatConvert, MMDu, TST_MMD_u, Pdist2


class ModelLatentF(torch.nn.Module):
    """Latent space for both domains."""
    def __init__(self, x_in, H, x_out):
        """Init latent features."""
        super(ModelLatentF, self).__init__()
        self.restored = False
        self.latent = torch.nn.Sequential(
            torch.nn.Linear(x_in, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, x_out, bias=True),
        )
    def forward(self, input):
        """Forward the LeNet."""
        fealant = self.latent(input)
        return fealant

np.random.seed(1102)
torch.manual_seed(1102)
torch.cuda.manual_seed(1102)
dtype = torch.float
device = torch.device("cpu")


def MMD_O(X_Tr, Y_Tr, learning_rate=0.005, N_epoch=1000, silence = True):
    n_X, D = np.shape(X_Tr)
    n_Y, _ = np.shape(Y_Tr)

    x_in = D
    H = 50       # number of neurons in the hidden layer
    x_out = 50   # number of neurons in the output layer
    model_u = ModelLatentF(x_in, H, x_out)

    epsilonOPT = MatConvert(np.random.rand(1) * (10 ** (-10)), device, dtype)
    epsilonOPT.requires_grad = True
    sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * D), device, dtype)
    sigmaOPT.requires_grad = True
    sigma0OPT = MatConvert(np.ones(1) * np.sqrt(2 * D), device, dtype)
    sigma0OPT.requires_grad = True

    optimizer_u = torch.optim.Adam(
                list(model_u.parameters())+[epsilonOPT]+[sigmaOPT]+[sigma0OPT], 
                lr=learning_rate)
    
    Data_Tr = np.concatenate((X_Tr, Y_Tr), axis=0)
    Data_Tr = MatConvert(Data_Tr, device=device, dtype=dtype)

    #J_star_u = np.zeros([N_epoch,])
    for t in range(N_epoch):
        # Compute epsilon, sigma and sigma_0
        ep = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
        sigma = sigmaOPT ** 2
        sigma0_u = sigma0OPT ** 2

        # Compute output of the deep network
        modelu_output = model_u(Data_Tr)
        # Compute J (STAT_u)
        TEMP = MMDu(modelu_output, n_X, Data_Tr, sigma, sigma0_u, ep)
        mmd_value_temp = -1 * (TEMP[0] + 10 ** (-8))
        mmd_std_temp = torch.sqrt(TEMP[1]+10**(-8))
        #print(mmd_std_temp)
        STAT_u = torch.div(mmd_value_temp, mmd_std_temp)

        #J_star_u[t] = STAT_u.item()

        optimizer_u.zero_grad()
        STAT_u.backward(retain_graph=True)
        optimizer_u.step()
        if (t % 100 == 0) and not(silence):
            print("mmd_value: ", -1 * mmd_value_temp.item(), 
                  "mmd_std: ", mmd_std_temp.item(), 
                  "Statistic J: ", -1 * STAT_u.item())
    return ep, sigma, sigma0_u, model_u


def MMD_O_MNIST(X_Tr, Y_Tr, learning_rate=0.005, N_epoch=1000, silence = True):
    n_X, D = np.shape(X_Tr)
    n_Y, _ = np.shape(Y_Tr)
    Data_Tr = np.concatenate((X_Tr, Y_Tr), axis=0)
    Data_Tr = MatConvert(Data_Tr, device=device, dtype=dtype)
    
    Dxy = Pdist2(Data_Tr[:n_X, :], Data_Tr[n_X:, :])
    
    
    
    sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * D), device, dtype)
    sigmaOPT.requires_grad = True
    #sigma0 = MatConvert(np.ones(1) * np.sqrt(0.1), device, dtype)
    sigma0 = Dxy.median() * 3
    sigma0.requires_grad = True
    optimizer_u = torch.optim.Adam([sigma0]+[sigmaOPT], lr=learning_rate)
    
    for t in range(N_epoch):
        sigma = sigmaOPT ** 2
        TEMPa = MMDu(Data_Tr, n_X, Data_Tr, sigma, sigma0, is_smooth=False)
        mmd_value_tempa = -1 * (TEMPa[0] + 10 ** (-8))
        mmd_std_tempa = torch.sqrt(TEMPa[1] + 10 ** (-8))
        STAT_adaptive = torch.div(mmd_value_tempa, mmd_std_tempa)
        
        optimizer_u.zero_grad()
        STAT_adaptive.backward(retain_graph=True)
        optimizer_u.step()
        if (t % 100 == 0) and not(silence):
            print("mmd_value: ", -1 * mmd_value_tempa.item(), 
                  "mmd_std: ", mmd_std_tempa.item(), 
                  "Statistic J: ", -1 * STAT_adaptive.item())
    return sigma, sigma0
    
    


    