import matplotlib.pyplot as plt 
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


np.random.seed(1102)
torch.manual_seed(1102)
torch.cuda.manual_seed(1102)
dtype = torch.float
device = torch.device("cpu")
num_neurons = 512

class nn_one_hidden_layer(nn.Module):
    def __init__(self,D):
        super( nn_one_hidden_layer, self).__init__()
        input_dim = D
        num_classes = 1
        
        self.fc1 = nn.Linear(input_dim, num_neurons)
        self.fc2 = nn.Linear(num_neurons, num_classes)

    def forward(self, x):
        x = F.softplus(self.fc1(x)) #softplus activation function
        x = self.fc2(x)
        return x

def NTK_Train(X_Tr, Y_Tr, learning_rate=1e-3):

    # X_Tr = np.float32(X_Tr)
    # Y_Tr = np.float32(Y_Tr)

    nX_Tr, D = np.shape(X_Tr)
    nY_Tr, _ = np.shape(Y_Tr)
    n_Tr = nX_Tr + nY_Tr

    #data_Tr = torch.tensor(np.concatenate((X_Tr, Y_Tr), axis=0))
    data_Tr = torch.tensor(np.concatenate((X_Tr, Y_Tr), axis=0).astype(np.float32))
    labels_Tr = torch.tensor(np.concatenate( (np.zeros(nX_Tr),  np.ones(nY_Tr)), axis=0), dtype=int)

    model = nn_one_hidden_layer(D=D).to(device) #init model
    # specificy 1st layer weights to be N(0,1)
    torch.manual_seed(1102)
    fc1_weight_value = torch.randn(model.fc1.weight.shape)
    fc1_bias_value = torch.zeros_like(model.fc1.bias)
    model.fc1.weight = torch.nn.parameter.Parameter(fc1_weight_value)
    model.fc1.bias = torch.nn.parameter.Parameter(fc1_bias_value)
    # specificy 2nd layer weights to be N(0,1/sqrt(M)) 
    fc2_weight_value = torch.randn(model.fc2.weight.shape)/np.sqrt(num_neurons)
    fc2_weight_bias = torch.zeros_like(model.fc2.bias)
    model.fc2.weight = torch.nn.parameter.Parameter(fc2_weight_value)
    model.fc2.bias = torch.nn.parameter.Parameter(fc2_weight_bias)
    # make 2nd layer not trainable
    model.fc2.weight.requires_grad = False
    model.fc2.bias.requires_grad= False


    optimizer = optim.SGD(model.parameters(), 
                      lr=learning_rate,
                      momentum=0, dampening=0, 
                      weight_decay=0, nesterov=False)

    model.train()

    for itr in range(0, n_Tr ):
        optimizer.zero_grad()
        
        # forward pass
        outputi = model(data_Tr[itr,:]).reshape([-1,1])
        loss = F.nll_loss( torch.cat( (outputi, -outputi),1),
                        labels_Tr[itr].reshape([1]), reduction='sum')
        
        #backward pass
        loss.backward()
        optimizer.step()
    return model

def NTK_test(X_Te, Y_Te, model, num_perm = 100, learning_rate=1e-3, compute_stat = False):
    # X_Tr = np.float32(X_Te)
    # Y_Tr = np.float32(Y_Te)
    nX_Te, D = np.shape(X_Te)
    nY_Te, _ = np.shape(Y_Te)
    n_Te = nX_Te + nY_Te
    data_Te =  torch.tensor(np.concatenate( (X_Te,  Y_Te), axis=0).astype(np.float32))
    labels_Te = torch.tensor(np.concatenate( (np.zeros(nX_Te),  np.ones(nY_Te)), axis=0), dtype=int)


    model0 = nn_one_hidden_layer(D=D).to(device)
    torch.manual_seed(1102)
    model0.fc1.weight = torch.nn.parameter.Parameter(torch.randn(model0.fc1.weight.shape))
    model0.fc1.bias = torch.nn.parameter.Parameter(torch.zeros_like(model0.fc1.bias))
    model0.fc2.weight = torch.nn.parameter.Parameter(torch.randn(model0.fc2.weight.shape)/np.sqrt(num_neurons))
    model0.fc2.bias = torch.nn.parameter.Parameter(torch.zeros_like(model0.fc2.bias))

    with torch.no_grad():
        fte0 = model0(data_Te)
    
    with torch.no_grad():
        fteT = model(data_Te)

    gte = (fteT - fte0)/learning_rate
    idx1_Te = (labels_Te == 0).nonzero(as_tuple=True)[0].numpy()
    idx2_Te = (labels_Te == 1).nonzero(as_tuple=True)[0].numpy()
    eta = np.mean(gte[idx1_Te].numpy())-np.mean(gte[idx2_Te].numpy())

    if compute_stat:
        return eta
    else:
        etastore = np.float32( np.zeros(num_perm) )
        for iboot in range(num_perm):
            tmp = torch.randperm(n_Te).numpy()
            idx1_perm, idx2_perm = tmp[0:nX_Te], tmp[nX_Te:n_Te]
            eta_perm = np.mean(gte[idx1_perm].numpy())-np.mean(gte[idx2_perm].numpy())
            etastore[iboot] = eta_perm
        talpha = np.quantile(etastore, 0.95)

        decision = np.float32( eta > talpha )
        return decision








