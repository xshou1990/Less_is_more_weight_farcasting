import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from collections.abc import Iterable

# generate gradient
def grad(X,y,w, SGD):
    indices = np.random.permutation(len(X))
    if SGD == 1:
        sampX = X[indices[0:8]]
        sampY = y[indices[0:8]]
    else:
        sampX = X[indices[0:len(X)]]
        sampY = y[indices[0:len(y)]]
        
    return (np.matmul(np.matmul(sampX.T,sampX),w)-np.matmul(sampX.T,sampY))

def gen_weight(X,y,w,rate ,SGD):
    w_new = w - rate* grad(X,y,w,SGD)
    return w_new 

# generate weight seq of length k
def gen_weight_seq(k,X,y,w_init,rate, SGD):
    w_seq = [w_init] 
    while k >0:
        w_next = gen_weight(X,y,w_init,rate, SGD)
        w_seq.append(w_next)
        w_init = w_next 
        k -= 1
    return w_seq

# m is # of sequences of weights
# k is sequence length
# d is dimension 
# sgd == 1 then SGD else SGD
def generate_m_seqs(m, k, d, seed, SGD ):
    tot_seq = []
    tot_x = []
    tot_y = []

    np.random.seed(seed)
    mean = np.zeros(d)
    cov = np.identity(d)
    w_star = np.random.multivariate_normal(mean, cov, size=m)
    
    for i in range(m):
        np.random.seed(i)
        X = np.random.multivariate_normal(mean, cov, size=100) # size = 100 x d
        tot_x.append(X)
        y = np.matmul(X, w_star[i])
        tot_y.append(y)
        if SGD == 0:
            hessian = np.matmul(X.T, X)
            eigval,_ = np.linalg.eig(hessian)         
            rate = 0.01*1.0/eigval.max() 
        else:
            rate = 0.001
        w_init = np.random.uniform(0,1,d) 
        my_w_seq = gen_weight_seq(k,X,y,w_init,rate, SGD )

        tot_seq.append(my_w_seq)
        
    return tot_seq




def flatten(x):
    if isinstance(x, Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def gen_weight_seq_nn(k,X,y,rate,opt):
    # Convert numpy arrays to PyTorch tensors
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(np.expand_dims(y,axis=1)).float()

    # Define the linear regression model
    class LinearRegressionModel(nn.Module):
        def __init__(self, input_size,output_size):
            super(LinearRegressionModel, self).__init__()
            self.linear1 = nn.Linear(input_size, 10, bias=True)
            self.linear2 = nn.Linear(10, output_size, bias=True)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.linear2(self.relu(self.linear1(x)))

    model = LinearRegressionModel(X.shape[1],1)
     
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Number of parameters: {total_params}")
    if opt == 'Adam':
        # Loss and optimizer
        optimizer = optim.SGD(model.parameters(), lr=rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=rate)        
        # Training the model
    num_epochs = 101

    #batchsize
    batch_size = 64
    
    param = []
    loss_his = [] 
    for epoch in range(num_epochs):
        # Forward pass
        indices = torch.randperm(len(X_tensor))

        for i in range(0, len(X_tensor), batch_size):
            batch_indices = indices[i:i+batch_size]

            outputs = model( X_tensor[batch_indices])

            # Compute the loss : || Psi(X) - y ||_2 ^2
            loss = torch.mean(torch.square(outputs - y_tensor[batch_indices]))
#             print(loss)
            
            loss_his.append(loss)
            
            tot = []
            for name in model.parameters():
                tot.append(name.detach().numpy().tolist())
            
            param.append(np.array(flatten(tot)))
            
            if len(param) == k:
                return param, model, loss_his  
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    

# m is # of sequences of weights
# k is sequence length
# d is dimension 
def generate_m_seqs_nn(m, k, d,seed, opt):
    tot_seq = []
    tot_model = []
    tot_loss = []
    tot_x = []
    tot_y = []

    np.random.seed(seed)
    mean = np.zeros(d)
    cov = np.identity(d)
    w_star = np.random.multivariate_normal(mean, cov, size=m)
    
    for i in range(m):
        np.random.seed(i)
        X = np.random.multivariate_normal(mean, cov, size=100) # size = 100 x d
        tot_x.append(X)
        y = np.matmul(X, w_star[i])
        tot_y.append(y)

        rate = 0.002
      #  w_init = np.random.uniform(0,1,d) 
        my_w_seq, model,loss_his = gen_weight_seq_nn(k,X,y,rate,opt)
        tot_seq.append(my_w_seq)
        tot_model.append(model)
        tot_loss.append(loss_his)
        
    return tot_seq

