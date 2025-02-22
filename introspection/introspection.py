import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


np.random.seed(0)
torch.manual_seed(0)
# random.seed(0)

data = np.load("../dataset/Least_Squares_3d_SGD/Sample_5/sample.npy")

data_tr = data[:100,:,:]
data_val = data[100:150,:,:]
data_te = data[150:,:,:]

def get_batch_n(data,batch_size,k):
#     if data.shape[2] > 20:
#         outputb = np.zeros((batch_size,4, 20))
#         outputl = np.zeros((batch_size,1, 20))
#     else: 
    outputb = np.zeros((batch_size,4, data.shape[2]))
    outputl = np.zeros((batch_size,1, data.shape[2]))
    indices = np.random.permutation(len(data))
    for i in range(batch_size):
        index = indices[i]
        b,l = get_batch(data[index,:,:],k)
        outputb[i,:,:] = b
        outputl[i,:,:] = l
        
    return outputb, outputl


def get_batch(train_data, k):
    
    # since we predict 2t, make sure all ts have corresponding tragets.
    maxval = train_data.shape[0] / 2
    data_width = train_data.shape[1]
    data_0 = train_data[0, :]

    
#    batch_indices = np.zeros(20, dtype=np.int32)
#     t = np.random.randint(low=1, high=maxval)

    t = 20 # now predict k*t

    t_1 = int(7 * t / 10)
    t_2 = int(4 * t / 10)
    data_t = train_data[t, :]
    data_t_1 = train_data[t_1, :]
    data_t_2 = train_data[t_2, :]

#     if data_width > 20:
#         diff = np.abs(data_0 - data_t)
#         sorted_diff = np.argsort(diff)
#         p_50_100 = sorted_diff[int(data_width / 2): ]
#         p_25_50 = sorted_diff[int(data_width / 4): int(data_width / 2)]
#         p_0_25 = sorted_diff[: int(data_width / 4)]

#         batch_indices[:10] = np.random.choice(p_50_100, 10, replace=False)
#         batch_indices[10:15] = np.random.choice(p_25_50, 5, replace=False)
#         batch_indices[15:] = np.random.choice(p_0_25, 5, replace=False)
#     else:
    batch_indices = np.arange(data_width)
    
    batch_t = np.take(data_t, batch_indices)

    batch_t_1 = np.take(data_t_1, batch_indices)
    batch_t_2 = np.take(data_t_2, batch_indices)
    batch_t_0 = np.take(data_0, batch_indices)
    labels = np.take(train_data[k * t, :], batch_indices)[np.newaxis, :]

    batch = np.vstack((batch_t, batch_t_1))
    batch = np.vstack((batch, batch_t_2))
    batch = np.vstack((batch, batch_t_0))

    return batch, labels
    

for k in [2,3,4,5,6,7,8,9,10]:

    tr_W1, tr_W2 = get_batch_n(data_tr,len(data_tr), k)
    tr_W1, tr_W2  = np.transpose(tr_W1, (0,2,1)),np.transpose(tr_W2, (0,2,1))
    val_W1, val_W2 = get_batch_n(data_val,len(data_val),k)
    val_W1, val_W2  = np.transpose(val_W1, (0,2,1)),np.transpose(val_W2, (0,2,1))
    te_W1, te_W2 = get_batch_n(data_te,len(data_te),k)
    te_W1, te_W2  = np.transpose(te_W1, (0,2,1)),np.transpose(te_W2, (0,2,1))



    # Convert numpy arrays to PyTorch tensors
    X_tensor = torch.from_numpy(tr_W1).float()
    y_tensor = torch.from_numpy(tr_W2).float()

    X_tensor_val = torch.from_numpy(val_W1).float()
    y_tensor_val = torch.from_numpy(val_W2).float()

    X_tensor_te = torch.from_numpy(te_W1).float()
    y_tensor_te = torch.from_numpy(te_W2).float()

    # Define the linear regression model
    class LinearRegressionModel(nn.Module):
        def __init__(self, input_size,output_size):
            super(LinearRegressionModel, self).__init__()
            self.linear = nn.Linear(input_size, output_size)

        def forward(self, x):
            return self.linear(x)

    input_size = tr_W1.shape[2]

    output_size = tr_W2.shape[2]

    learn_rate = 0.005

    model = LinearRegressionModel(input_size,output_size)

    # Loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    # Training the model
    num_epochs = 10000

    #batchsize
    batch_size = 32

    best_val_loss = float('inf')
    best_model = deepcopy(model.state_dict())
    val_loss_list = [] 
    impatience = 0 

    for epoch in range(num_epochs):
        # Forward pass
        indices = torch.randperm(len(X_tensor))

        for i in range(0, len(X_tensor), batch_size):
            batch_indices = indices[i:i+batch_size]

            outputs = model( X_tensor[batch_indices])

            # Compute the loss : 
            loss = torch.squeeze(torch.mean(torch.abs(outputs - y_tensor[batch_indices])))


            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #tr_loss = torch.squeeze(torch.mean(torch.abs(model(X_tensor) - y_tensor)))
        val_loss= torch.squeeze(torch.mean(torch.abs(model(X_tensor_val) - y_tensor_val)))
        #print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {tr_loss.item():.10f}, val Loss: {val_loss.item():.10f} ')

        if  val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = deepcopy(model.state_dict())
                impatient = 0 # reset
        else:
            impatient += 1


        if impatient >= 5:
          #  print(f'Breaking due to early stopping at epoch {epoch}')
            break

    # evaluate
    model.load_state_dict(best_model)

    model.eval()
    mse = torch.mean(torch.squeeze(torch.square(model(X_tensor_te) - y_tensor_te)))

    print("mse of test samples {} at predicting time step {} ".format(mse, k*20 ) )


