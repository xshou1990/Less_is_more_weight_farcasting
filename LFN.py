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

    
tr_W1 = data_tr[:,:21,:]
tr_W2 = data_tr[:,21:,:]
val_W1 = data_val[:,:21,:]
val_W2 = data_val[:,21:,:]
te_W1 = data_te[:,:21,:]
te_W2 = data_te[:,21:,:]

tr_W1, tr_W2  = np.transpose(tr_W1, (0,2,1)),np.transpose(tr_W2, (0,2,1))
val_W1, val_W2  = np.transpose(val_W1, (0,2,1)),np.transpose(val_W2, (0,2,1))
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
        print(f'Breaking due to early stopping at epoch {epoch}')
        break


# evaluate
model.load_state_dict(best_model)
    
model.eval()

pred = model(X_tensor_te)

for k in [2,3,4,5,6,7,8,9,10]:
    mse = torch.mean(torch.squeeze(torch.square(pred[:,:,20*(k-1)-1] - y_tensor_te[:,:,20*(k-1)-1])))

    print("mse of test samples {} at predicting time step {} ".format(mse, k*20 ) )


