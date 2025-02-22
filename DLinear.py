import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Invertible import RevIN
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from copy import deepcopy

data = np.load("dataset/Least_Squares_3d_SGD/Sample_5/sample.npy")

data_tr = data[:100,:,:]
data_val = data[100:150,:,:]
data_te = data[150:,:,:]

np.random.seed(0)
torch.manual_seed(0)


class Model(nn.Module):
    def __init__(self, channel, d_model, seq_len, pred_len):
        super(Model, self).__init__()
# Temporal mixing, you can choose any type of extractor such as Attention, Conv 
        self.temporal = nn.Sequential(
        nn.Linear(seq_len, d_model),
        nn.GELU(),
        nn.Linear(d_model, seq_len)
)
        # Channel mixing, you can choose any type of extractor such as MLP, GNN, and et
        self.channel = nn.Sequential(
            nn.Linear(channel, d_model),
            nn.GELU(),
            nn.Linear(d_model, channel)
        )
        self.projection = nn.Linear(seq_len, pred_len)
        self.rev = RevIN(channel)
        self.apply(self._init_weights)
        # Frozen randomly initialized parameters in temporal feature extractor if neces
        # for p in self.temporal.parameters():
        #     p.requires_grad = False
    # Design your initialization, or random init
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    # Design your loss function
    def forward_loss(self, pred, true):
        return F.mse_loss(pred, true)

    def forward(self, x, y): # x: [B, L, D]
        x = self.rev(x, 'norm')
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)
        pred = self.projection(x.transpose(1, 2)).transpose(1, 2)
        pred = self.rev(pred, 'denorm')
        return pred, self.forward_loss(pred, y)


tr_W1 = data_tr[:,:21,:]
tr_W2 = data_tr[:,21:,:]
val_W1 = data_val[:,:21,:]
val_W2 = data_val[:,21:,:]
te_W1 = data_te[:,:21,:]
te_W2 = data_te[:,21:,:]

X_tensor = torch.from_numpy(tr_W1).float()
y_tensor = torch.from_numpy(tr_W2).float()
X_tensor_val = torch.from_numpy(val_W1).float()
y_tensor_val = torch.from_numpy(val_W2).float()
X_tensor_te = torch.from_numpy(te_W1).float()
y_tensor_te = torch.from_numpy(te_W2).float()


model = Model(channel= X_tensor.shape[2], d_model=512, seq_len=X_tensor.shape[1], pred_len=y_tensor.shape[1])

learn_rate = 0.001
# Loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=learn_rate)
#scheduler = ExponentialLR(optimizer, gamma=1)
# Training the model
num_epochs = 10000 #batchsize
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
    
        _, loss = model( X_tensor[batch_indices], y_tensor[batch_indices] )

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    
   # _, tr_loss = model(X_tensor, y_tensor)
    _, val_loss= model(X_tensor_val, y_tensor_val)
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

pred,_ = model(X_tensor_te,y_tensor_te)

for k in [2,3,4,5,6,7,8,9,10]:
    mse = torch.mean(torch.squeeze(torch.square(pred[:,20*(k-1)-1,:] - y_tensor_te[:,20*(k-1)-1,:])))

    print("mse of test samples {} at predicting time step {} ".format(mse, k*20 ) )
