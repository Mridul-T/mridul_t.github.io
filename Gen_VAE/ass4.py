#%%-
import pandas as pd
import numpy as np

# from IPython import get_ipython
# get_ipython().magic('reset -sf')

import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.set_default_dtype(torch.float64)

#%% DATA PREPROCESSING
df=pd.read_csv("BIKED_processed.csv",index_col=0)
scaler = MinMaxScaler()
normalized_data=scaler.fit_transform(df)
data = torch.utils.data.DataLoader(normalized_data,
        batch_size=128,
        shuffle=False)

#%% Encoder class
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(2395, 200)
        self.linear2 = nn.Linear(200, 200)
        self.linear3 = nn.Linear(200, 200)
        self.linear4 = nn.Linear(200, 200)
        self.linear5 = nn.Linear(200, 200)
        self.linear6 = nn.Linear(200, latent_dims)
        self.linear7 = nn.Linear(200, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        # x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        mu =  (self.linear6(x))
        sigma = torch.sigmoid(self.linear7(x))
        z = mu + sigma*self.N.sample(mu.shape)
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        self.kl = (0.5*(sigma**2 + mu**2) - torch.log(sigma) - 1/2).sum()
        # print(self.kl)
        return z

#%% Decoder class
class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 200)
        self.linear2 = nn.Linear(200, 200)
        self.linear3 = nn.Linear(200, 200)
        self.linear4 = nn.Linear(200, 200)
        self.linear5 = nn.Linear(200, 200)
        self.linear6 = nn.Linear(200, 2395)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = F.relu(self.linear3(z))
        z = F.relu(self.linear4(z))
        z = F.relu(self.linear5(z))
        z = torch.sigmoid(self.linear6(z))
        return z

#%% VAE class
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
#%%
def train(vae, data, epochs=20):
    opt = torch.optim.Adam(vae.parameters(),lr=0.001)
    for epoch in range(epochs):
        overall_loss = 0
        for x in data:
            x = x.to(device) # GPU
            opt.zero_grad()
            # print(vae.linear1.weight.dtype)
            x_hat = vae(x)
            loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            # print(loss)
            overall_loss += loss.item()
            # print(vae.encoder.kl)
            loss.backward()
            opt.step()
        print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(36*128))

#%%
latent_dims = 128

vae = VariationalAutoencoder(latent_dims).to(device)
print(type(vae))
train(vae, data)

#%%
#%% Saving & loading the model
torch.save(vae, 'BIKED_vae')
# vae = torch.load('BIKED_vae',map_location='cuda').to(device)
# %%
print(type(vae))
vae = torch.load('BIKED_vae',map_location='cpu').to('cpu')

import random
import deOH
from pathlib import Path

points=[]
points.append(random.randint(1,36))
points.append(random.randint(1,36))
i=1
z=[]
for x in data:
    if i in points:
        print(True)
        z.append(vae.encoder(x[19]))
    i=i+1
# print(len(z))
interpolation=[]
for i in range(10):
    interpolation.append((z[0]+i/10*(z[1]-z[0])))
# print(interpolation)
# interpolation=torch.utils.data.DataLoader(interpolation,batch_size=1,shuffle=False)
out=[]
for i in interpolation:
    i.to(device)
    out.append(vae.decoder(i).detach().numpy())
    
denormalized_data=scaler.inverse_transform(out)
col=df.columns
denorm_df=pd.DataFrame(denormalized_data,columns=col)
new_data=deOH.deOH(denorm_df)
new_data.to_csv('genDesign_VAE.csv')
#%%
import getXML
getXML.genBCAD(new_data,getXML.sourcepath,"C:\\Users\\91638\\Desktop\\SEM - 5\\APL302\\ass_4\\BCAD_models_v2\\model")
# %%
