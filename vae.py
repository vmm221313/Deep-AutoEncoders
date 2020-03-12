# -*- coding: utf-8 -*-
import os
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable

from my_utils import load_data, train_evaluate_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tag = '4D/'
if not os.path.exists(tag):
	os.mkdir(tag)

class InfoVAE(nn.Module):
    def __init__(self, nfeat=4, ncode=3, alpha=0, lambd=10000, nhidden=128, nhidden2=35, dropout=0.2):
        super(InfoVAE, self).__init__()
        
        self.ncode = int(ncode)
        self.alpha = float(alpha)
        self.lambd = float(lambd)
        
        self.encd = nn.Linear(nfeat, nhidden)
        self.d1 = nn.Dropout(p=dropout)
        self.enc2 = nn.Linear(nhidden, nhidden2)
        self.d2 = nn.Dropout(p=dropout)
        self.mu = nn.Linear(nhidden2, ncode)
        self.lv = nn.Linear(nhidden2, ncode)
        
        self.decd = nn.Linear(ncode, nhidden2)
        self.d3 = nn.Dropout(p=dropout)
        self.dec2 = nn.Linear(nhidden2, nhidden)
        self.d4 = nn.Dropout(p=dropout)
        self.outp = nn.Linear(nhidden, nfeat)
        
    def encode(self, x):
        x = self.d1(F.leaky_relu(self.encd(x)))
        x = self.d2(F.leaky_relu(self.enc2(x)))
        mu = self.mu(x)
        logvar = self.lv(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def decode(self, x):
        x = self.d3(F.leaky_relu(self.decd(x)))
        x = self.d4(F.leaky_relu(self.dec2(x)))
        x = self.outp(x)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    # https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/
    def compute_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1).to(device) # (x_size, 1, dim)
        y = y.unsqueeze(0).to(device) # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim).to(device)
        tiled_y = y.expand(x_size, y_size, dim).to(device)
        # The example code divides by (dim) here, making <kernel_input> ~ 1/dim
        # excluding (dim) makes <kernel_input> ~ 1
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2)#/float(dim)
        kernel_input = kernel_input.to(device)
        return torch.exp(-kernel_input) # (x_size, y_size)
    
    # https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/
    def compute_mmd(self, x, y):
        xx_kernel = self.compute_kernel(x,x)
        yy_kernel = self.compute_kernel(y,y)
        xy_kernel = self.compute_kernel(x,y)
        return torch.mean(xx_kernel) + torch.mean(yy_kernel) - 2*torch.mean(xy_kernel)
    
    def loss(self, x):
        recon_x, mu, logvar = self.forward(x)
        MSE = torch.sum(0.5 * (x - recon_x).pow(2))
        
        # KL divergence (Kingma and Welling, https://arxiv.org/abs/1312.6114, Appendix B)
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        #return MSE + self.beta*KLD, MSE
                
        # https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/
        true_samples = Variable(torch.randn(200, self.ncode), requires_grad=False)
        z = self.reparameterize(mu, logvar) #duplicate call
        # compute MMD ~ 1, so upweight to match KLD which is ~ n_batch x n_code
        MMD = self.compute_mmd(true_samples,z) * x.size(0) * self.ncode
        return MSE + (1-self.alpha)*KLD + (self.lambd+self.alpha-1)*MMD, MSE, KLD, MMD

train_dl, valid_dl, train_x, train_y, test_x, test_y = load_data(batch_size = 512)

def train(): #model, optimizer, epoch, min_valid_loss, badepochs
    model.train()
    train_loss = 0
    train_logL = 0
    for batch_idx, data in enumerate(train_dl):
        x = data[0].to(device)
        optimizer.zero_grad()
        loss, logL, KLD, MMD = model.loss(x)
        loss = loss.to(device)
        logL = logL.to(device)
        KLD = KLD.to(device)
        MMD = MMD.to(device)
        loss.backward()
        train_loss += loss.item()
        train_logL += logL.item()
        optimizer.step()
    train_loss /= len(train_dl.dataset)
    
    with torch.no_grad():
        model.eval()
        valid_loss = 0
        valid_logL = 0
        valid_KLD = 0
        valid_MMD = 0

        for data in valid_dl:
            x = data[0].to(device)
            loss, logL, KLD, MMD = model.loss(x)
            valid_loss += loss.item()
            valid_logL += logL.item()
            valid_KLD += KLD.item()
            valid_MMD += MMD.item()
        
        valid_loss /= len(valid_dl.dataset)
        valid_logL /= -len(valid_dl.dataset)
        valid_KLD  /= len(valid_dl.dataset)
        valid_MMD  /= len(valid_dl.dataset)
    return valid_loss, valid_logL, valid_KLD, valid_MMD

class EarlyStopper:
    def __init__(self, precision=1e-3, patience=10):
        self.precision = precision
        self.patience = patience
        self.badepochs = 0
        self.min_valid_loss = float('inf')
        
    def step(self, valid_loss):
        if valid_loss < self.min_valid_loss*(1-self.precision):
            self.badepochs = 0
            self.min_valid_loss = valid_loss
        else:
            self.badepochs += 1
        return not (self.badepochs == self.patience)

epochs = 100
log_interval = 10
mdl_ncode = 3
n_config = 100
nfeat = 4

mdl_MSE = np.zeros(n_config)
mdl_KLD = np.zeros(n_config)
mdl_MMD = np.zeros(n_config)

for config in range(n_config):
    alpha = 0
    lambd = np.exp(np.random.uniform(0, np.log(1e5)))
    dropout = 0#0.9*np.random.uniform()
    dfac = 1./(1.-dropout)

    nhidden = int(np.ceil(np.exp(np.random.uniform(np.log(dfac*mdl_ncode+1), np.log(dfac*2*nfeat)))))*50
    nhidden2 = int(np.ceil(np.exp(np.random.uniform(np.log(dfac*mdl_ncode+1), np.log(nhidden)))))*50
    print('config %i, alpha = %0.1f, lambda = %0.1f, dropout = %0.2f; 2 hidden layers with %i, %i nodes' % (config, alpha, lambd, dropout, nhidden, nhidden2))
    model = InfoVAE(alpha=alpha, lambd=lambd, nfeat=nfeat, nhidden=nhidden, nhidden2=nhidden2, ncode=mdl_ncode, dropout=dropout)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=5)
    stopper = EarlyStopper(patience=10)

    for epoch in tqdm((range(1, epochs + 1))):
        valid_loss, valid_logL, valid_KLD, valid_MMD = train()
        if epoch % log_interval == 0:
            print('====> Epoch: {} VALIDATION Loss: {:.2e} logL: {:.2e} KL: {:.2e} MMD: {:.2e}'.format(
                  epoch, valid_loss, valid_logL, valid_KLD, valid_MMD))

        scheduler.step(valid_loss)
        if (not stopper.step(valid_loss)) or (epoch == epochs):
            print('Stopping')
            print('====> Epoch: {} VALIDATION Loss: {:.2e} logL: {:.2e} KL: {:.2e} MMD: {:.2e}'.format(
                  epoch, valid_loss, valid_logL, valid_KLD, valid_MMD))
            model.MSE = -valid_logL
            model.KLD = valid_KLD
            model.MMD = valid_MMD
            mdl_MSE[config] = model.MSE
            mdl_KLD[config] = model.KLD
            mdl_MMD[config] = model.MMD
            torch.save(model, tag+'/%04i.pth' % config)
            break

