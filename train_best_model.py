# -*- coding: utf-8 -*-
import os
import utils
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import corner.corner as corner
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import fastai
from fastai import train as tr 
from fastai.callbacks import ActivationStats
from fastai import data_block, basic_train, basic_data

import matplotlib as mpl
import my_matplotlib_style as ms
mpl.rc_file('my_matplotlib_rcparams')

from utils import plot_activations

if torch.cuda.is_available():
  fastai.torch_core.defaults.device = 'cuda'

from my_utils import load_data, make_plots, train_evaluate_model

class AE_3D_500cone_bn_custom(nn.Module):
    def __init__(self, hidden_dim_1, hidden_dim_2, hidden_dim_3, n_features=4):
        super(AE_3D_500cone_bn_custom, self).__init__()
        self.en1 = nn.Linear(n_features, hidden_dim_1)
        self.bn1 = nn.BatchNorm1d(hidden_dim_1)
        self.en2 = nn.Linear(hidden_dim_1,  hidden_dim_2)
        self.bn2 = nn.BatchNorm1d(hidden_dim_2)
        self.en3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.bn3 = nn.BatchNorm1d(hidden_dim_3)
        self.en4 = nn.Linear(hidden_dim_3, 3)
        self.bn5 = nn.BatchNorm1d(3)
        self.de1 = nn.Linear(3, hidden_dim_3)
        self.bn6 = nn.BatchNorm1d(hidden_dim_3)
        self.de2 = nn.Linear(hidden_dim_3, hidden_dim_2)
        self.bn7 = nn.BatchNorm1d(hidden_dim_2)
        self.de3 = nn.Linear(hidden_dim_2, hidden_dim_1)
        self.bn8 = nn.BatchNorm1d(hidden_dim_1)
        self.de4 = nn.Linear(hidden_dim_1, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        h1 = self.bn1(self.tanh(self.en1(x)))
        h2 = self.bn2(self.tanh(self.en2(h1)))
        h3 = self.bn3(self.tanh(self.en3(h2)))
        z = self.en4(h3)
        return z

    def decode(self, x):
        h5 = self.bn6(self.tanh(self.de1(self.bn5(self.tanh(x)))))
        h6 = self.bn7(self.tanh(self.de2(h5)))
        h7 = self.bn8(self.tanh(self.de3(h6)))
        return self.de4(h7)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        pass

num_epochs = 1000
hidden_dim_1 = 200 #Optimized using Grid Search. See tune_model.py
hidden_dim_2 = 300
hidden_dim_3 = 200
learning_rate = 3e-4

model = AE_3D_500cone_bn_custom(hidden_dim_1, hidden_dim_2, hidden_dim_3)
model_name = 'AE_3D_500cone_bn_custom'
loss = train_evaluate_model(model, model_name, num_epochs, learning_rate, hidden_dim_1, hidden_dim_2, hidden_dim_3)
loss

