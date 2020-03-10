# -*- coding: utf-8 -*-
import os
import utils
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from tqdm import tqdm_notebook
import corner.corner as corner
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, STATUS_OK

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

from my_utils import make_plots, load_data, train_evaluate_model
from utils import plot_activations

if torch.cuda.is_available():
    fastai.torch_core.defaults.device = 'cuda'

class AE_3D_200_custom(nn.Module):
    def __init__(self, hidden_dim_1, hidden_dim_2, hidden_dim_3, n_features=4):
        super(AE_3D_200_custom, self).__init__()
        self.en1 = nn.Linear(n_features, hidden_dim_1)
        self.en2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.en3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.en4 = nn.Linear(hidden_dim_3, 3)
        self.de1 = nn.Linear(3, hidden_dim_3)
        self.de2 = nn.Linear(hidden_dim_3, hidden_dim_2)
        self.de3 = nn.Linear(hidden_dim_2, hidden_dim_1)
        self.de4 = nn.Linear(hidden_dim_1, n_features)
        self.relu = nn.ReLU()

    def encode(self, x):
        return self.en4(self.relu(self.en3(self.relu(self.en2(self.relu(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.relu(self.de3(self.relu(self.de2(self.relu(self.de1(self.relu(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-' + str(hidden_dim_1) + '-' + str(hidden_dim_2) + '-' + str(hidden_dim_3) + '-' +'3-' + str(hidden_dim_3) + '-' + str(hidden_dim_2) + '-' + str(hidden_dim_1) + '-out'

global best_loss
best_loss = 10000

def gridSearch(space):
        global best_hidden_dim_1, best_hidden_dim_2, best_hidden_dim_3, best_loss
        hidden_dim_1 = int(space['hidden_dim_1'])
        hidden_dim_2 = int(space['hidden_dim_2'])
        hidden_dim_3 = int(space['hidden_dim_3'])

        model = AE_3D_200_custom(hidden_dim_1, hidden_dim_2, hidden_dim_3)
        model_name = 'AE_3D_200_custom'
        num_epochs = 50
        learning_rate = 3e-4
        loss = train_evaluate_model(model, model_name, num_epochs, learning_rate, hidden_dim_1, hidden_dim_2, hidden_dim_3)

        if (loss < best_loss):
            best_loss = loss
            best_hidden_dim_1 = hidden_dim_1
            best_hidden_dim_2 = hidden_dim_2
            best_hidden_dim_3 = hidden_dim_3
            print('best_loss = {}'.format(best_loss))
            print('best_hidden_dim_1 = {}'.format(hidden_dim_1))
            print('best_hidden_dim_2 = {}'.format(hidden_dim_2))
            print('best_hidden_dim_3 = {}'.format(hidden_dim_3))

        return {'loss': loss, 'status': STATUS_OK }

space = {
'hidden_dim_1': hp.quniform('hidden_dim_1', 50, 200, 50),
'hidden_dim_2': hp.quniform('hidden_dim_2', 100, 400, 100),
'hidden_dim_3': hp.quniform('hidden_dim_3', 50, 200, 50),    
}

best_scores = fmin(fn=gridSearch, space=space, algo=tpe.suggest, max_evals=100)

print('best_scores = {}'.format(best_scores))
print('best_loss = {}'.format(best_loss))

