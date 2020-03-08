# -*- coding: utf-8 -*-
# !pip install corner

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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from nn_utils import *
from run_model import train_evaluate_model

if torch.cuda.is_available():
  fastai.torch_core.defaults.device = 'cuda'

models = [AE_3D_100, AE_3D_200, AE_3D_small, AE_3D_small_v2, AE_big, AE_big_no_last_bias, AE_3D_50, 
          AE_3D_50_no_last_bias, AE_3D_50cone, AE_3D_500cone_bn, AE_big_2D_v1, AE_big_2D_v2, 
          AE_2D, AE_2D_v2, AE_big_2D_v3, AE_2D_v3, AE_2D_v4, AE_2D_v5, AE_2D_v100, AE_2D_v50,
          AE_2D_v1000]

dropout_models = [AE_3D_50_bn_drop, AE_3D_50cone_bn_drop, AE_3D_100_bn_drop, AE_3D_100cone_bn_drop, 
                  AE_3D_200_bn_drop]

num_epochs = 100

model_losses = {}
min_loss = 10000000
best_model = AE_3D_100

for model_class in tqdm_notebook(models):
  model_name = str(model_class).split('.')[1].split("'")[0]
  model = model_class()
  loss = train_evaluate_model(model_class, num_epochs);
  model_losses[model_name] = loss

  if loss < min_loss:
    min_loss = loss
    best_model = model
    print(loss)

print('Best Model -> {}'.format(str(best_model).split('.')[1].split("'")[0]))

print('Min loss = {}'.format(min_loss))



