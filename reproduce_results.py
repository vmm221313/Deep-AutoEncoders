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

from nn_utils import AE_3D_200
from my_utils import load_data, make_plots, train_evaluate_default_model

num_epochs = 100
learning_rate = 3e-4

model = AE_3D_200()
model_name = 'AE_3D_200'
loss = train_evaluate_default_model(model, model_name, num_epochs , learning_rate)
loss
