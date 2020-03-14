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


def make_plots_dir(model_class):
    model_name = str(model_class).split('.')[1].split("'")[0]
    if not os.path.exists('plots/' + model_name + '/'):
        os.mkdir('plots/' + model_name)

    plots_path = 'plots/' + model_name + '/'

    return plots_path, model_name


batch_size = 256

# +
train = pd.read_pickle('processed_data/all_jets_train_4D_100_percent.pkl')
test = pd.read_pickle('processed_data/all_jets_test_4D_100_percent.pkl')
n_features = len(train.loc[0])

train = (train - train.mean()) / train.std()
test = (test - test.mean()) / test.std()

train_x = train
test_x = test
train_y = train_x  # y = x since we are building an AE (ideally the ouput should be the same as the input)
test_y = test_x

train_ds = TensorDataset(torch.tensor(train_x.values, dtype = torch.float), torch.tensor(train_y.values, dtype = torch.float))
valid_ds = TensorDataset(torch.tensor(test_x.values, dtype = torch.float), torch.tensor(test_y.values, dtype = torch.float))

train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size = batch_size)

# -

train_ds.tensors


def load_data(batch_size):
    train = pd.read_pickle('processed_data/all_jets_train_4D_100_percent.pkl')
    test = pd.read_pickle('processed_data/all_jets_test_4D_100_percent.pkl')
    n_features = len(train.loc[0])
    
    train = (train - train.mean()) / train.std()
    test = (test - test.mean()) / test.std()
    
    train_x = train
    test_x = test
    train_y = train_x  # y = x since we are building an AE (ideally the ouput should be the same as the input)
    test_y = test_x
    
    train_ds = TensorDataset(torch.tensor(train_x.values, dtype = torch.float), torch.tensor(train_y.values, dtype = torch.float))
    valid_ds = TensorDataset(torch.tensor(test_x.values, dtype = torch.float), torch.tensor(test_y.values, dtype = torch.float))
    
    train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size = batch_size)
    
    return train_dl, valid_dl, train_x, train_y, test_x, test_y


def make_plots(model, train_x, train_y, test_x, test_y, curr_save_folder, model_name):
  unit_list = ['[GeV]', '[rad]', '[rad]', '[GeV]']
  variable_list = [r'$p_T$', r'$\eta$', r'$\phi$', r'$E$']
  line_style = ['--', '-']
  colors = ['orange', 'c']
  markers = ['*', 's']

  model.to('cpu')

  # Histograms
  idxs = (0, 100000)  # Choose events to compare
  data = torch.tensor(test_x[idxs[0]:idxs[1]].values, dtype = torch.float)
  pred = model(data).detach().numpy()
  pred = np.multiply(pred, train_x.std().values)
  pred = np.add(pred, train_x.mean().values)
  data = np.multiply(data, train_x.std().values)
  data = np.add(data, train_x.mean().values)

  alph = 0.8
  n_bins = 50
  for kk in np.arange(4):
      plt.figure(kk + 4)
      n_hist_data, bin_edges, _ = plt.hist(data[:, kk], color=colors[1], label='Input', alpha=1, bins=n_bins)
      n_hist_pred, _, _ = plt.hist(pred[:, kk], color=colors[0], label='Output', alpha=alph, bins=bin_edges)
      plt.suptitle(train_x.columns[kk])
      plt.xlabel(variable_list[kk] + ' ' + unit_list[kk])
      plt.ylabel('Number of events')
      ms.sciy()
      # plt.yscale('log')
      plt.legend()
      fig_name = model_name + '_hist_%s' % train_x.columns[kk]
      plt.savefig(curr_save_folder + fig_name)


  residual_strings = [r'$(p_{T,out} - p_{T,in}) / p_{T,in}$',
                          r'$(\eta_{out} - \eta_{in}) / \eta_{in}$',
                          r'$(\phi_{out} - \phi_{in}) / \phi_{in}$',
                          r'$(E_{out} - E_{in}) / E_{in}$']
  residuals = (pred - data.detach().numpy()) / data.detach().numpy()
  range = (-.02, .02)
  for kk in np.arange(4):
      plt.figure()
      n_hist_pred, bin_edges, _ = plt.hist(
          residuals[:, kk], label='Residuals', linestyle=line_style[0], alpha=alph, bins=100, range=range)
      plt.suptitle('Residuals of %s' % train_x.columns[kk])
      plt.xlabel(residual_strings[kk])  # (train_x.columns[kk], train_x.columns[kk], train_x.columns[kk]))
      plt.ylabel('Number of jets')
      ms.sciy()
      #plt.yscale('log')
      std = np.std(residuals[:, kk])
      std_err = utils.std_error(residuals[:, kk])
      mean = np.nanmean(residuals[:, kk])
      sem = stats.sem(residuals[:, kk], nan_policy='omit')
      ax = plt.gca()
      plt.text(.75, .8, 'Mean = %f$\pm$%f\n$\sigma$ = %f$\pm$%f' % (mean, sem, std, std_err), bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10},
              horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=18)
      fig_name = model_name + '_residual_%s' % train_x.columns[kk]
      plt.savefig(curr_save_folder + fig_name)

  res_df = pd.DataFrame({'pt': residuals[:, 0], 'eta': residuals[:, 1], 'phi': residuals[:, 2], 'E': residuals[:, 3]})
  save = True

  # Generate a custom diverging colormap
  cmap = sns.diverging_palette(10, 220, as_cmap=True)
  #cmap = 'RdBu'
  norm = mpl.colors.Normalize(vmin=-1, vmax=1, clip=False)
  mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

  group = ['pt', 'eta', 'phi', 'E']

  label_kwargs = {'fontsize': 20}
  title_kwargs = {"fontsize": 11}
  mpl.rcParams['lines.linewidth'] = 1
  mpl.rcParams['xtick.labelsize'] = 12
  mpl.rcParams['ytick.labelsize'] = 12
  group_arr = res_df.values
  corr = res_df.corr()
  qs = np.quantile(group_arr, q=[.0025, .9975], axis=0)
  ndim = qs.shape[1]
  ranges = [tuple(qs[:, kk]) for kk in np.arange(ndim)]
  figure = corner(group_arr, range=ranges, plot_density=True, plot_contours=True, no_fill_contours=False, #range=[range for i in np.arange(ndim)],
                  bins=50, labels=group, label_kwargs=label_kwargs, #truths=[0 for kk in np.arange(qs.shape[1])],
                  show_titles=True, title_kwargs=title_kwargs, quantiles=(0.16, 0.84),
                  # levels=(1 - np.exp(-0.5), .90), fill_contours=False, title_fmt='.2e')
                  levels=(1 - np.exp(-0.5), .90), fill_contours=False, title_fmt='.1e')

  # # Extract the axes
  axes = np.array(figure.axes).reshape((ndim, ndim))
  # Loop over the diagonal
  linecol = 'r'
  linstyl = 'dashed'
  # Loop over the histograms
  for yi in np.arange(ndim):
      for xi in np.arange(yi):
          ax = axes[yi, xi]
          # Set face color according to correlation
          ax.set_facecolor(color=mappable.to_rgba(corr.values[yi, xi]))
  cax = figure.add_axes([.87, .4, .04, 0.55])
  cbar = plt.colorbar(mappable, cax=cax, format='%.1f', ticks=np.arange(-1., 1.1, 0.2))
  cbar.ax.set_ylabel('Correlation', fontsize=20)

  if save:
      fig_name = 'corner_3d.png'
      plt.savefig(curr_save_folder + fig_name)


def train_evaluate_default_model(model, model_name, num_epochs , learning_rate):
  train_dl, valid_dl, train_x, train_y, test_x, test_y = load_data(batch_size = 256)
  db = basic_data.DataBunch(train_dl, valid_dl)

  loss_func = nn.MSELoss() 
  bn_wd = False  # Don't use weight decay fpr batchnorm layers
  true_wd = True  # wd will be used for all optimizers
  wd = 1e-6
  plots_path = 'plots/' + model_name + '/'

  if not os.path.exists(plots_path):
    os.mkdir(plots_path)

  if not os.path.exists('results.csv'):
    results_df = pd.DataFrame(columns=['model', 'num_epochs', 'learning_rate', 'hidden_dim_1', 'hidden_dim_2', 'hidden_dim_3', 'final_loss'])
    results_df.to_csv('results.csv', index = False)

  learn = basic_train.Learner(data=db, model=model, loss_func=loss_func, wd=wd, callback_fns=ActivationStats, bn_wd=bn_wd, true_wd=true_wd)

  learn.fit(num_epochs, lr=learning_rate, wd=wd)
  learn.save(model_name + '__' + str(num_epochs) + '_epochs__' + 'lr_' + str(learning_rate)) 

  loss = learn.validate()[0]

  result = {'model': model_name,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate, 
            'final_loss': loss
            }

  #adding row to results DataFrame 
  results_df = pd.read_csv('results.csv')
  results_df = results_df.append(result, ignore_index=True)
  results_df.to_csv('results.csv', index = False)

  loss_plot = learn.recorder.plot_losses(return_fig=True);
  loss_plot.savefig(plots_path + 'loss_plot.png')

  plt.plot(learn.recorder.val_losses, marker='>');
  plt.savefig(plots_path + 'val_losses.png')

  plot_activations(learn, save = plots_path);

  plt.close('all')

  make_plots(model, train_x, train_y, test_x, test_y, plots_path, model_name)

  plt.close('all')

  return loss


def train_evaluate_model(model, model_name, num_epochs, learning_rate, hidden_dim_1, hidden_dim_2, hidden_dim_3):
  train_dl, valid_dl, train_x, train_y, test_x, test_y = load_data(batch_size = 256)
  db = basic_data.DataBunch(train_dl, valid_dl)

  loss_func = nn.MSELoss() 
  bn_wd = False  # Don't use weight decay fpr batchnorm layers
  true_wd = True  # wd will be used for all optimizers
  wd = 1e-6
  plots_path = 'plots/'+ model_name + '/' + str(hidden_dim_1) + '_' + str(hidden_dim_2) + '_' + str(hidden_dim_3) + '/'

  if not os.path.exists(plots_path):
    if not os.path.exists('plots/'+ model_name + '/'):
      os.mkdir('plots/'+ model_name + '/')
    os.mkdir(plots_path)

  if not os.path.exists('results.csv'):
    results_df = pd.DataFrame(columns=['model', 'num_epochs', 'learning_rate', 'hidden_dim_1', 'hidden_dim_2', 'hidden_dim_3', 'final_loss'])
    results_df.to_csv('results.csv', index = False)

  learn = basic_train.Learner(data=db, model=model, loss_func=loss_func, wd=wd, callback_fns=ActivationStats, bn_wd=bn_wd, true_wd=true_wd)

  learn.fit(num_epochs, lr=learning_rate, wd=wd)
  learn.save(model_name+str(hidden_dim_1) + '_' + str(hidden_dim_2) + '_' + str(hidden_dim_3))

  loss = learn.validate()[0]

  result = {'model': model_name,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate, 
            'hidden_dim_1': hidden_dim_1,
            'hidden_dim_2': hidden_dim_2,
            'hidden_dim_3': hidden_dim_3,
            'final_loss': loss
            }

  results_df = pd.read_csv('results.csv')
  results_df = results_df.append(result, ignore_index=True)
  results_df.to_csv('results.csv', index = False)

  loss_plot = learn.recorder.plot_losses(return_fig=True);
  loss_plot.savefig(plots_path + 'loss_plot.png')

  plt.plot(learn.recorder.val_losses, marker='>');
  plt.savefig(plots_path + 'val_losses.png')

  plot_activations(learn, save = plots_path);

  plt.close('all')

  current_save_folder = plots_path
  make_plots(model, train_x, train_y, test_x, test_y, plots_path, model_name);

  plt.close('all')

  return loss
