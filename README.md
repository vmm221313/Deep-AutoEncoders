# Deep-AutoEncoders

This repository contains my implementation of Deep Autoencoders for ATLAS data compression. Some of the code is based on code from this repository - https://github.com/Skelpdar/HEPAutoencoders. The Variational Autoencoder code is based on the paper 'Dimensionality Reduction of SDSS Spectra with Variational Autoencoders' by Portillo, Parejko, Vergara, and Connolly (2020). Many thanks to @Skelpdar, Portillo, Parejko, Vergara, and Connolly for their work.  

### Requirements - 
Install the following 


### Pretrained versions of the following models are available in the 'models' directory.

models = [AE_3D_100, AE_3D_200, AE_3D_small, AE_3D_small_v2, AE_big, AE_big_no_last_bias, AE_3D_50, 
          AE_3D_50_no_last_bias, AE_3D_50cone, AE_3D_500cone_bn, AE_big_2D_v1, AE_big_2D_v2, 
          AE_2D, AE_2D_v2, AE_big_2D_v3, AE_2D_v3, AE_2D_v4, AE_2D_v5, AE_2D_v100, AE_2D_v50,
          AE_2D_v1000]
          
dropout_models = [AE_3D_50_bn_drop, AE_3D_50cone_bn_drop, AE_3D_100_bn_drop, AE_3D_100cone_bn_drop, 
                  AE_3D_200_bn_drop]

###### Of these models, AE_3D_200 performs best for 4D compression
The plots evaluating the performance of all models can be found in the 'plots' directory.

### Scripts - (also find IPython notebooks with the same name that perform the same function but can be run locally as a Jupyter Notebook on Google Colab)

For running the IPython notebooks on Google Colab, simply clone this repository onto a folder and change the filepath. For running the Python scripts, clone the repository and run the scripts from within the directory. 

##### utils.py
Contains util functions for loading/normalizing data.

##### nn_utils.py
Contains the model classes for each of the standard models.

##### my_utils.py
Contains new util functions I have defined to make model training and graph plotting more streamlined. 
These functions outsource the bulk of the plotting and training code present in the notebooks to separate Python scripts to make the notebooks cleaner and more understandable. 

###### Functions in my_utils.py
1. load_data - loads 4D raw data and returns PyTorch Dataloaders and TensorDatasets. Will add support for loading any file with given filepath. Called internally by train_evaluate_default_models.

2. make_plots - plots the graphs comparing the input and output signals, the residuals and the correlation between the latent space vectors. Called internally by train_evaluate_default_models.

3. train_evaluate_default_model - function for training, evaluating and plotting graphs for any model in nn_utils. 
	parameters - model - instance of model class,
    		 	 model_name - name to save plots and weights,
    		 	 num_epochs,
    		 	 learning_rate

4. train_evaluate_model - function for training, evaluating and plotting graphs for any model general model (including custom models)
	parameters - model - instance of model class,
    		 	 model_name - name to save plots and weights,
    		 	 num_epochs,
    		 	 learning_rate
    		 	 hidden_dim_1, hidden_dim_2, hidden_dim_3

##### tune_model.py
Performs Bayesian Hyperparameter optimization using the hyperopt library to optimize the hidden dimension sizes of the Autoencoder

##### relu_vs_tanh.py
Script to compare ReLU and Tanh activation functions

##### train_best_model.py
Script to train the best model and produce plots

##### vae.py
Script to train Variational Autoencoders



