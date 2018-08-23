
###########
EXP CONFIGS
###########

exp_configs.py contains all of the possible parameters for the grid search and models. 

Below are the keys and definitions for each parameter. Each key is only 2 or 3 characters in order to keep file names as short as possible -- in order to save each model's graphs in a different file, the config strings are used as the file name. However, file names are limited to 255 characters, so the keys are kept short. 

auto_epochs_key = 'AEP'		number of epochs for training the autoencoder
super_epochs_key = 'SEP'		number of epochs for supervised learning
batch_size_key = 'BS'			batch size
loss_func_key = 'LF'			loss function for autoencoder
enc_dim_key = 'ED'			number of encoding dimensions (last layer before classification)
enc_act_key = 'EA'			encoding activation function (for all layers before code layer)
code_act_key = 'CA'			code activation function (for last layer before classification)
dec_act_key = 'DA'			decoding activation function (for decoding layers in autoencoder)
out_act_key = 'OA'			output activation function (for output layer in autoencoder)
layers_key = 'LS'			layers up to and including the code layer
batch_norm_key = 'BN'			whether to perform batch normalization
act_reg_key = 'AR'			whether to use activation regularization
norm_input_key = 'NI'			whether to standardize the input to have zero mean and unit std over training samples
early_stop_key = 'ES'			whether to use early stopping
patience_key = 'PA'			patience for early stopping
dataset_key = 'DS'			name of dataset to be run
norm_sample_key = 'NO'			whether to use L1 or L2 normalization
backend_key = 'BE'			backend used for running
use_ae_key = 'AU'			whether to use the autoencoder before supervised learning 
use_kfold_key = 'UK'			number of kfolds
kfold_key = 'KF'			index of current kfold
no_random_key = 'NR'			whether to eliminate randomness
iter_key = 'IT'				index of current iteration
shuffle_labels_key = 'SL'		whether to shuffle labels for supervised learning null
num_iters_key = 'ITS'			number of iterations
shuffle_abunds_key = 'SA'		whether to shuffle abundances for autoencoder null
kmer_size_key = 'KS'			kmer size being used
dropout_pct_key = 'DP'			% of dropout for layers after the input layer (and not including output layer of autoencoder)
input_dropout_pct_key = 'IDP'		% of dropout for the input layer
max_norm_key = 'MN'			max norm, used to limit the magnitude of the network's weights
pca_dim_key = 'PC'			number of principal components for PCA
ae_datasets_key = 'AD'			datasets used to train the autoencoder, with each dataset name's initial representing the dataset
after_ae_act_key = 'AA'		the activation function used when the code layer of the autoencoder is used as input to supervised learning
after_ae_layers_key = 'AL'		layer structure if the code layer of the autoencoder is used as input to supervised learning	
nmf_dim_key = 'NM'			number of components for nmf (0 if no nmf should be used)
version_key = 'V'			version, meant to catch any unnamed parameters
class_weight_key = 'CW'		class weight ratio (diseased:healthy)			

######################
TO RUN SPECIFIC MODELS
######################

In deep_learning_supervised_Katherine_jf.py: 

Change exp_mode (line 1315) to SUPER_MODELS

In "if exp_mode == "SUPER_MODELS":" (line 1317), set configs for each dataset by doing:

dataset_configs[DATASET NAME] = [
					CONFIG STRING 1,
					CONFIG STRING 2, 
					...
					]

The dataset names are defined in dataset_dict in deep_learning_supervised_Katherine_jf.py.

The config strings can be directly copied and pasted from the blog and/or grid search results. When copying from the grid search log, change UK to 10 and ITS to 20 (20 x 10 k-fold cross-validation) to make the stats more reliable.

Run the code: 

nohup [python] deep_learning_supervised_Katherine_jf.py >> [output file] &

To look at the aggregated statistics for the best models, sorted from best to worst by F1 score: 

cat [output file] | [python] process_perf_logs.py | sort -n -r -k 17 | less

All the plots are saved in analysis/kmers.

#########################################
TO RECORD FEATURE IMPORTANCES FOR A MODEL
#########################################

Run the code as follows: 

nohup [python] deep_learning_supervised_Katherine_jf.py -featimps [number of features] >> [output file] &

Number of features should be set to 0 if no features should be dumped and -1 for all features. The default value is -1. 

This also now dumps the feature importances into a file named as follows:

feat_imps_[config].txt

###############################
TO SAVE WEIGHTS FOR AUTOENCODER
###############################

Run the code as follows: 

nohup [python] deep_learning_supervised_Katherine_jf.py -saveweightsauto True >> [output file] &

This will save the weights (per fold per iteration) in a file in the same directory as the code under a name of the following format:

ae_weights_[config].h5

#######################################
TO SAVE WEIGHTS FOR SUPERVISED LEARNING
#######################################

Run the code as follows: 

nohup [python] deep_learning_supervised_Katherine_jf.py -saveweightssuper True >> [output file] &

This will save the weights (per fold per iteration) in a file in the same directory as the code under a name of the following format:

super_weights_[config].h5

###############################
TO LOAD WEIGHTS FOR AUTOENCODER
###############################

Run the code as follows: 

nohup [python] deep_learning_supervised_Katherine_jf.py -autoweightsfile [file name] >> [output file] &

This will load the weights from the given file and use them instead of retraining the autoencoder. 

#######################################
TO LOAD WEIGHTS FOR SUPERVISED LEARNING
#######################################

Run the code as follows: 

nohup [python] deep_learning_supervised_Katherine_jf.py -superweightsfile [file name] >> [output file] &

This will load the weights from the given file and use them instead of retraining the supervised learning models. 

#############################################################
Of course, all the command line options above can be combined.
#############################################################

######################
TO RUN THE GRID SEARCH
######################

In exp_configs.py: 

Modify the list of possible values for each parameter in the exp_configs dictionary (line 69). Since the server I use has 4 GPUs, I usually run the grid search one dataset at a time and evenly distribute the datasets among the GPUs. To do that, each time, I comment out all of the datasets except the one I want to run.

In deep_learning_supervised_Katherine_jf.py: 

Change exp_mode (line 1315) to SEARCH_SUPER_MODELS

Run the code: 

nohup [python] deep_learning_supervised_Katherine_jf.py >> [output file] &

To look at the aggregated statistics for the best models, sorted from best to worst by F1 score: 

cat [output file] | [python] process_perf_logs.py | sort -n -r -k 17 | less

All the plots are saved in analysis/kmers.

######################################
LOGGING THE PER-ITERATION FEATURE IMPS
######################################

In deep_learning_supervised_Katherine_jf.py, I added the parameter logiterfeats, which indicates whether the features should be logged per iteration. 

Run deep_learning_supervised_Katherine_jf.py with the parameter logiterfeats set to True. 

nohup [python] [file to run] -logiterfeats True >> [output file] &

########
VERSIONS
########

The new key -version can be used to manually set a version for the config (for different runs of the same model, for example). 

###################################################
TO COMBINE ROC PLOTS AND CALCULATE T-STATS/P-VALUES
###################################################

Run deep_learning_merge_plots.py and give the names of the pickle files of all the models whose ROC's are to be merged should be in the list of filenames. In order to get the t-statistics and p-values for a certain model, both the real model's pickle file and the corresponding shuffled model's pickle file should be in the list. 

[python] deep_learning_merge_plots.py [file names] >> [output file]

To view the t-statistics and p-values: 

less [output file]

The plots will be generated in analysis/kmers. 
