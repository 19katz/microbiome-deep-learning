# always run miniconda for keras:
# ./miniconda3/bin/python

import matplotlib
matplotlib.use('Agg') # this suppresses the console for plotting
import matplotlib.pyplot as plt
import bz2
import numpy as np
from numpy import random
import pandas as pd
import os
import pylab
from importlib import reload
from sklearn.preprocessing import normalize
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import History, TensorBoard
from keras import backend as K
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle
from scipy import interp
import argparse

backend = K.backend()

# import our private scripts
import load_kmer_cnts_jf
import deep_learning_models
import plotting_utils
import stats_utils
import config_file
import importlib
import imp
import os

# read in directories:
data_directory = config_file.data_directory
analysis_directory = config_file.analysis_directory
scripts_directory = config_file.scripts_directory


def run_model(data_set, kmer_size, norm_input, encoding_dim, encoded_activation, input_dropout_pct, dropout_pct, num_epochs, batch_size, n_splits, n_repeats, compute_informative_features, plot_iteration, graph_dir, outFile):
    
    # format strings for outputting the paramters associated with this run:
    summary_string, plotting_string= stats_utils.format_input_parameters_printing(data_set, kmer_size, norm_input, encoding_dim, encoded_activation,input_dropout_pct,dropout_pct,num_epochs,batch_size,n_splits,n_repeats,compute_informative_features,plot_iteration)

    #################
    # Load the data # 
    #################
    print('Loading data...')

    data_normalized, labels, rskf = load_kmer_cnts_jf.load_single_disease(data_set, kmer_size, n_splits, n_repeats, precomputed_kfolds=False)

    # rskf = repeated stratified k fold. This contains all the kfold-by-iteration combos. 


    ###################################################
    # iterate through the data kfolds and iterations #
    ###################################################

    # Create a dictionary to store the metrics of each fold 
    aggregated_statistics={} # key=n_repeat, values= dictionary with stats

    for n_repeat in range(0,len(rskf[0])):
        
        print('Iteration %s...' %n_repeat)
        
        aggregated_statistics[n_repeat] = {}
        
        train_idx = rskf[0][n_repeat]
        test_idx = rskf[1][n_repeat]
        x_train, y_train = data_normalized[train_idx], labels[train_idx]
        x_test, y_test = data_normalized[test_idx], labels[test_idx]
    
        #standardize the data, mean=0, std=1
        if norm_input:
            x_train, x_test= stats_utils.standardize_data(x_train, x_test)
    
        ###########################################
        # set up a model (supervised learning)    #
        ###########################################
        # note that the model has to be instantiated each time a new fold is started otherwise the weights will not start from scratch. 
    
        input_dim=len(data_normalized[0]) # this is the number of input kmers

        model=deep_learning_models.create_supervised_model(input_dim, encoding_dim, encoded_activation,input_dropout_pct, dropout_pct)
    
        #weightFile = os.environ['HOME'] + '/deep_learning_microbiome/data/weights.txt'
       
        ##################################################
        # Fit the model with the train data of this fold #
        ##################################################
        history = History()
        # history is a dictionary. To get the keys, type print(history.history.keys())
        
        model.fit(x_train, y_train, 
                  epochs=num_epochs, 
                  batch_size=batch_size, 
                  shuffle=True,
                  validation_data=(x_test, y_test),
                  verbose=0,
                  callbacks=[history])
    
        # predict using the held out data
        y_pred=model.predict(x_test)
        
        # save the weights of this model. TODO 
    
        ################################################################
        # Compute summary statistics                                   #
        ################################################################
        # Store the results of this fold in aggregated_statistics
        aggregated_statistics = stats_utils.compute_summary_statistics(y_test, y_pred, history, aggregated_statistics, n_repeat)

        # could  plot everything (roc, accuracy vs epoch, loss vs epoch, confusion matrix, precision recall) for each fold, but this will produce a lot of graphs. 
        if compute_informative_features:
            shap_values, shap_values_summed = stats_utils.compute_shap_values_deeplearning(input_dim, model, x_test)
            aggregated_statistics[n_repeat]['shap_values_summed']=shap_values_summed
            aggregated_statistics[n_repeat]['shap_values']=shap_values

        # also plot:
        #shap.summary_plot(shap_values, X, plot_type="bar")
        #shap.summary_plot(shap_values, X)

    ##############################################
    # aggregate the results from all the k-folds #
    # Print and Plot                             #
    ##############################################
    print('Aggregating statistics across iterations and printing/plotting...')

    stats_utils.aggregate_statistics_across_folds(aggregated_statistics, rskf, n_splits, outFile, summary_string, plotting_string)


    ###################
    # Aggregate shap: #
    ###################

    if compute_informative_features: 
        print('Computing informative features with Shap...')
        stats_utils.aggregate_shap(aggregated_statistics, rskf)


    #####################################
    # TSNE visualization                #
    # Annamarie                         #
    # find the weights of the best fold #
    #####################################




##############################
# parser for the config dict #
##############################
def parse_config_and_run(config_dict, outFile):
    data_sets=config_dict['data_set']
    kmer_sizes=config_dict['kmer_size']
    norm_inputs=config_dict['norm_input']
    encoding_dims=config_dict['encoding_dim']
    encoded_activations=config_dict['encoded_activation']
    input_dropout_pcts=config_dict['input_dropout_pct']
    dropout_pcts=config_dict['dropout_pct'] 
    num_epochss=config_dict['num_epochs']
    batch_sizes=config_dict['batch_size']
    n_splitss=config_dict['n_splits']
    n_repeatss=config_dict['n_repeats']
    compute_informative_featuress=config_dict['compute_informative_features']
    plot_iterations=config_dict['plot_iteration'] 
    graph_dirs=config_dict['graph_dir'] 

    for data_set in data_sets:
        for kmer_size in kmer_sizes:
            for norm_input in norm_inputs:
                for encoding_dim in encoding_dims:
                    for encoded_activation in encoded_activations:
                        for input_dropout_pct in input_dropout_pcts:
                            for dropout_pct in dropout_pcts:
                                for num_epochs in num_epochss:
                                    for batch_size in batch_sizes:
                                        for n_splits in n_splitss:
                                            for n_repeats in n_repeatss:
                                                for compute_informative_features in compute_informative_featuress:
                                                    for plot_iteration in plot_iterations:
                                                        for graph_dir in graph_dirs:
                                                        
                                                            run_model(data_set, 
                                                                      kmer_size,
                                                                      norm_input,
                                                                      encoding_dim,
                                                                      encoded_activation,
                                                                      input_dropout_pct,
                                                                      dropout_pct,
                                                                      num_epochs,
                                                                      batch_size,
                                                                      n_splits,
                                                                      n_repeats,
                                                                      compute_informative_features,
                                                                      plot_iteration,
                                                                      graph_dir, 
                                                                      outFile)


                                                            
###########
# Main    #
###########

if __name__ == '__main__':
    
    # read in command-line arguments
    parser = argparse.ArgumentParser(description= "Program to run deep learning models on kmer datasets")
    parser.add_argument('-outFile', type = str, default = 'summary_statistics.txt', help = "OutFile for saving summary statistics")
    parser.add_argument('-configFile', type = str, default = 'none', help = "Config file for running the code")

    arg_vals = parser.parse_args()
    outFile = arg_vals.outFile
    configFile=arg_vals.configFile

    # Parse the config file and run the code!
    if configFile=='none':
        config_dict=config_file.config
    else:

        with open(configFile, 'rb') as fp:
            config_file = imp.load_module(
                configFile, fp, configFile,
                ('.py', 'rb', imp.PY_SOURCE)
            )

        config_dict=config_file.config
            

    parse_config_and_run(config_dict, outFile)
        
    
