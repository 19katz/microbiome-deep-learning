# to use example: python3 all_linear_models_AEB.py -m 'lassoLR' -k 5 -cvg 10 -ng 1

import os
import pandas as pd
import numpy as np
import csv

from itertools import cycle, product
import argparse
import warnings

from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, RepeatedStratifiedKFold, train_test_split
from sklearn import cross_validation, metrics
from sklearn.utils import shuffle
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

import matplotlib.pyplot as plt
plt.switch_backend('Agg') # this suppresses the console for plotting  
import seaborn as sns

# import private scripts
import load_kmer_cnts_jf
import stats_utils_AEB


# directories (make sure date exists)
date = '101618_grid_RF/'
output_dir = os.environ['HOME'] + '/deep_learning_microbiome/analysis/kmers/linear/' + str(date)


# filter out warnings about convergence 
warnings.filterwarnings("ignore", category=ConvergenceWarning)


kmer_size = None
cv_gridsearch = None # number of folds for grid search
cv_testfolds = None # number of folds for actual training of the best model
n_iter = None # number of iterations of cross-validation
random_state = None


# Lists of data sets to be tested
# Each item consists of two lists: from the first, the healthy samples will be extracted.
# From the second, diseased samples will be extracted.
# The two sets will then be combined. 
data_sets_to_use = [
#    [['MetaHIT'], ['MetaHIT']],
#    [['Qin_et_al'], ['Qin_et_al']],
#    [['Zeller_2014'], ['Zeller_2014']],
    [['LiverCirrhosis'], ['LiverCirrhosis']],
#    [['Karlsson_2013_no_adapter'], ['Karlsson_2013_no_adapter']],
#    [['RA_no_adapter'], ['RA_no_adapter']],
#    [['LeChatelier'], ['LeChatelier']],
#    [['Feng'], ['Feng']]
   ]


# Dictionary of parameters for each model 
param_dict = {       
    "rf": {"n_estimators": [400, 500, 750, 1000],
           "criterion": ["gini"],
           "max_features": ["sqrt", "log2"],
           "max_depth": [None],
           "min_samples_split": [2, 5, 10],
           "n_jobs": [1]},
    "lassoLR": {"penalty": ["l1"], 
                "solver": ["saga", "liblinear"]}
    }


# User passes the model to be used as a command-line argument, which is parsed here.
#easier to try multiple factors from list, so commenting out -f for now 
if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description= "Program to run linear machine learning models on kmer datasets")
    parser.add_argument('-m', type = str, default = 'rf', help = "Model type")
    parser.add_argument('-k', type = int, default = 5, help = "Kmer Size")
    parser.add_argument('-cvg', type = int, default = 10, help = "Number CV folds for grid search")
    parser.add_argument('-cvt', type = int, default = 10, help = "Number CV folds for testing")
    parser.add_argument('-ng', type = int, default = 20, help = "Number iterations of k-fold cross validation for grid search")
    parser.add_argument('-nt', type = int, default = 20, help = "Number iterations of k-fold cross validation for testing")
    #append allows you to add multiple -f values that get considered as a list
    #parser.add_argument('-f', type = int, action='append', help="NMF factorization number")

    arg_vals = parser.parse_args()
    learn_type = arg_vals.m
    kmer_size = arg_vals.k
    cv_gridsearch = arg_vals.cvg
    cv_testfolds = arg_vals.cvt
    n_iter_grid = arg_vals.ng
    n_iter_test = arg_vals.nt
    #n_factor = arg_vals.f

    #initiate csv file for output
    with open(output_dir + str(kmer_size) + 'mers_summ_table.csv', 'w', newline='') as csvfile:
        fieldnames = ['dataset', 'kmer_size', 'n_splits', 'n_repeats', 'acc', 'auc', 'precision', 'recall', 'f1',
                                  'model', 'NMF_factors', 'params']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Loop over all data sets
        for data_set in data_sets_to_use:
            data_set = data_set[0]
        
            # Retrieve diseased data and labels
            allowed_labels = ['0', '1']
            kmer_cnts, accessions, labelz, domain_labels = load_kmer_cnts_jf.load_kmers(kmer_size, data_set, allowed_labels)
            print("LOADED DATASET " + str(data_set[0]) + ": " + str(len(kmer_cnts)) + " SAMPLES")
            labelz=np.asarray(labelz)
            labelz=labelz.astype(np.int)
        
            # Conduct NMF and resave to data_normalized
            #enter in desired no.factors
            for n in [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                if n == 0:
                    data_normalized = normalize(kmer_cnts, axis = 1, norm = 'l1')
                    data_normalized, labels = shuffle(data_normalized, labelz, random_state=0) 
                    x = data_normalized
                    y = labels
                    
                else:
                    data_normalized = normalize(kmer_cnts, axis = 1, norm = 'l1')
                    data_normalized = stats_utils_AEB.NMF_factor(data_normalized, kmer_size, n_components = int(n), 
                                                     title=(str(data_set) + str(kmer_size) + "mers" 
                                                            + str(n) + "factors"))
                    data_normalized, labels = shuffle(data_normalized, labelz, random_state=0)
                    x = data_normalized
                    y = labels
                
                param_grid = param_dict[learn_type]
    
                scoring = {
                        'Acc': 'accuracy',
                        'AUC': 'roc_auc',
                        'Precision': 'precision',
                        'Recall': 'recall',
                        'F1': 'f1' # weighted average of precision and recall
                    }

                #random forest
                if learn_type == "rf":
                    estimator = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, n_jobs=1)
                    k_fold = RepeatedStratifiedKFold(n_splits=cv_gridsearch, n_repeats=n_iter_grid)
                    grid_search = GridSearchCV(estimator, param_grid, cv = k_fold, n_jobs = 5, scoring=scoring, refit=False)
                    grid_search.fit(x, y)            
                    grid_search_results = grid_search.cv_results_
            
                    #scores
                    aucs = np.array(grid_search_results['mean_test_AUC'])
                    accuracies = np.array(grid_search_results['mean_test_Acc'])
                    f1s = np.array(grid_search_results['mean_test_F1'])
                    precisions = np.array(grid_search_results['mean_test_Precision'])
                    recalls = np.array(grid_search_results['mean_test_Recall'])
                
                    #parameters
                    all_params = np.array(grid_search_results['params'])
                    
                    #save best models
                    for i in range(len(aucs)):
                        param_grid = all_params[i]
                        criterion = param_grid["criterion"]
                        max_depth = param_grid["max_depth"]
                        max_features = param_grid["max_features"]
                        min_samples_split = param_grid["min_samples_split"]
                        n_estimators = param_grid["n_estimators"]
                        n_jobs = 1
                        current_estimator = RandomForestClassifier(criterion=criterion, max_depth=max_depth, 
                                                    max_features=max_features, min_samples_split=min_samples_split,
                                                    n_estimators=n_estimators,n_jobs=n_jobs)
                        writer.writerow({
                                 'dataset': str(data_set),
                                 'kmer_size': kmer_size,
                                 'n_splits': cv_gridsearch,
                                 'n_repeats': n_iter_grid,
                                 'acc': accuracies[i],
                                 'auc': aucs[i],
                                 'precision': precisions[i],
                                 'recall': recalls[i],
                                 'f1': f1s[i],
                                 'model': learn_type,
                                 'NMF_factors': n,
                                 'params': str(all_params[i])
                                })
        
                #LassoLR
                elif learn_type == "lassoLR":
                    estimator = LogisticRegression(penalty = 'l1', solver = 'saga', n_jobs = 1)
                    k_fold = RepeatedStratifiedKFold(n_splits=cv_gridsearch, n_repeats=n_iter_grid)
                    grid_search = GridSearchCV(estimator, param_grid, cv = k_fold, n_jobs = 5, 
                                                   scoring=scoring, refit = False)
                    grid_search.fit(x, y)
                    grid_search_results = grid_search.cv_results_
                
                    #scores
                    aucs = np.array(grid_search_results['mean_test_AUC'])
                    accuracies = np.array(grid_search_results['mean_test_Acc'])
                    f1s = np.array(grid_search_results['mean_test_F1'])
                    precisions = np.array(grid_search_results['mean_test_Precision'])
                    recalls = np.array(grid_search_results['mean_test_Recall'])
                    
                    #parameters
                    all_params = np.array(grid_search_results['params'])
                
                    #save best models
                    for i in range(len(aucs)):
                        param_grid = all_params[i]
                        n_jobs = 1
                        solver = param_grid["solver"]
                        penalty = param_grid["penalty"]
                        current_estimator = LogisticRegression(solver=solver, penalty=penalty, n_jobs=n_jobs)
        
                        #spit into csv
                        writer.writerow({
                                'dataset': str(data_set),
                                'kmer_size': kmer_size,
                                'n_splits': cv_gridsearch,
                                'n_repeats': n_iter_grid,
                                'acc': accuracies[i],
                                'auc': aucs[i],
                                'precision': precisions[i],
                                'recall': recalls[i],
                                'f1': f1s[i],
                                'model': learn_type,
                                'NMF_factors': n,
                                'params': str(all_params[i])
                            })
