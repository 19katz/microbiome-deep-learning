# to use example: python3 all_linear_models_AEB.py -m 'lassoLR_saga' -k 5 -cvg 10 -ng 1

import os
import pandas as pd
import numpy as np
import csv

from itertools import cycle, product
import argparse
import warnings

from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LassoCV
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
date = '031219_NMF_on_all/'
output_dir = os.environ['HOME'] + '/deep_learning_microbiome/analysis/kmers/linear/' + str(date)

# current directory
os.chdir("/pollard/home/abustion/deep_learning_microbiome/scripts/")


# filter out warnings about convergence 
warnings.filterwarnings("ignore", category=ConvergenceWarning)


kmer_size = None
cv_gridsearch = None # number of folds for grid search
cv_testfolds = None # number of folds for actual training of the best model
n_iter = None # number of iterations of cross-validation
random_state = None


# Dictionary of parameters for each model 
param_dict = {       
    "rf": {"n_estimators": [400, 500, 750, 1000],
           "criterion": ["gini"],
           "max_features": ["sqrt", "log2"],
           "max_depth": [None],
           "min_samples_split": [2, 5, 10],
           "n_jobs": [1]},
    "lasso": {"alpha": [np.logspace(-4, -0.5, 50)]}
    "lassoLR_saga": {"penalty": ["l1"], 
                "solver": ["saga"]},
    "lassoLR_liblin": {"penalty": ["l1"], 
                "solver": ["liblinear"]}
    }


# User passes the model to be used as a command-line argument, which is parsed here.
if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description= "Program to run linear machine learning models on kmer datasets")
    parser.add_argument('-m', type = str, default = 'rf', help = "Model type")
    parser.add_argument('-k', type = int, default = 5, help = "Kmer Size")
    parser.add_argument('-cvg', type = int, default = 10, help = "Number CV folds for grid search")
    parser.add_argument('-cvt', type = int, default = 10, help = "Number CV folds for testing")
    parser.add_argument('-ng', type = int, default = 20, help = "Number iterations of k-fold cross validation for grid search")
    parser.add_argument('-nt', type = int, default = 20, help = "Number iterations of k-fold cross validation for testing")
    parser.add_argument('-ds', type = str, default = 'LiverCirrhosis', help = 'Data set')

    arg_vals = parser.parse_args()
    learn_type = arg_vals.m
    kmer_size = arg_vals.k
    cv_gridsearch = arg_vals.cvg
    cv_testfolds = arg_vals.cvt
    n_iter_grid = arg_vals.ng
    n_iter_test = arg_vals.nt
    data_set = arg_vals.ds
    
    #initiate csv file for output
    with open(output_dir + data_set + learn_type + str(kmer_size) + 'mers_summ_table.csv', 'w', newline='') as csvfile:
        fieldnames = ['dataset', 'kmer_size', 'n_splits', 'n_repeats', 'acc', 'auc', 'precision', 'recall', 'f1',
                                  'model', 'NMF_factors', 'params']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        data_set = [data_set]
        allowed_labels = ['0', '1']
        kmer_cnts, accessions, labelz, domain_labels = load_kmer_cnts_jf.load_kmers(kmer_size,
                                                                            data_set,
                                                                            allowed_labels)

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
                grid_search = GridSearchCV(estimator, param_grid, cv = k_fold, n_jobs = 1, scoring=scoring, refit=False)
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
            elif learn_type == "lassoLR_saga":
                k_fold = RepeatedStratifiedKFold(n_splits=cv_gridsearch, n_repeats=n_iter_grid)
                estimator_acc = LogisticRegressionCV(cv = k_fold, 
                                                     scoring= 'accuracy', 
                                                     penalty = 'l1', 
                                                     solver = 'saga', 
                                                     n_jobs = 1).fit(x,y)
                estimator_auc = LogisticRegressionCV(cv = k_fold, 
                                                     scoring= 'roc_auc', 
                                                     penalty = 'l1', 
                                                     solver = 'saga', 
                                                     n_jobs = 1).fit(x,y)
                    
                #scores
                acc = estimator_acc.scores_[1].mean(axis=0).max()
                auc = estimator_auc.scores_[1].mean(axis=0).max()
                    
                #save best models
                writer.writerow({
                        'dataset': str(data_set),
                        'kmer_size': kmer_size,
                        'n_splits': cv_gridsearch,
                        'n_repeats': n_iter_grid,
                        'acc': acc,
                        'auc': auc,
                        'model': learn_type,
                        'NMF_factors': n,
                        'params': 'saga'
                        })
                
            elif learn_type == "lassoLR_liblin":
                k_fold = RepeatedStratifiedKFold(n_splits=cv_gridsearch, n_repeats=n_iter_grid)
                estimator_acc = LogisticRegressionCV(cv = k_fold, 
                                                     scoring= 'accuracy', 
                                                     penalty = 'l1', 
                                                     solver = 'saga', 
                                                     n_jobs = 7).fit(x,y)
                estimator_auc = LogisticRegressionCV(cv = k_fold, 
                                                     scoring= 'roc_auc', 
                                                     penalty = 'l1', 
                                                     solver = 'saga', 
                                                     n_jobs = 7).fit(x,y)
                    
                #scores
                acc = estimator_acc.scores_[1].mean(axis=0).max()
                auc = estimator_auc.scores_[1].mean(axis=0).max()
                    
                #save best models
                writer.writerow({
                            'dataset': str(data_set),
                            'kmer_size': kmer_size,
                            'n_splits': cv_gridsearch,
                            'n_repeats': n_iter_grid,
                            'acc': acc,
                            'auc': auc,
                            'model': learn_type,
                            'NMF_factors': n,
                            'params': 'liblinear'
                            })
                
            elif learn_type == "lasso":
                k_fold = RepeatedStratifiedKFold(n_splits=cv_gridsearch, n_repeats=n_iter_grid)
                estimator = LassoCV(alphas = param_grid["alpha"][0], cv = k_fold, n_jobs = -1)
                accuracies = []
                for train_i, test_i in skf.split(x, y):
                    x_train, x_test = x[train_i], x[test_i]
                    y_train, y_test = y[train_i], y[test_i]
                    y_train = list(map(int, y_train))
                    y_test = list(map(int, y_test))

                    estimator.fit(x_train, y_train)
                
                    accuracy = evaluate(estimator, x_test, y_test)
                    accuracies.append(estimator.get_params())
                    accuracies.append(accuracy)

                with open('/pollard/home/abustion/deep_learning_microbiome/lasso.txt', 'w') as f:
                    for item in accuracies:
                        f.write("%s\n" % item)
