import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from itertools import cycle, product
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn import cross_validation, metrics

import argparse

import matplotlib.pyplot as plt
plt.switch_backend('Agg') # this suppresses the console for plotting  
import seaborn as sns

import load_kmer_cnts_jf
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import shuffle
from sklearn.decomposition import NMF
import _search
import _validation
import load_kmer_cnts_pasolli_jf

graph_dir = os.environ['HOME'] + '/deep_learning_microbiome/analysis/kmers/linear/'

# filter out warnings about convergence 
warnings.filterwarnings("ignore", category=ConvergenceWarning)

kmer_size = None
# number of folds for grid search
cv_gridsearch = None
# number of folds for actual training of the best model
cv_testfolds = None
# number of iterations of cross-validation
n_iter = None
random_state = None

# Lists of data sets to be tested
# Each item consists of two lists: from the first, the healthy samples will be extracted.
# From the second, diseased samples will be extracted.
# The two sets will then be combined. 
data_sets_to_use = [
    #[['MetaHIT'], ['MetaHIT']],
    [['Qin_et_al'], ['Qin_et_al']],
    [['Zeller_2014'], ['Zeller_2014']],
    [['LiverCirrhosis'], ['LiverCirrhosis']],
    #[['Karlsson_2013'], ['Karlsson_2013']],
    #[['RA'], ['RA']],
    [['Feng'], ['Feng']],
    #[['Karlsson_2013', 'Qin_et_al'], ['Karlsson_2013', 'Qin_et_al']],
    #[['Feng', 'Zeller_2014'],['Feng', 'Zeller_2014']]
    ]


# Dictionary of parameters for each model
# Values based on the values used in the
# Pasolli paper 
param_dict = {
    "svm": [ {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
             {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}],
    
    "rf": {"n_estimators": [500],
           "criterion": ["gini"],
           "max_features": ["sqrt"],
           "max_depth": [None],
           "min_samples_split": [2],
           "n_jobs": [1]
            },
    
    "lasso": {"alpha": [np.logspace(-4, -0.5, 50)]},

    "lassoLR": {"penalty": ["l1"], 
                "solver": ["saga"]},
    
    "enet": {"alpha": [np.logspace(-4, -0.5, 50)],
             "l1": [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]}                              
    }



# Uses the model to predict labels given the test features
# and compares them to the labels by calculating accuracy and error
# This is used by Lasso and Elastic Net
def evaluate(model, test_features, test_labels):
    predictions = np.array(model.predict(test_features))
    # Convert the predicted values to 0 or 1
    for r in range(len(predictions)):
        if (predictions[r] > 0.5):
            predictions[r] = 1
        else:
            predictions[r] = 0
            
    # Calculates error and accuracy
    test_labels = np.array(test_labels)
    errors = abs(predictions - test_labels)
    total_error = np.sum(errors)
    
    mape = total_error / len(test_labels)
    accuracy = 1 - mape
    return accuracy

# This reference explains some of the things I'm doing here
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
if __name__ == '__main__':
    # User passes the model to be used as a command-line argument, which is parsed here.
    # The default model is Random Forest 
    parser = argparse.ArgumentParser(description= "Program to run linear machine learning models on kmer datasets")
    parser.add_argument('-m', type = str, default = 'rf', help = "Model type")
    parser.add_argument('-k', type = int, default = 5, help = "Kmer Size")
    parser.add_argument('-cvg', type = int, default = 10, help = "Number of CV folds for grid search")
    parser.add_argument('-cvt', type = int, default = 10, help = "Number of CV folds for testing")
    parser.add_argument('-ng', type = int, default = 20, help = "Number of iterations of k-fold cross validation for grid search")
    parser.add_argument('-nt', type = int, default = 20, help = "Number of iterations of k-fold cross validation for testing")
    parser.add_argument('-nrf', type = bool, default = True, help = "Whether to use normalization on the random forest")


    arg_vals = parser.parse_args()
    learn_type = arg_vals.m
    kmer_size = arg_vals.k
    cv_gridsearch = arg_vals.cvg
    cv_testfolds = arg_vals.cvt
    n_iter_grid = arg_vals.ng
    n_iter_test = arg_vals.nt
    norm_for_rf = arg_vals.nrf
    # Loop over all data sets

    
    for data_set in data_sets_to_use:
        data_set = data_set[0]
        kmer_dir = os.environ['HOME'] + '/deep_learning_microbiome/data/' + str(kmer_size) + 'mers_jf/'
        
        # Retrieve diseased data and labels
        allowed_labels = ['0', '1']
        kmer_cnts, accessions, labelz, domain_labels = load_kmer_cnts_pasolli_jf.load_kmers(kmer_size, data_set, allowed_labels)
        print("LOADED DATASET " + str(data_set[0]) + ": " + str(len(kmer_cnts)) + " SAMPLES")
        labelz=np.asarray(labelz)
        labelz=labelz.astype(np.int)

        n_comp = [80]
        for n in n_comp:
            if n == 0:
                data_normalized = normalize(kmer_cnts, axis = 1, norm = 'l1')
                data_normalized, labels = shuffle(data_normalized, labelz, random_state=0) 
                x = data_normalized
                y = labels
            else:
                data_normalized = normalize(kmer_cnts, axis = 1, norm = 'l1')
                data_normalized, labels = shuffle(data_normalized, labelz, random_state=0)
                V = data_normalized.T
                model = NMF(n_components = n, init='random', random_state=0, solver = 'mu', beta_loss = 'frobenius', max_iter = 1000)
                W = model.fit_transform(V)
                H = model.components_
                data_normalized = H.T
                data_normalized, labels = shuffle(data_normalized, labels, random_state=0)
                x = data_normalized
                y = labels

                W_all = pd.DataFrame(W)
                W_all['Features'] = pd.read_csv(kmer_dir + str(kmer_size) + "mer_dictionary.gz", compression='gzip', header=None)
                meltedW = pd.melt(W_all, id_vars = "Features", var_name='Signature (i.e. Factor)', value_name='Weight')
                sns.set(style="white")
                g = sns.FacetGrid(meltedW, row = 'Signature (i.e. Factor)', sharey = True)
                g.map(sns.barplot, 'Features', 'Weight', color="blue", alpha = 0.7)
                g.set(xticklabels=[])
                #plt.xticks(rotation=90)
                plt.savefig(graph_dir + "NMF" + str(n) + "kmerProfile_" + str(data_set) + str(kmer_size) + "mers.png")
            
            param_grid = param_dict[learn_type]
            
            if learn_type == "svm" or learn_type == "rf":
                if (learn_type == "svm"):
                    estimator = SVC(C = 1, probability = True)
                else:
                    estimator = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, n_jobs=-1)
                k_fold = RepeatedStratifiedKFold(n_splits=cv_gridsearch, n_repeats=n_iter_grid)
                if learn_type == "rf" and not norm_for_rf:
                    grid_search = GridSearchCV(estimator, param_grid, cv = k_fold, n_jobs = -1)
                else:
                    grid_search = _search.GridSearchCV(estimator, param_grid, cv = k_fold, n_jobs = 1)
                grid_search.fit(x, y)
                
                grid_search_results = grid_search.cv_results_
                rank = np.array(grid_search_results['rank_test_score'])
                accuracies = np.array(grid_search_results['mean_test_score'])
                all_params = np.array(grid_search_results['params'])

                sort_idx = np.argsort(rank)
                rank = rank[sort_idx]
                accuracies = accuracies[sort_idx]
                all_params = all_params[sort_idx]

                for i in range(len(rank)):
                    param_grid = all_params[i]
                    current_estimator = None
                    if (learn_type == "svm"):
                        C = param_grid["C"]
                        kernel = param_grid["kernel"]
                        if not kernel == "linear":
                            gamma = param_grid["gamma"]
                            current_estimator = SVC(C = C, gamma = gamma, kernel = kernel, probability = True)
                        else:
                            current_estimator = SVC(C = C, kernel = kernel, probability = True)
                    else:
                        criterion = param_grid["criterion"]
                        max_depth = param_grid["max_depth"]
                        max_features = param_grid["max_features"]
                        min_samples_split = param_grid["min_samples_split"]
                        n_estimators = param_grid["n_estimators"]
                        n_jobs = -1
                        current_estimator = RandomForestClassifier(criterion=criterion, max_depth=max_depth, max_features=max_features,
                                                                   min_samples_split=min_samples_split, n_estimators=n_estimators, n_jobs=n_jobs)

                    normalized = " with normalization"
                    if learn_type == "rf" and not norm_for_rf:
                        normalizd = " without normalization"
                    print( str(accuracies[i]) + "(acc) produced by params for samples from " + str(data_set) +
                  " with model " + learn_type + normalized + ", and NMF factorization of: "+ str(n) + " and kmer size " + str(kmer_size)
                  + ": " + str(all_params[i]))
                '''
                if learn_type == "rf" and not norm_for_rf:
                    cross_val = cross_val_score(current_estimator, x, y, cv = RepeatedStratifiedKFold(n_splits = cv_testfolds, n_repeats = n_iter_test))
                else:
                    cross_val = _validation.cross_val_score(current_estimator, x, y, cv = RepeatedStratifiedKFold(n_splits = cv_testfolds, n_repeats = n_iter_test))
                print(str(np.mean(cross_val)) + "\tAggregated cross validation accuracy for healthy samples from " + str(data_sets_healthy) +
                          " and diseased samples from " + str(data_sets_diseased) + 
                          " with model " + learn_type + " and kmer size " + str(kmer_size) + " with params " + str(all_params[i]))
                '''

            elif learn_type == "lassoLR":
                accuracies = []
                estimator = LogisticRegression(penalty = 'l1', solver = 'saga', n_jobs = -1)
                k_fold = RepeatedStratifiedKFold(n_splits=cv_gridsearch, n_repeats=n_iter_grid)
                grid_search = _search.GridSearchCV(estimator, param_grid, cv = k_fold, n_jobs = 2)
                grid_search.fit(x, y)
                grid_search_results = grid_search.cv_results_
                rank = np.array(grid_search_results['rank_test_score'])
                accuracies = np.array(grid_search_results['mean_test_score'])
                all_params = np.array(grid_search_results['params'])
                for i in range(len(rank)):
                    param_grid = all_params[i]
                    n_jobs = -1
                    solver = param_grid["solver"]
                    penalty = param_grid["penalty"]
                    current_estimator = LogisticRegression(solver=solver, penalty=penalty, n_jobs=n_jobs)
                    print( str(accuracies[i]) + "(acc) produced by params for samples from " + str(data_set) +
                  " with model " + learn_type + ", and NMF factorization of: "+ str(n) + " and kmer size " + str(kmer_size)
                  + ": " + str(all_params[i]))

            elif learn_type == "enet" or learn_type == "lasso":
                accuracies = []
                if (learn_type == "enet"):
                    estimator = ElasticNetCV(alphas = param_grid["alpha"][0], l1_ratio = param_grid["l1"], cv = cv_gridsearch,
                                         n_jobs = -1)
                else:
                    estimator = LassoCV(alphas = param_grid["alpha"][0], cv = cv_gridsearch,
                                    n_jobs = -1)
                skf = RepeatedStratifiedKFold(n_splits = cv_testfolds, n_repeats = n_iter_test)
                for train_i, test_i in skf.split(x, y):
                    x_train, x_test = x[train_i], x[test_i]
                    y_train, y_test = y[train_i], y[test_i]
                    y_train = list(map(int, y_train))
                    y_test = list(map(int, y_test))

                    estimator.fit(x_train, y_train)
                
                    accuracy = evaluate(estimator, x_test, y_test)
                    accuracies.append(accuracy)
                    print("Best params for samples from " + str(data_set) +                                                                                                              
                          " with model " + learn_type + " and kmer size " + str(kmer_size) +
                          ": " + str(estimator.get_params()) + " produces "+
                          " accuracy of " + str(accuracy)) 
'''
                    print( str(accuracies[i]) + "(acc) produced by params for samples from " + str(data_set) +
                  " with model " + learn_type + normalized + ", and NMF factorization of: "+ str(n) + " and kmer size " + str(kmer_size)
                  + ": " + str(all_params[i]))
'''                    
'''
                    print("Best params for samples from " + str(data_set) +
                          " with model " + learn_type + " and kmer size " + str(kmer_size) + 
                          ": " + str(estimator.get_params()) + " produces "+ 
                          " accuracy of " + str(accuracy))
'''
'''
                    print("Aggregated cross validation accuracies for healthy samples from " + str(data_sets_healthy) +
                      " and diseased samples from " + str(data_sets_diseased) + 
                      " with model " + learn_type + " and kmer size " + str(kmer_size) + ": " + str(np.mean(accuracies)) + 
                      " with standard deviation " +  str(np.std(accuracies)))
                    '''

            
