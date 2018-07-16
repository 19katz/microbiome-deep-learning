import os
import numpy as np
from sklearn.preprocessing import normalize

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split

import argparse

import load_kmer_cnts_jf
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import shuffle
import _search


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
    [['Qin_et_al'], ['Qin_et_al']],
    [['MetaHIT'], ['MetaHIT']],
    [['RA'], ['RA']],
    [['Feng'], ['Feng']],
    [['Zeller_2014'], ['Zeller_2014']],
    [['LiverCirrhosis'], ['LiverCirrhosis']],
    [['Karlsson_2013'], ['Karlsson_2013']]
    ]


# Dictionary of parameters for each model
# Values based on the values used in the
# Pasolli paper 
param_dict = {
    "svm": [ {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
             {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}],
    
    "rf": {"n_estimators": [100, 200, 400, 500],
            "criterion": ["gini"],
            # "max_features": ["auto", "sqrt", "log2", None],
            "max_features": ["sqrt"],
            "max_depth": [None],
            "min_samples_split": [2],
            "n_jobs": [1]
            },
    
    "lasso": {"alpha": [np.logspace(-4, -0.5, 50)]},
    
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
    parser.add_argument('-k', type = int, default = 5, help = "Kmer size")
    parser.add_argument('-cvg', type = int, default = 9, help = "Number of CV folds for grid search")
    parser.add_argument('-cvt', type = int, default = 10, help = "Number of CV folds for testing")
    parser.add_argument('-ng', type = int, default = 10, help = "Number of iterations of k-fold cross validation for grid search")
    parser.add_argument('-nt', type = int, default = 20, help = "Number of iterations of k-fold cross validation for testing")


    arg_vals = parser.parse_args()
    learn_type = arg_vals.m
    kmer_size = arg_vals.k
    cv_gridsearch = arg_vals.cvg
    cv_testfolds = arg_vals.cvt
    n_iter_grid = arg_vals.ng
    n_iter_test = arg_vals.nt
    # Loop over all data sets
    for data_set in data_sets_to_use:
        data_sets_healthy=data_set[0]
        data_sets_diseased=data_set[1]

        # Retrieve healthy data and labels
        allowed_labels=['0']
        kmer_cnts_healthy, accessions_healthy, labels_healthy, domain_labels = load_kmer_cnts_jf.load_kmers(kmer_size, data_sets_healthy, allowed_labels)

        # Retrieve diseased data and labels
        allowed_labels=['1']
        kmer_cnts_diseased, accessions_diseased, labels_diseased, domain_labels_diseased = load_kmer_cnts_jf.load_kmers(kmer_size,data_sets_diseased, allowed_labels)

        # concatenate with healthy
        kmer_cnts=np.concatenate((kmer_cnts_healthy,kmer_cnts_diseased))
        accessions=np.concatenate((accessions_healthy,accessions_diseased))
        labels=np.concatenate((labels_healthy,labels_diseased))
        #domain_labels=np.concatenate((domain_labels, domain_labels_diseased))

        labels=np.asarray(labels)
        labels=labels.astype(np.int)

        # Normalize and shuffle the data
        kmer_cnts = normalize(kmer_cnts, axis = 1, norm = 'l1')
        kmer_cnts, labels = shuffle(kmer_cnts, labels)


        # Set up data and labels
        x = kmer_cnts
        y = labels

        param_grid = param_dict[learn_type]
        
                
        # For SVM and Random forest, use GridSearchCV
        # and cross_val_score to do a nested cross-validation
        if learn_type == "svm" or learn_type == "rf":
            # Set the estimator based on the model type
            if (learn_type == "svm"):
                estimator = SVC(C = 1, probability = True)
            else:
                estimator = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, n_jobs=-1)
            k_fold = RepeatedStratifiedKFold(n_splits=cv_gridsearch, n_repeats=n_iter_grid)
            grid_search = _search.GridSearchCV(estimator, param_grid, cv = k_fold, n_jobs = -1)
            grid_search.fit(x, y)
            
            best_grid = grid_search.best_estimator_
            
            print("Best params for healthy samples from " + str(data_sets_healthy) +
                  " and diseased samples from " + str(data_sets_diseased) + 
                  " with model " + learn_type 
                  + ": " + str(grid_search.best_params_) + " produces "
                  + " best score of " + str(grid_search.best_score_))
            cross_val = cross_val_score(best_grid, x, y, cv = RepeatedStratifiedKFold(n_splits = cv_testfolds, n_repeats = n_iter_test))
             
            print("Aggregated cross validation accuracies for healthy samples from " + str(data_sets_healthy) +
                      " and diseased samples from " + str(data_sets_diseased) + 
                      " with model " + learn_type + ": " + str(np.mean(cross_val)))

        

        # For Elastic Net and Lasso, do a stratified k-fold cross validation
        # For each test fold, fit the estimator to the training data
        # and evaluate on the test data
        # This essentially performs the nested cross-validation as well. 
        elif learn_type == "enet" or learn_type == "lasso":
            accuracies = []
            # Set the estimator based on the model type
            if (learn_type == "enet"):
                # doing a separate grid search using stratified k fold -- k - 1 folds should be used
                # for training/grid search, the last fold should be used for test
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

                print("Best params for healthy samples from " + str(data_sets_healthy) +
                  " and diseased samples from " + str(data_sets_diseased) + 
                  " with model " + learn_type 
                  + ": " + str(estimator.get_params()) + " produces "
                  + " accuracy of " + str(accuracy))
            print("Aggregated cross validation accuracies for healthy samples from " + str(data_sets_healthy) +
                      " and diseased samples from " + str(data_sets_diseased) + 
                      " with model " + learn_type + ": " + str(np.mean(accuracies)) + 
                      " with standard deviation " +  str(np.std(accuracies)))
            