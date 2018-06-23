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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
import matplotlib
matplotlib.use('Agg') # this suppresses the console for plotting
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import cycle, product
from sklearn.utils import shuffle

import pickle


import argparse
import load_kmer_cnts_jf
import warnings
import pylab
from sklearn.exceptions import ConvergenceWarning

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

# Turn off per-fold plotting when doing the grid search - otherwise, you'll get tons of plots.
plot_fold = False

# Per-iteration plotting
plot_iter = False
# Overall plotting - aggregate results across both folds and iteration
plot_overall = True



# controls transparency of the CI (Confidence Interval) band around the ROCs as in Pasolli
plot_alpha = 0.2

# factor for calculating band around the overall ROC with per fold ROCs (see Pasolli).
plot_factor = 2.26

class_name = "Disease"

graph_dir = os.environ['HOME'] + '/deep_learning_microbiome/analysis/kmers'


plot_title_size = 12
plot_text_size = 10

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

dataset_config_iter_fold_results = {}

# For indicating which models to run
# ["svm", "rf", "lasso", "enet"]

# Dictionary of parameters for each model
# Values based on the values used in the
# Pasolli paper 
dataset_model_grid = {
    "Qin": "rf1_norm",
    "MetaHIT": "rf2_norm",
    "Feng": "rf3_norm",
    "RA": "rf4_norm",
    "Zeller": "rf5_norm",
    "LiverCirrhosis": "rf6_norm",
    "Karlsson": "rf7_norm",
    }
model_param_grid = {
    "rf1": {'data': [["Qin_et_al"],["Qin_et_al"]],'k': 5,'cvt': 10,'n': 20,'m': "rf",'classes': [0, 1],
            'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 400, 'n_jobs': 1},
    "rf2": {'data': [["MetaHIT"],["MetaHIT"]],'k': 5,'cvt': 10,'n': 20,'m': "rf",'classes': [0, 1],
            'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 200, 'n_jobs': 1},
    "rf3": {'data': [["Feng"],["Feng"]],'k': 5,'cvt': 10,'n': 20,'m': "rf",'classes': [0, 1],
            'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 400, 'n_jobs': 1},
    "rf4": {'data': [["RA"],["RA"]],'k': 5,'cvt': 10,'n': 20,'m': "rf",'classes': [0, 1],
            'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 400, 'n_jobs': 1},
    "rf5": {'data': [["Zeller_2014"],["Zeller_2014"]],'k': 5,'cvt': 10,'n': 20,'m': "rf",'classes': [0, 1],
            'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 500, 'n_jobs': 1},
    "rf6": {'data': [["LiverCirrhosis"],["LiverCirrhosis"]],'k': 5,'cvt': 10,'n': 20,'m': "rf",'classes': [0, 1],
            'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 400, 'n_jobs': 1},
    "rf7": {'data': [["Karlsson_2013"],["Karlsson_2013"]],'k': 5,'cvt': 10,'n': 20,'m': "rf",'classes': [0, 1],
            'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 100, 'n_jobs': 1},
    "svm1": {'data': [["Qin_et_al"],["Qin_et_al"]],'k': 5, 'cvt': 10,'n': 2,'m': "svm",'classes': [0, 1],
             'C': 1, 'kernel': 'linear', 'gamma': 'auto'},
    "svm2": {'data': [["MetaHIT"],["MetaHIT"]],'k': 5,'cvt': 10,'n': 20,'m': "svm",'classes': [0, 1],
             'C': 1, 'kernel': 'linear', 'gamma': 'auto'},
    "svm3": {'data': [["Feng"],["Feng"]],'k': 5,'cvt': 10,'n': 20,'m': "svm",'classes': [0, 1],
             'C': 1, 'kernel': 'linear', 'gamma': 'auto'},
    "svm4": {'data': [["RA"],["RA"]],'k': 5,'cvt': 10,'n': 20,'m': "svm",'classes': [0, 1],
             'C': 1, 'gamma': 0.001, 'kernel': 'rbf'},
    "svm5": {'data': [["Zeller_2014"],["Zeller_2014"]],'k': 5,'cvt': 10,'n': 20,'m': "svm",'classes': [0, 1],
             'C': 1, 'kernel': 'linear', 'gamma': 'auto'},
    "svm6": {'data': [["LiverCirrhosis"],["LiverCirrhosis"]],'k': 5,'cvt': 10,'n': 20,'m': "svm",'classes': [0, 1],
             'C': 1, 'kernel': 'linear', 'gamma': 'auto'},
    "svm7": {'data': [["Karlsson_2013"],["Karlsson_2013"]],'k': 5,'cvt': 10,'n': 20,'m': "svm",'classes': [0, 1],
             'C': 1, 'gamma': 0.001, 'kernel': 'rbf'},
    "svm1_norm": {'data': [["Qin_et_al"],["Qin_et_al"]],'k': 5, 'cvt': 10,'n': 2,'m': "svm",'classes': [0, 1],
             'C': 10, 'kernel': 'rbf', 'gamma': 0.001},
    "svm2_norm": {'data': [["MetaHIT"],["MetaHIT"]],'k': 5,'cvt': 10,'n': 20,'m': "svm",'classes': [0, 1],
             'C': 1000, 'kernel': 'rbf', 'gamma': 0.0001},
    "svm3_norm": {'data': [["Feng"],["Feng"]],'k': 5,'cvt': 10,'n': 20,'m': "svm",'classes': [0, 1],
             'C': 1, 'kernel': 'rbf', 'gamma': 0.001},
    "svm4_norm": {'data': [["RA"],["RA"]],'k': 5,'cvt': 10,'n': 20,'m': "svm",'classes': [0, 1],
             'C': 100, 'gamma': 0.0001, 'kernel': 'rbf'},
    "svm5_norm": {'data': [["Zeller_2014"],["Zeller_2014"]],'k': 5,'cvt': 10,'n': 20,'m': "svm",'classes': [0, 1],
             'C': 100, 'kernel': 'rbf', 'gamma': 0.0001},
    "svm6_norm": {'data': [["LiverCirrhosis"],["LiverCirrhosis"]],'k': 5,'cvt': 10,'n': 20,'m': "svm",'classes': [0, 1],
             'C': 10, 'kernel': 'rbf', 'gamma': 0.001},
    "svm7_norm": {'data': [["Karlsson_2013"],["Karlsson_2013"]],'k': 5,'cvt': 10,'n': 20,'m': "svm",'classes': [0, 1],
             'C': 100, 'gamma': 0.0001, 'kernel': 'rbf'},
    "rf1_norm": {'data': [["Qin_et_al"],["Qin_et_al"]],'k': 5,'cvt': 10,'n': 20,'m': "rf",'classes': [0, 1],
            'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 500, 'n_jobs': -1},
    "rf2_norm": {'data': [["MetaHIT"],["MetaHIT"]],'k': 5,'cvt': 10,'n': 20,'m': "rf",'classes': [0, 1],
            'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 100, 'n_jobs': -1},
    "rf3_norm": {'data': [["Feng"],["Feng"]],'k': 5,'cvt': 10,'n': 20,'m': "rf",'classes': [0, 1],
            'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 200, 'n_jobs': -1},
    "rf4_norm": {'data': [["RA"],["RA"]],'k': 5,'cvt': 10,'n': 20,'m': "rf",'classes': [0, 1],
            'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 400, 'n_jobs': -1},
    "rf5_norm": {'data': [["Zeller_2014"],["Zeller_2014"]],'k': 5,'cvt': 10,'n': 20,'m': "rf",'classes': [0, 1],
            'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 400, 'n_jobs': -1},
    "rf6_norm": {'data': [["LiverCirrhosis"],["LiverCirrhosis"]],'k': 5,'cvt': 10,'n': 20,'m': "rf",'classes': [0, 1],
            'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 400, 'n_jobs': -1},
    "rf7_norm": {'data': [["Karlsson_2013"],["Karlsson_2013"]],'k': 5,'cvt': 10,'n': 20,'m': "rf",'classes': [0, 1],
            'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 400, 'n_jobs': -1},
    }

def class_to_target(cls):
    target = np.zeros((n_classes,))
    target[class_to_ind[cls]] = 1.0
    return target
def plot_confusion_matrix(cm, name = '', config='', cmap=pylab.cm.Reds):
    """
    This function plots the confusion matrix.
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    fig, ax = pylab.subplots(1, 2)
    for sub_plt, conf_mat, title, fmt in zip(ax, [cm, cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]], ['Unnormalized Confusion Matrix', 'Normalized Confusion Matrix'], ['d', '.2f']):
        im = sub_plt.imshow(conf_mat, interpolation='nearest', cmap=cmap)
        sub_plt.set_title(title, size=plot_title_size)
        divider = make_axes_locatable(sub_plt)
        cax1 = divider.append_axes("right", size="5%", pad=0.1)
        #fig.colorbar(im, ax=sub_plt)
        fig.colorbar(im, cax=cax1)
        tick_marks = np.arange(len(cm))
        sub_plt.set_xticks(tick_marks)
        sub_plt.set_yticks(tick_marks)
        sub_plt.set_xticklabels(classes)
        sub_plt.set_yticklabels(classes)
        sub_plt.tick_params(labelsize=plot_text_size, axis='both')
        thresh = 0.8*conf_mat.max()
        for i, j in product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
            sub_plt.text(j, i, format(conf_mat[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if conf_mat[i, j] > thresh else "black", size=plot_title_size)
        sub_plt.set_ylabel('True Label', size=plot_text_size)
        sub_plt.set_xlabel('Predicted Label', size=plot_text_size)
    pylab.tight_layout()
    pylab.gca().set_position((.1, 10, 0.8, .8))
    add_figtexts_and_save(fig, name + '_confusion_mat', "Confusion matrix for predicting sample's " + class_name + " status using 5mers", y_off=1.3, config=config)

def plot_roc_aucs(fpr, tpr, roc_auc, accs, std_down=None, std_up=None, config='', name= '', title='ROC Curves with AUCs/ACCs', 
                  desc="ROC/AUC plots using 5mers", xlabel='False Positive Rate', ylabel='True Positive Rate'):
    fig = pylab.figure()
    lw = 2
    if n_classes > 2:
        pylab.plot(fpr["micro"], tpr["micro"],
                   label='micro-average ROC (AUC = {0:0.4f})'
                   ''.format(roc_auc["micro"]),
                   color='deeppink', linestyle=':', linewidth=4)
        if (std_down is not None) and (std_up is not None):
            pylab.fill_between(fpr['micro'], std_down['micro'], std_up['micro'], color='deeppink', lw=0, alpha=plot_alpha)

        pylab.plot(fpr["macro"], tpr["macro"],
                   label='macro-average ROC (AUC = {0:0.4f})'
                   ''.format(roc_auc["macro"]),
                   color='navy', linestyle=':', linewidth=4)
        if (std_down is not None) and (std_up is not None):
            pylab.fill_between(fpr['macro'], std_down['macro'], std_up['macro'], color='navy', lw=0, alpha=plot_alpha)

        roc_colors = cycle(['green', 'red', 'purple', 'darkorange'])
        for i, color in zip(range(n_classes), roc_colors):
            pylab.plot(fpr[i], tpr[i], color=color, lw=lw,
                       label='ROC of {0} (AUC = {1:0.4f}, acc={2:0.4f})'
                       ''.format(classes[i], roc_auc[i], accs[i]))
            if (std_down is not None) and (std_up is not None):
                pylab.fill_between(fpr[i], std_down[i], std_up[i], color=color, lw=0, alpha=plot_alpha)
    else:
        i = 1
        pylab.plot(fpr[i], tpr[i], color='darkorange', lw=lw,
                   label='ROC of {0} (AUC = {1:0.4f}, acc={2:0.4f})'
                   ''.format(classes[i], roc_auc[i], accs[i]))
        if (std_down is not None) and (std_up is not None):
            pylab.fill_between(fpr[i], std_down[i], std_up[i], color='darkorange', lw=0, alpha=plot_alpha)
    
    pylab.plot([0, 1], [0, 1], 'k--', lw=lw)
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.05])
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)
    pylab.title(title, size=plot_title_size)
    pylab.legend(loc="lower right", prop={'size': plot_text_size})
    pylab.gca().set_position((.1, .7, .8, .8))
    add_figtexts_and_save(fig, name, desc, config=config)
    
def add_figtexts_and_save(fig, name, desc, x_off=0.02, y_off=0.56, step=0.04, config=None):
    filename = graph_dir + '/' + name + "_roc_auc" + '_' + config + '.svg'
    pylab.savefig(filename , bbox_inches='tight')
    pylab.close(fig)


# This reference explains some of the things I'm doing here
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
if __name__ == '__main__':
    # User passes the model to be used as a command-line argument, which is parsed here.
    # The default model is Random Forest 
    
    # Loop over all data sets
    for dataset in dataset_model_grid.keys():
        dataset_config_iter_fold_results[dataset] = {}
        config_results = {}
        model = dataset_model_grid[dataset]
        param_grid = model_param_grid[model]
        kmer_size = param_grid["k"]
        classes = param_grid["classes"]
        n_classes = len(classes)
        global class_to_ind
        class_to_ind = { classes[i]: i for i in range(n_classes)}

        data_sets = param_grid["data"]
        data_sets_healthy=data_sets[0]
        data_sets_diseased=data_sets[1]
        
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

        labels=np.asarray(labels)
        labels=labels.astype(np.int)

        # Normalize and shuffle the data
        data_normalized = normalize(kmer_cnts, axis = 1, norm = 'l1')
        data_normalized, labels = shuffle(data_normalized, labels, random_state=0)

        # Set up data and labels
        x = data_normalized
        y = labels
        n_iter = param_grid["n"]
        learn_type = param_grid["m"]
        cv_testfolds = param_grid["cvt"]

        
        for i in range(n_iter):
            # Set the estimator based on the model type
            if (learn_type == "enet"):
                # doing a separate grid search using stratified k fold -- k - 1 folds should be used
                # for training/grid search, the last fold should be used for test
                estimator = ElasticNet(alphas = param_grid["alpha"], l1_ratio = param_grid["l1"], cv = cv_gridsearch)
            elif (learn_type == "lasso"):
                estimator = Lasso(alphas = param_grid["alpha"], cv = cv_gridsearch)
            elif (learn_type == "svm"):
                C = param_grid["C"]
                gamma = param_grid["gamma"]
                kernel = param_grid["kernel"]
                estimator = SVC(C = C, gamma = gamma, kernel = kernel, probability = True)
            else:
                criterion = param_grid["criterion"]
                max_depth = param_grid["max_depth"]
                max_features = param_grid["max_features"]
                min_samples_split = param_grid["min_samples_split"]
                n_estimators = param_grid["n_estimators"]
                n_jobs = -1
                estimator = RandomForestClassifier(criterion=criterion, max_depth=max_depth, max_features=max_features,
                                                   min_samples_split=min_samples_split, n_estimators=n_estimators, n_jobs=n_jobs)

            skf = StratifiedKFold(n_splits = cv_testfolds, shuffle = True)
            kfold = 0
            for train_i, test_i in skf.split(x, y):
                x_train, y_train = x[train_i], y[train_i]
                x_test, y_test = x[test_i], y[test_i]

                sample_mean = x_train.mean(axis=0)
                sample_std = x_train.std(axis=0)

                # Normalize both training and test samples with the training mean and std
                x_train = (x_train - sample_mean) / sample_std
                # test samples are normalized using only the mean and std of the training samples
                x_test = (x_test - sample_mean) / sample_std
                
                y_train = np.array(y_train)
                y_test = np.array(y_test)

                estimator.fit(x_train, y_train)
                y_test_pred= np.array(estimator.predict_proba(x_test))

                if learn_type == 'enet' or learn_type == 'lasso':
                    # Convert the predicted values to 0 or 1
                    for r in range(len(predictions)):
                        if (predictions[r] > 0.5):
                            predictions[r] = 1
                        else:
                            predictions[r] = 0

                # plot the confusion matrix
                # compare the true label index (with max value (1.0) in the target vector) against the predicted
                # label index (index of label with highest predicted probability)

                conf_mat = confusion_matrix(y_test, np.argmax(y_test_pred, axis=1), labels=range(n_classes))
                if plot_fold:
                    plot_confusion_matrix(conf_mat, name = dataset + "_" + model, config = "IT_" + str(i) + "_FO_" + str(kfold))

                # printing the accuracy rates for diagnostics
                print("Total accuracy for " + str(len(y_test_pred)) + " test samples: " +
                      str(np.mean(np.equal(y_test, np.argmax(y_test_pred, axis=1)))))

                for cls in classes:
                    idx = []
                    for j in range(y_test_pred.shape[0]):
                        if y_test[j] == cls:
                            idx.append(j)
                    if len(idx) == 0:
                        continue
                    idx = np.array(idx)
                    print("Accuracy for total of " + str(len(idx)) + " " + str(cls) + " samples: " +
                          str(np.mean(np.equal(y_test[idx], np.argmax(y_test_pred[idx], axis=1)))))

                # Compute ROC curve and AUC for each class - http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                acc = dict()
                for j in range(n_classes):
                    fpr[j], tpr[j], _ = roc_curve(y_test, y_test_pred[:, j])
                    roc_auc[j] = auc(fpr[j], tpr[j])
                    # Round float 1.0 to integer 1 and 0.0 to 0 in the target vectors, and 1 for max predicted prob
                    # index being this one (i), 0 otherwise
                    acc[j] = accuracy_score(np.round(y_test), np.equal(np.argmax(y_test_pred, axis=1), j))
                
                # Compute micro-average ROC curve and ROC area
                fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), np.argmax(y_test_pred, axis = 1).ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

                # First aggregate all false positive rates
                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
                
                # Then interpolate all ROC curves at these points
                mean_tpr = np.zeros_like(all_fpr)
                for j in range(n_classes):
                    mean_tpr += np.interp(all_fpr, fpr[j], tpr[j])

                # Finally average it and compute AUC
                mean_tpr /= n_classes


                fpr["macro"] = all_fpr
                tpr["macro"] = mean_tpr
                roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

                # Plot all ROC curves
                if plot_fold:
                    # plot the ROCs with AUCs
                    plot_roc_aucs(fpr, tpr, roc_auc, acc, name = dataset + "_" + model, config = "IT_" + str(i) + "_FO_" + str(kfold))

                # calculate the accuracy/f1/precision/recall for this test fold - same way as in Pasolli
                test_true_label_inds = y_test
                test_pred_label_inds = np.argmax(y_test_pred, axis = 1)
                accuracy = accuracy_score(test_true_label_inds, test_pred_label_inds)
                f1 = f1_score(test_true_label_inds, test_pred_label_inds, pos_label=None, average='weighted')
                precision = precision_score(test_true_label_inds, test_pred_label_inds, pos_label=None, average='weighted')
                recall = recall_score(test_true_label_inds, test_pred_label_inds, pos_label=None, average='weighted')
                print(('{}\t{}\t{}\t{}\t{}\t{}\tfold-perf-metrics for ' + class_name).
                      format(accuracy, f1, precision, recall, roc_auc[1], roc_auc['macro']))
                # the config info for this exp but no fold/iter indices because we need to aggregate stats over them
        
                config_iter_fold_results = dataset_config_iter_fold_results[dataset]
                if model not in config_iter_fold_results:
                    config_iter_fold_results[model] = []

                # the iteration and fold indices
                # extend the list for the iteration if necessary

                if len(config_iter_fold_results[model]) <= i:
                    config_iter_fold_results[model].append([])

                # extend the list for the fold if necessary
                if len(config_iter_fold_results[model][i]) <= kfold:
                    config_iter_fold_results[model][i].append([])


                config_iter_fold_results[model][i][kfold] = [conf_mat, [fpr, tpr, roc_auc], classes, [y_test, y_test_pred], [accuracy, f1, precision, recall, roc_auc[1], roc_auc['macro']]]
                fold_results = np.array(config_iter_fold_results[model][i])
                kfold += 1


                
            for config in config_iter_fold_results:
                # K-fold results
                fold_results = np.array(config_iter_fold_results[config][i])

                # the config for this iteration
                config_iter = dataset + '_' + 'iter:' + str(i)

                # sum the confusion matrices over the folds
                conf_mat = np.sum(fold_results[:, 0], axis=0)

                # mean accuracy across K folds for this iteration
                mean_accuracy = np.mean([fold_results[k, 4][0] for k in range(len(fold_results))], axis=0)

                # save the results for this iteration so we can aggregate the overall results at the end
                if not config in config_results:
                    config_results[config] = []

                config_results[config].append([conf_mat, fold_results[:, 1], fold_results[:, 3], fold_results[:, 4]])

                # Per-iteration plots across K folds
                if plot_iter:
                    # plot the confusion matrix
                    plot_confusion_matrix(conf_mat, name = dataset + "_" + model, config = "IT_" + str(i))
        
        for config in config_results:
            # per iteration results
            iter_results = np.array(config_results[config])

            # sum the confusion matrices over iterations
            conf_mat = np.sum(iter_results[:, 0], axis=0)

            all_fold_roc_aucs = np.concatenate(iter_results[:, 1])
            all_y_test_pred = np.concatenate(iter_results[:, 2])
            
            all_y_test = np.array(np.concatenate([r[0] for r in all_y_test_pred], axis = 0))
            all_y_pred = np.array(np.concatenate([r[1] for r in all_y_test_pred], axis = 0))

            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            accs = dict()
            std_down = dict()
            std_up = dict()

            for i in range(n_classes):
                true_probs = [r for r in all_y_test]
                pred_probs = [r[i] for r in all_y_pred]
                fpr[i], tpr[i], _ = roc_curve(true_probs, pred_probs)
                roc_auc[i] = auc(fpr[i], tpr[i])
                accs[i] = accuracy_score(np.round(true_probs), np.equal(np.argmax(all_y_pred, axis=1), i))

            fpr["micro"], tpr["micro"], _ = roc_curve(all_y_test.ravel(), np.argmax(all_y_pred, axis=1).ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at these points
            all_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                all_tpr += np.interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            all_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = all_tpr
            roc_auc['macro'] = auc(all_fpr, all_tpr)

            i_tpr = dict()
            # use the per-fold ROCs to calculate the CI band around the overall ROC as done in Pasolli
            for fprs, tprs, roc_aucs in all_fold_roc_aucs:
                for i in list(range(n_classes)) + ['micro', 'macro']:
                    if not np.isnan(tprs[i][0]):
                        try:
                            i_tpr[i].append(np.interp(fpr[i], fprs[i], tprs[i]))
                        except KeyError:
                            i_tpr[i] = [np.interp(fpr[i], fprs[i], tprs[i])]

            for i in list(range(n_classes)) + ['micro', 'macro']:
                std_down[i] = np.maximum(tpr[i] - np.std(i_tpr[i], axis=0)*plot_factor/np.sqrt(cv_testfolds), 0)
                std_up[i] = np.minimum(tpr[i] + np.std(i_tpr[i], axis=0)*plot_factor/np.sqrt(cv_testfolds), 1.0)

            if plot_overall:
                # plot the confusion matrix
                plot_confusion_matrix(conf_mat, name = dataset + "_" + model)

                # plot the ROCs with AUCs/ACCs
                plot_roc_aucs(fpr, tpr, roc_auc, accs, std_down, std_up, name = dataset + "_" + model)

            print('tkfold-overall-conf-mat: ' + str(conf_mat) + ' for ' + class_name + ':' + dataset + "_" + model)
            
            perf_metrics = np.vstack(np.concatenate(iter_results[:, 3]))
            perf_means = np.mean(perf_metrics, axis=0)
            perf_stds = np.std(perf_metrics, axis=0)

            # log the results for offline analysis
            print(('{}({})\t{}({})\t{}({})\t{}({})\t{}\t{}\tkfold-overall-perf-metrics for ' + class_name + ':{}').
                  format(perf_means[0], perf_stds[0], perf_means[1], perf_stds[1], perf_means[2], perf_stds[2], perf_means[3], perf_stds[3], roc_auc[1], roc_auc['macro'], config))

            # dump the model results into a file for offline analysis and plotting, e.g., merging plots from
            # different model instances (real and null)
            aggr_results = [conf_mat, fpr, tpr, roc_auc, accs, std_down, std_up, perf_means, perf_stds]

            with open("aggr_results" + config +".pickle", "wb") as f:
                dump_dict = { "dataset_info": dataset, "results": [aggr_results, dataset_config_iter_fold_results[dataset][config]]}
                pickle.dump(dump_dict, f)
                
            
