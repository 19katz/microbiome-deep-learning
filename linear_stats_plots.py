import os
import numpy as np
from sklearn.preprocessing import normalize

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV

from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import cross_validation, metrics

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
from sklearn.metrics import precision_recall_curve

import operator

import pickle
import matplotlib.pyplot as plt

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



dataset_config_iter_fold_results = {}


# Dictionary of parameters for each model
# Values based on the values used in the
# Pasolli paper 
dataset_model_grid = {
    "Zeller": "rf5-norm",
    "LiverCirrhosis": "rf9",
    "Qin": "rf1-norm",
    "MetaHIT": "rf2",
    "Feng": "rf3-norm",
    "RA": "rf4",
    #"Karlsson": "rf_karlsson",

    #"Qin": "rf0-norm",
    # "MetaHIT": "rf2",
    #"Feng": "rf3-norm",
    #"RA": "rf4-norm",
    #"Zeller": "rf5-norm",
    #"LiverCirrhosis": "rf6-norm",
    #"Karlsson": "rf_karlsson",
    #"All-CRC": "rf9_norm",
    #"All-T2D": "rf8_norm",
    }


model_param_grid = {
    "rf_karlsson": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 400, 'NJ': 1, 'KS': 10, 'NR': 0},
    
    "rf0-norm": {'DS': [["Qin_et_al"],["Qin_et_al"]], 'CVT': 3,'N': 1,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 10, 'NJ': 1, 'KS': 5, 'NR': 1},
    "rf1-norm": {'DS': [["Qin_et_al"],["Qin_et_al"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 200, 'NJ': 1, 'KS': 10, 'NR': 1},
    "rf2-norm": {'DS': [["MetaHIT"],["MetaHIT"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 500, 'NJ': 1, 'KS': 10, 'NR': 1},
    "rf3-norm": {'DS': [["Feng"],["Feng"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 500, 'NJ': 1, 'KS': 7, 'NR': 1},
    "rf4-norm": {'DS': [["RA"],["RA"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 400, 'NJ': 1, 'KS': 10, 'NR': 1},
    "rf5-norm": {'DS': [["Zeller_2014"],["Zeller_2014"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 200, 'NJ': 1, 'KS': 10, 'NR': 1},
    "rf6-norm": {'DS': [["LiverCirrhosis"],["LiverCirrhosis"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 100, 'NJ': 1, 'KS': 8, 'NR': 1},
    "rf7-norm": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 100, 'NJ': 1, 'KS': 10, 'NR': 1},
    "rf8-norm": {'DS': [["Karlsson_2013", "Qin_et_al"],["Karlsson_2013", "Qin_et_al"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 200, 'NJ': -1, 'KS': 5, 'NR': 1},
    "rf9-norm": {'DS': [["Zeller_2014", "Feng"],["Zeller_2014", "Feng"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 500, 'NJ': -1, 'KS': 5, 'NR': 1},
    
    "svm1": {'DS': [["Qin_et_al"],["Qin_et_al"]],  'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 100, 'KN': 'linear', 'GM': 'auto', 'KS': 7},
    "svm2": {'DS': [["MetaHIT"],["MetaHIT"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 10, 'KN': 'linear', 'GM': 'auto', 'KS': 7},
    "svm3": {'DS': [["Feng"],["Feng"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 100, 'GM': 0.0001, 'KN': 'rbf', 'KS': 6},
    "svm4": {'DS': [["RA"],["RA"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 10, 'GM': 0.0001, 'KN': 'rbf', 'KS': 7},
    "svm5": {'DS': [["Zeller_2014"],["Zeller_2014"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 100, 'GM': 0.0001, 'KN': 'rbf', 'KS': 5},
    "svm6": {'DS': [["LiverCirrhosis"],["LiverCirrhosis"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 10, 'KN': 'linear', 'GM': 'auto', 'KS': 6},
    "svm7": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 10, 'GM': 0.0001, 'KN':'rbf', 'KS': 5},
    "svm8": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
        'C': 1, 'GM': 0.001, 'KN':'rbf', 'KS': 5},
    
    "svm1_norm": {'DS': [["Qin_et_al"],["Qin_et_al"]],  'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 10, 'KN': 'rbf', 'GM': 0.001, 'KS': 5},
    "svm2_norm": {'DS': [["MetaHIT"],["MetaHIT"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 1000, 'KN': 'rbf', 'GM': 0.0001, 'KS': 5},
    "svm3_norm": {'DS': [["Feng"],["Feng"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 1, 'KN': 'rbf', 'GM': 0.001, 'KS': 5},
    "svm4_norm": {'DS': [["RA"],["RA"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 100, 'GM': 0.0001, 'KN': 'rbf', 'KS': 5},
    "svm5_norm": {'DS': [["Zeller_2014"],["Zeller_2014"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 100, 'KN': 'rbf', 'GM': 0.0001, 'KS': 5},
    "svm6_norm": {'DS': [["LiverCirrhosis"],["LiverCirrhosis"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 10, 'KN': 'rbf', 'GM': 0.001, 'KS': 5},
    "svm7_norm": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 100, 'GM': 0.0001, 'KN': 'rbf', 'KS': 5},
    "svm8_norm": {'DS': [["Karlsson_2013", "Qin_et_al"],["Karlsson_2013", "Qin_et_al"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 10, 'GM': 0.001, 'KN': 'rbf', 'KS': 5},
    "svm9_norm": {'DS': [["Zeller_2014", "Feng"],["Zeller_2014", "Feng"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 10, 'KN': 'rbf', 'GM': 0.001, 'KS': 5},
    
    "rf1": {'DS': [["Qin_et_al"],["Qin_et_al"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 400, 'NJ': 1, 'KS': 8, 'NR': 0},
    "rf2": {'DS': [["MetaHIT"],["MetaHIT"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 200, 'NJ': 1, 'KS': 10, 'NR': 0},
    "rf3": {'DS': [["Feng"],["Feng"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 400, 'NJ': 1, 'KS': 7, 'NR': 0},
    "rf4": {'DS': [["RA"],["RA"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 500, 'NJ': 1, 'KS': 10, 'NR': 0},
    "rf5": {'DS': [["Zeller_2014"],["Zeller_2014"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 500, 'NJ': 1, 'KS': 10, 'NR': 0},
    "rf6": {'DS': [["LiverCirrhosis"],["LiverCirrhosis"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 500, 'NJ': 1, 'KS': 8, 'NR': 0},
    "rf7": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 500, 'NJ': 1, 'KS': 10, 'NR': 0},
    "rf8": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 400, 'NJ': 1, 'KS': 10, 'NR': 0},
    "rf9": {'DS': [["LiverCirrhosis"],["LiverCirrhosis"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 500, 'NJ': 1, 'KS': 10, 'NR': 0},

    "gb1": {'DS': [["Qin_et_al"],["Qin_et_al"]], 'CVT': 10,'N': 20,'M': "gb",'CL': [0, 1],
             'LR': 0.1, 'MD': None, 'MF': 'sqrt', 'ML': 5, 'MS': 2, 'NE': 500, 'SS': 0.8, 'KS': 5},
    "gb2": {'DS': [["MetaHIT"],["MetaHIT"]], 'CVT': 10,'N': 20,'M': "gb",'CL': [0, 1],
            'LR': 0.1, 'MD': None, 'MF': 'sqrt', 'ML': 5, 'MS': 2, 'NE': 400, 'SS': 0.8, 'KS': 5},
    "gb3": {'DS': [["Feng"],["Feng"]], 'CVT': 10,'N': 20,'M': "gb",'CL': [0, 1],
            'LR': 0.01, 'MD': None, 'MF': 'sqrt', 'ML': 1, 'MS': 2, 'NE': 500, 'SS': 0.8, 'KS': 5},
    "gb4": {'DS': [["RA"],["RA"]], 'CVT': 10,'N': 20,'M': "gb",'CL': [0, 1],
            'LR': 0.05, 'MD': None, 'MF': 'sqrt', 'ML': 2, 'MS': 2, 'NE': 100, 'SS': 0.8, 'KS': 5},
    "gb5": {'DS': [["Zeller_2014"],["Zeller_2014"]], 'CVT': 10,'N': 20,'M': "gb",'CL': [0, 1],
            'LR': 0.01, 'MD': None, 'MF': 'sqrt', 'ML': 2, 'MS': 2, 'NE': 500, 'SS': 0.8, 'KS': 5},
    "gb6": {'DS': [["LiverCirrhosis"],["LiverCirrhosis"]], 'CVT': 10,'N': 20,'M': "gb",'CL': [0, 1],
            'LR': 0.05, 'MD': None, 'MF': 'sqrt', 'ML': 4, 'MS': 2, 'NE': 500, 'SS': 0.8, 'KS': 5},
    "gb7": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "gb",'CL': [0, 1],
            'LR': 0.01, 'MD': None, 'MF': 'sqrt', 'ML': 4, 'MS': 2, 'NE': 200, 'SS': 0.8, 'KS':5}
    }

def class_to_target(cls):
    target = np.zeros((n_classes,))
    target[class_to_ind[cls]] = 1.0
    return target

def plot_precision_recall(precision, recall,  average_precision, f1_score, name ='precision_recall', config = ''):
    fig = pylab.figure()

    plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
         color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    pylab.gca().set_position((.1, .7, 0.8, .8))
    add_figtexts_and_save(fig, name, '2-class Precision-Recall curve: AP={0:0.4f}, F1={1:0.4f}'.format(
              average_precision, f1_score), config=config)

    
def plot_confusion_matrix(cm, name = 'confusion_mat', config='', cmap=pylab.cm.Reds):
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
    add_figtexts_and_save(fig, name , "Confusion matrix for predicting sample's " + config + " status ", y_off=1.3, config=config)

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
    add_figtexts_and_save(fig, name + "roc_auc", desc, config=config)
    
def add_figtexts_and_save(fig, name, desc, x_off=0.02, y_off=0.56, step=0.04, config=None):
    filename = graph_dir + '/' + name + '_' + config + '.png'
    pylab.figtext(x_off, y_off, desc)
    pylab.savefig(filename , bbox_inches='tight')
    pylab.close(fig)

def config_info(dataset_name, model_name, config,  kmer_size, skip_keys=['DS', 'CL']):
    config_info = "DS:" + dataset_name
    for k in config:              
        # skip the specified keys, used for skipping the fold and iteration indices (for aggregating results across them)
        if not k in skip_keys:
            config_info += '_' + k + ':' +str(get_config_val(k, config))
    return config_info

def get_config_val(config_key, config):
    val = config[config_key]
    if type(val) is list:
        val = '-'.join([ str(c) for c in val])
    return val

def get_reverse_complement(kmer):
    kmer_rev = ''
    for c in kmer:
        if c == 'A':
            kmer_rev += 'T'
        elif c == 'T':
            kmer_rev += 'A'
        elif c == 'C':
            kmer_rev += 'G'
        else:
            kmer_rev += 'C'

    return kmer_rev[::-1]
    

def get_feature_importances(clf, kmer_imps):
    print("GETTING FEATURE IMPORTANCES")
    importances = clf.feature_importances_
    #std = np.std([tree.feature_importances_ for tree in clf.estimators_],
    #             axis=0)
    for i in range(len(importances)):
        kmer_imps[i] += importances[i]
    print("FINISHED ADDING IMPORTANCES")
    

# This reference explains some of the things I'm doing here
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= "Program to run linear machine learning models on kmer datasets")
    parser.add_argument('-features', type = int, default = -1, help = "Number of feature importances")

    arg_vals = parser.parse_args()
    num_features = arg_vals.features
    
    # Loop over all data sets
    for dataset in dataset_model_grid.keys():
        
        dataset_config_iter_fold_results[dataset] = {}
        config_results = {}
        model = dataset_model_grid[dataset]
        param_grid = model_param_grid[model]
        classes = param_grid["CL"]
        n_classes = len(classes)
        global class_to_ind
        class_to_ind = { classes[i]: i for i in range(n_classes)}
        print("GETTING DATA")
        data_sets = param_grid["DS"]
        data_sets_healthy=data_sets[0]
        data_sets_diseased=data_sets[1]
        kmer_size = param_grid["KS"]
        
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
        n_iter = param_grid['N']
        learn_type = param_grid['M']
        cv_testfolds = param_grid['CVT']

        config_string = config_info(data_sets[0][0], learn_type, param_grid, kmer_size)

        kmers_no_comp = []
        print("GENERATING ALL KMERS CAPS")
        all_kmers_caps = [''.join(_) for _ in product(['A', 'C', 'G', 'T'], repeat = kmer_size)]
        print("GENERATED ALL KMERS CAPS")
        for kmer in all_kmers_caps:
            if get_reverse_complement(kmer) not in kmers_no_comp:
                kmers_no_comp.append(kmer)
        kmer_imps = np.zeros(len(kmers_no_comp))

        print("GENERATED KMERS NO COMP")
        
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
                gamma = param_grid["GM"]
                kernel = param_grid["KN"]
                estimator = SVC(C = C, gamma = gamma, kernel = kernel, probability = True)
            elif (learn_type == "gb"):
                learning_rate = param_grid["LR"]
                n_estimators = param_grid["NE"]
                subsample = param_grid["SS"]
                max_depth = param_grid["MD"]
                max_features = param_grid["MF"]
                min_samples_split = param_grid["MS"]
                min_samples_leaf = param_grid["ML"]
                estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=400, max_depth=9, max_features='sqrt', subsample=0.8)
            else:
                criterion = param_grid["CR"]
                max_depth = param_grid["MD"]
                max_features = param_grid["MF"]
                min_samples_split = param_grid["MS"]
                n_estimators = param_grid["NE"]
                n_jobs = -1
                estimator = RandomForestClassifier(criterion=criterion, max_depth=max_depth, max_features=max_features,
                                                   min_samples_split=min_samples_split, n_estimators=n_estimators, n_jobs=n_jobs)
            skf = StratifiedKFold(n_splits = cv_testfolds, shuffle = True)
            kfold = 0
            for train_i, test_i in skf.split(x, y):
                x_train, y_train = x[train_i], y[train_i]
                x_test, y_test = x[test_i], y[test_i]
                use_norm = True
                print("KFOLD CROSS")
                if learn_type == 'rf':
                    use_norm = not not param_grid["NR"]

                if use_norm:
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
                print("FIT TO ESTIMATOR")

                if learn_type == 'rf':
                    get_feature_importances(estimator, kmer_imps)

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
                    plot_confusion_matrix(conf_mat, config = config_string + "_IT_" + str(i) + "_FO_" + str(kfold))

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
                    plot_roc_aucs(fpr, tpr, roc_auc, acc, config = config_string + "_IT:" + str(i) + "_FO:" + str(kfold))

                # calculate the accuracy/f1/precision/recall for this test fold - same way as in Pasolli
                test_true_label_inds = y_test
                test_pred_label_inds = np.argmax(y_test_pred, axis = 1)
                accuracy = accuracy_score(test_true_label_inds, test_pred_label_inds)
                f1 = f1_score(test_true_label_inds, test_pred_label_inds, pos_label=None, average='weighted')
                precision = precision_score(test_true_label_inds, test_pred_label_inds, pos_label=None, average='weighted')
                recall = recall_score(test_true_label_inds, test_pred_label_inds, pos_label=None, average='weighted')

                print(('{}\t{}\t{}\t{}\t{}\t{}\tfold-perf-metrics for ' + config_string).
                      format(accuracy, f1, precision, recall, roc_auc[1], roc_auc['macro']))
                # the config info for this exp but no fold/iter indices because we need to aggregate stats over them
        
                config_iter_fold_results = dataset_config_iter_fold_results[dataset]
                if config_string not in config_iter_fold_results:
                    config_iter_fold_results[config_string] = []

                # the iteration and fold indices
                # extend the list for the iteration if necessary

                if len(config_iter_fold_results[config_string]) <= i:
                    config_iter_fold_results[config_string].append([])

                # extend the list for the fold if necessary
                if len(config_iter_fold_results[config_string][i]) <= kfold:
                    config_iter_fold_results[config_string][i].append([])

                config_iter_fold_results[config_string][i][kfold] = [conf_mat, [fpr, tpr, roc_auc], classes, [y_test, y_test_pred], [accuracy, f1, precision, recall, roc_auc[1], roc_auc['macro']]]
                fold_results = np.array(config_iter_fold_results[config_string][i])
                kfold += 1
                
            for config in config_iter_fold_results:
                # K-fold results
                fold_results = np.array(config_iter_fold_results[config][i])

                # the config for this iteration
                config_iter = config + '_' + 'IT:' + str(i)

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
                    plot_confusion_matrix(conf_mat, config = config_iter)
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

            precision_graph, recall_graph, _ = precision_recall_curve(all_y_test, all_y_pred[:, 1])
            
            perf_metrics = np.vstack(np.concatenate(iter_results[:, 3]))
            perf_means = np.mean(perf_metrics, axis=0)
            perf_stds = np.std(perf_metrics, axis=0)

            if learn_type == 'rf':
                print("SORTING FEATURE IMPORTANCES")
                if (num_features == -1):
                    num_features = len(kmer_imps)
                
                indices = np.argsort(kmer_imps)[::-1][0:num_features]
                kmer_imps = kmer_imps[indices]
                kmers_no_comp = [kmers_no_comp[i] for i in indices]
                print("Importances\tfor\t" + str(dataset) + "\t" + config)
                for i in range(len(kmer_imps)):
                    if kmer_imps[i] > 0:
                        print(kmers_no_comp[i] + "\t" + str(kmer_imps[i] / (n_iter * cv_testfolds)))
                print("END FEATURE IMPORTANCE DUMP")

            if plot_overall:
                # plot the confusion matrix
                plot_confusion_matrix(conf_mat, config = config)

                # plot the ROCs with AUCs/ACCs
                plot_roc_aucs(fpr, tpr, roc_auc, accs, std_down, std_up, config = config)

                plot_precision_recall(precision_graph, recall_graph, perf_means[2], perf_means[1], config = config)

            print('tkfold-overall-conf-mat: ' + str(conf_mat) + ' for ' + config)

            # log the results for offline analysis
            print(('{}({})\t{}({})\t{}({})\t{}({})\t{}\t{}\tkfold-overall-perf-metrics for ' + config).
                  format(perf_means[0], perf_stds[0], perf_means[1], perf_stds[1], perf_means[2], perf_stds[2], perf_means[3], perf_stds[3], roc_auc[1], roc_auc['macro']))

            # dump the model results into a file for offline analysis and plotting, e.g., merging plots from
            # different model instances (real and null)
            aggr_results = [conf_mat, fpr, tpr, roc_auc, accs, std_down, std_up, perf_means, perf_stds]

            with open("aggr_results" + config +".pickle", "wb") as f:
                dump_dict = { "dataset_info": dataset, "results": [aggr_results, dataset_config_iter_fold_results[dataset][config]]}
                pickle.dump(dump_dict, f)
                   

