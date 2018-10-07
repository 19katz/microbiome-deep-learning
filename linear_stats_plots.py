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
import load_kmer_cnts_pasolli_jf
import warnings
import pylab
from sklearn.exceptions import ConvergenceWarning

import shap
import logging

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
    #"RA": "svm-ra-nandita",
    #"Zeller": "svm-zeller-nandita",
    #"Zeller": "svm-zeller-ks8-nandita",
    #"MetaHIT": "svm9",
    #"Feng": "svm10",
    #"Karlsson": "svm11",
    #"LiverCirrhosis": "svm12",
    #"Qin": "svm13",
    #"Zeller": "svm14",
    #"RA": "svm15",

    # "MetaHIT1": "svm46",
    # "MetaHIT2": "svm47",
    # "MetaHIT3": "svm48",

    # "LeChatelier1": "svm49",
    # "LeChatelier2": "svm50",
    # "LeChatelier3": "svm51",
    # "LeChatelier4": "svm52",

    # "LeChatelier1": "svm53",
    # "LeChatelier2": "svm54",
    # "LeChatelier3": "svm55",
    # "LeChatelier4": "svm56",
    #"LeChatelier1": "svm49-shuffle",

    # "Feng1": "svm43",
    # "Feng2": "svm44",
    # "Feng3": "svm45",

    # "Liver1": "svm40",
    # "Liver2": "svm41",
    # "Liver3": "svm42",

    # "Qin1": "svm37",
    # "Qin2": "svm38",
    # "Qin3": "svm39",
    #"Qin2-shuffle": "svm38-shuffle",

    # "RA": "svm16",
    # "RA1": "svm17",
    # "RA2": "svm18",
    # "RA3": "svm19",
    # "RA": "svm17-shuffle",
    # "RA1": "svm34",
    # "RA2": "svm35",
    # "RA3": "svm36",

    # "Zeller": "svm20",
    # "Zeller1": "svm21",
    # "Zeller2": "svm22",
    # "Zeller3": "svm23",
    #"Zeller": "svm20-shuffle",

    # "Zeller1": "svm31",
    # "Zeller2": "svm32",
    # "Zeller3": "svm33",

    # "Karlsson": "svm24",
    # "Karlsson": "svm24-shuffle",
    # "Karlsson1": "svm25",
    # "Karlsson2": "svm26",
    # "Karlsson3": "svm27",

    # "Karlsson1": "svm28",
    # "Karlsson2": "svm29",
    # "Karlsson3": "svm30",
    #"KarlssonSVMNoAdapter": "svm24-no-adapter",
    #"KarlssonSVMNoAdapter": "svm24-no-adapter-shuffle",

    #"RASVMNoAdapter": "svm17-no-adapter",
    #"RASVMNoAdapter": "svm17-no-adapter-shuffle",

    #"Feng": "rf0"
    

    #"Qin": "rf1-norm-shuffled",
    #"MetaHIT": "rf2-norm-shuffled",
    #"Feng": "rf3-norm-shuffled",
    #"RA": "rf4-norm-shuffled",
    #"Zeller": "rf5-norm-shuffled",
    #"LiverCirrhosis": "rf6-shuffled",
    #"Karlsson": "rf8-shuffled",
    #"RA": "svm-ra-nandita-shuffle",
    #"Zeller": "svm-zeller-ks8-nandita-shuffle",
    #"MetaHIT": "svm9-shuffle",
    #"Feng": "svm10-shuffle",
    #"Karlsson": "svm11-shuffle",
    #"LiverCirrhosis": "svm12-shuffle",
    

    #"Qin": "svm1-shuffled",
    #"MetaHIT": "svm2-shuffled",
    #"Feng": "svm3-shuffled",
    #"RA": "svm4-shuffled",
    #"Zeller": "svm5-shuffled",
    #"LiverCirrhosis": "svm6-shuffled",
    #"Karlsson": "svm8-shuffled",


    #"Qin": "rf1-norm",
    #"MetaHIT": "rf2-norm",
    #"Feng": "rf3-norm",
    "RA": "rf4-norm",
    #"Zeller": "rf5-norm",
    #"LiverCirrhosis": "rf6",
    #"Karlsson": "rf8",
    #"LeChatelier": "rf10-norm",
    #"LeChatelier": "rf10-norm-shuffle",

    #"KarlssonNoAdapter": "rf-karlsson-no-adapter",
    #"RANoAdapter": "rf4-norm-no-adapter",
    #"KarlssonNoAdapter2": "rf-karlsson-no-adapter2",
    #"RANoAdapter": "rf4-norm-no-adapter2-shuffle",
    #"KarlssonNoAdapter2": "rf-karlsson-no-adapter2-shuffle",
    

    #"MetaHIT": "rf10",

    #"Qin": "svm1",
    #"MetaHIT": "svm2",
    #"Feng": "svm3",
    #"RA": "svm4",
    #"Zeller": "svm5",
    #"LiverCirrhosis": "svm6",
    #"Karlsson": "svm8",
    }

model_param_grid = {
    "svm1-shuffled": {'DS': [["Qin_et_al"],["Qin_et_al"]],  'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 100, 'KN': 'linear', 'GM': 'auto', 'KS': 7, 'SL': 1},
    "rf1-norm-shuffled": {'DS': [["Qin_et_al"],["Qin_et_al"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 200, 'NJ': 1, 'KS': 10, 'NR': 1, 'SL':1},
    "svm2-shuffled": {'DS': [["MetaHIT"],["MetaHIT"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 1, 'KN': 'linear', 'GM': 'auto', 'KS': 7, 'SL':1},
    "rf2-norm-shuffled": {'DS': [["MetaHIT"],["MetaHIT"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 400, 'NJ': 1, 'KS': 10, 'NR': 1, 'SL':1},
    "svm4-shuffled": {'DS': [["RA"],["RA"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 10, 'GM': 0.0001, 'KN': 'rbf', 'KS': 7, 'SL':1},
    "rf4-norm-shuffled": {'DS': [["RA"],["RA"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 500, 'NJ': 1, 'KS': 10, 'NR': 1, 'SL':1},
    "rf3-norm-shuffled": {'DS': [["Feng"],["Feng"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 500, 'NJ': 1, 'KS': 8, 'NR': 1, 'SL':1},
    "svm3-shuffled": {'DS': [["Feng"],["Feng"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 100, 'GM': 0.0001, 'KN': 'rbf', 'KS': 6, 'SL':1},
    "rf5-norm-shuffled": {'DS': [["Zeller_2014"],["Zeller_2014"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 200, 'NJ': 1, 'KS': 10, 'NR': 1, 'SL':1},
    "svm5-shuffled": {'DS': [["Zeller_2014"],["Zeller_2014"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 100, 'GM': 0.0001, 'KN': 'rbf', 'KS': 5, 'SL':1},
    "rf8-shuffled": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 400, 'NJ': 1, 'KS': 10, 'NR': 0, 'SL':1},
    "svm8-shuffled": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
        'C': 10, 'GM': 0.0001, 'KN':'rbf', 'KS': 5, 'SL':1},
    "rf6-shuffled": {'DS': [["LiverCirrhosis"],["LiverCirrhosis"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 500, 'NJ': 1, 'KS': 10, 'NR': 0, 'SL':1},
    "svm6-shuffled": {'DS': [["LiverCirrhosis"],["LiverCirrhosis"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 10, 'KN': 'linear', 'GM': 'auto', 'KS': 6, 'SL':1},
    
    "rf_karlsson": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 400, 'NJ': 1, 'KS': 10, 'NR': 0, 'SL':0},
    "svm2-norm": {'DS': [["MetaHIT"],["MetaHIT"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 1, 'KN': 'linear', 'GM': 'auto', 'KS': 7, 'SL':0},
    "rf0-norm": {'DS': [["Qin_et_al"],["Qin_et_al"]], 'CVT': 3,'N': 1,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 10, 'NJ': 1, 'KS': 5, 'NR': 1, 'SL':0},
    "rf1-norm": {'DS': [["Qin_et_al"],["Qin_et_al"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 200, 'NJ': 1, 'KS': 10, 'NR': 1, 'SL':0},
    "rf2-norm": {'DS': [["MetaHIT"],["MetaHIT"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 400, 'NJ': 1, 'KS': 10, 'NR': 1, 'SL':0},
    "rf3-norm": {'DS': [["Feng"],["Feng"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 500, 'NJ': 1, 'KS': 8, 'NR': 1, 'SL':0},
    "rf4-norm": {'DS': [["RA"],["RA"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 500, 'NJ': 1, 'KS': 10, 'NR': 1, 'SL':0},
    "rf4-norm-no-adapter": {'DS': [["RA_no_adapter"],["RA_no_adapter"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 500, 'NJ': 1, 'KS': 10, 'NR': 1, 'SL':0},
    "rf4-norm-no-adapter2": {'DS': [["RA_no_adapter"],["RA_no_adapter"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
                             'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 200, 'NJ': 1, 'KS': 10, 'NR': 0, 'SL':0},
    "rf4-norm-no-adapter2-shuffle": {'DS': [["RA_no_adapter"],["RA_no_adapter"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
                                     'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 200, 'NJ': 1, 'KS': 10, 'NR': 0, 'SL':1},
    "rf5-norm": {'DS': [["Zeller_2014"],["Zeller_2014"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 200, 'NJ': 1, 'KS': 10, 'NR': 1, 'SL':0},
    "rf6-norm": {'DS': [["LiverCirrhosis"],["LiverCirrhosis"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 500, 'NJ': 1, 'KS': 8, 'NR': 1, 'SL':0},
    "rf7-norm": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 200, 'NJ': 1, 'KS': 10, 'NR': 1, 'SL':0},
    "rf8-norm": {'DS': [["Karlsson_2013", "Qin_et_al"],["Karlsson_2013", "Qin_et_al"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 200, 'NJ': -1, 'KS': 5, 'NR': 1, 'SL':0},
    "rf9-norm": {'DS': [["Zeller_2014", "Feng"],["Zeller_2014", "Feng"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 500, 'NJ': -1, 'KS': 5, 'NR': 1, 'SL':0},
    "rf10-norm": {'DS': [["LeChatelier"],["LeChatelier"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
                  'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 400, 'NJ': 1, 'KS': 10, 'NR': 1, 'SL':0},
    "rf10-norm-shuffle": {'DS': [["LeChatelier"],["LeChatelier"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
                          'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 400, 'NJ': 1, 'KS': 10, 'NR': 1, 'SL':1},
    "rf-karlsson-no-adapter": {'DS': [["Karlsson_2013_no_adapter"],["Karlsson_2013_no_adapter"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 400, 'NJ': 1, 'KS': 10, 'NR': 0, 'SL':0},
    "rf-karlsson-no-adapter2": {'DS': [["Karlsson_2013_no_adapter"],["Karlsson_2013_no_adapter"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 100, 'NJ': 1, 'KS': 10, 'NR': 0, 'SL':0},
    "rf-karlsson-no-adapter2-shuffle": {'DS': [["Karlsson_2013_no_adapter"],["Karlsson_2013_no_adapter"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
                                'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 100, 'NJ': 1, 'KS': 10, 'NR': 0, 'SL':1},

    "svm0": {'DS': [["Feng"],["Feng"]],  'CVT': 3,'N': 1,'M': "svm",'CL': [0, 1],
             'C': 100, 'KN': 'rbf', 'GM': 0.0001, 'KS': 5, 'SL':0},
    "svm1": {'DS': [["Qin_et_al"],["Qin_et_al"]],  'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 100, 'KN': 'linear', 'GM': 'auto', 'KS': 7, 'SL':0},
    "svm2": {'DS': [["MetaHIT"],["MetaHIT"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 1, 'KN': 'linear', 'GM': 'auto', 'KS': 7, 'SL':0},
    "svm3": {'DS': [["Feng"],["Feng"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 100, 'GM': 0.0001, 'KN': 'rbf', 'KS': 6, 'SL':0},
    "svm4": {'DS': [["RA"],["RA"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 10, 'GM': 0.0001, 'KN': 'rbf', 'KS': 7, 'SL':0},
    "svm5": {'DS': [["Zeller_2014"],["Zeller_2014"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 100, 'GM': 0.0001, 'KN': 'rbf', 'KS': 5, 'SL':0},
    "svm6": {'DS': [["LiverCirrhosis"],["LiverCirrhosis"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 10, 'KN': 'linear', 'GM': 'auto', 'KS': 6, 'SL':0},
    "svm7": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 10, 'GM': 0.0001, 'KN':'rbf', 'KS': 5, 'SL':0},
    "svm8": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
        'C': 1, 'GM': 0.001, 'KN':'rbf', 'KS': 5, 'SL':0},
    "svm-ra-nandita": {'DS': [["RA"],["RA"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
                       'C': 1, 'GM': 'auto', 'KN': 'linear', 'KS': 8, 'SL':0},
    "svm-ra-nandita-shuffle": {'DS': [["RA"],["RA"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
                       'C': 1, 'GM': 'auto', 'KN': 'linear', 'KS': 8, 'SL':1},
    "svm-zeller-nandita": {'DS': [["Zeller_2014"],["Zeller_2014"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
                       'C': 1000, 'GM': 'auto', 'KN': 'linear', 'KS': 10, 'SL':0},
    "svm-zeller-ks8-nandita": {'DS': [["Zeller_2014"],["Zeller_2014"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
                               'C': 10, 'GM': 'auto', 'KN': 'linear', 'KS': 8, 'SL':0},
    "svm-zeller-ks8-nandita-shuffle": {'DS': [["Zeller_2014"],["Zeller_2014"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
                               'C': 10, 'GM': 'auto', 'KN': 'linear', 'KS': 8, 'SL':1},
    "svm9": {'DS': [["MetaHIT"],["MetaHIT"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 1, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm10": {'DS': [["Feng"],["Feng"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 1, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm11": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 1, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm9-shuffle": {'DS': [["MetaHIT"],["MetaHIT"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
                     'C': 1, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':1},
    "svm10-shuffle": {'DS': [["Feng"],["Feng"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 1, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':1},
    "svm11-shuffle": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 1, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':1},
    "svm12": {'DS': [["LiverCirrhosis"],["LiverCirrhosis"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 1, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm12-shuffle": {'DS': [["LiverCirrhosis"],["LiverCirrhosis"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
                      'C': 1, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':1},
    "svm13": {'DS': [["Qin_et_al"],["Qin_et_al"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 1, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm13-shuffle": {'DS': [["Qin_et_al"],["Qin_et_al"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
                      'C': 1, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':1},
    "svm14": {'DS': [["Zeller_2014"],["Zeller_2014"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 1, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm14-shuffle": {'DS': [["Zeller_2014"],["Zeller_2014"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
                      'C': 1, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':1},
    "svm15": {'DS': [["RA"],["RA"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 1, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm15-shuffle": {'DS': [["RA"],["RA"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
                      'C': 1, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':1},

    "svm16": {'DS': [["RA"],["RA"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 1, 'KN': 'linear', 'GM': 'auto', 'KS': 10, 'SL':0},
    "svm17": {'DS': [["RA"],["RA"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 10, 'KN': 'linear', 'GM': 'auto', 'KS': 10, 'SL':0},
    "svm17-no-adapter": {'DS': [["RA_no_adapter"],["RA_no_adapter"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 10, 'KN': 'linear', 'GM': 'auto', 'KS': 10, 'SL':0},
    "svm17-no-adapter-shuffle": {'DS': [["RA_no_adapter"],["RA_no_adapter"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
                                 'C': 10, 'KN': 'linear', 'GM': 'auto', 'KS': 10, 'SL':1},
    "svm17-shuffle": {'DS': [["RA"],["RA"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
                      'C': 10, 'KN': 'linear', 'GM': 'auto', 'KS': 10, 'SL':1},
    "svm18": {'DS': [["RA"],["RA"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 100, 'KN': 'linear', 'GM': 'auto', 'KS': 10, 'SL':0},
    "svm19": {'DS': [["RA"],["RA"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 1000, 'KN': 'linear', 'GM': 'auto', 'KS': 10, 'SL':0},

    "svm20": {'DS': [["Zeller_2014"],["Zeller_2014"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 1, 'KN': 'linear', 'GM': 'auto', 'KS': 10, 'SL':0},
    "svm20-shuffle": {'DS': [["Zeller_2014"],["Zeller_2014"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
                      'C': 1, 'KN': 'linear', 'GM': 'auto', 'KS': 10, 'SL':1},
    "svm21": {'DS': [["Zeller_2014"],["Zeller_2014"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 10, 'KN': 'linear', 'GM': 'auto', 'KS': 10, 'SL':0},
    "svm22": {'DS': [["Zeller_2014"],["Zeller_2014"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 100, 'KN': 'linear', 'GM': 'auto', 'KS': 10, 'SL':0},
    "svm23": {'DS': [["Zeller_2014"],["Zeller_2014"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 1000, 'KN': 'linear', 'GM': 'auto', 'KS': 10, 'SL':0},

    "svm24": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 1, 'KN': 'linear', 'GM': 'auto', 'KS': 10, 'SL':0},
    "svm24-no-adapter": {'DS': [["Karlsson_2013_no_adapter"],["Karlsson_2013_no_adapter"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 1, 'KN': 'linear', 'GM': 'auto', 'KS': 10, 'SL':0},
    "svm24-no-adapter-shuffle": {'DS': [["Karlsson_2013_no_adapter"],["Karlsson_2013_no_adapter"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
                                 'C': 1, 'KN': 'linear', 'GM': 'auto', 'KS': 10, 'SL':1},
    "svm24-shuffle": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 1, 'KN': 'linear', 'GM': 'auto', 'KS': 10, 'SL':1},
    "svm25": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 10, 'KN': 'linear', 'GM': 'auto', 'KS': 10, 'SL':0},
    "svm26": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 100, 'KN': 'linear', 'GM': 'auto', 'KS': 10, 'SL':0},
    "svm27": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 1000, 'KN': 'linear', 'GM': 'auto', 'KS': 10, 'SL':0},
    "svm28": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 10, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm29": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 100, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm30": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 1000, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},

    "svm31": {'DS': [["Zeller_2014"],["Zeller_2014"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 10, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm32": {'DS': [["Zeller_2014"],["Zeller_2014"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 100, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm33": {'DS': [["Zeller_2014"],["Zeller_2014"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 1000, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm34": {'DS': [["RA"],["RA"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 10, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm35": {'DS': [["RA"],["RA"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 100, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm36": {'DS': [["RA"],["RA"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 1000, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm37": {'DS': [["Qin_et_al"],["Qin_et_al"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 10, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm38": {'DS': [["Qin_et_al"],["Qin_et_al"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 100, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm38-shuffle": {'DS': [["Qin_et_al"],["Qin_et_al"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
                      'C': 100, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':1},
    "svm39": {'DS': [["Qin_et_al"],["Qin_et_al"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 1000, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm40": {'DS': [["LiverCirrhosis"],["LiverCirrhosis"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 10, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm41": {'DS': [["LiverCirrhosis"],["LiverCirrhosis"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 100, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm42": {'DS': [["LiverCirrhosis"],["LiverCirrhosis"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 1000, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm43": {'DS': [["Feng"],["Feng"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 10, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm44": {'DS': [["Feng"],["Feng"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 100, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm45": {'DS': [["Feng"],["Feng"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 1000, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm46": {'DS': [["MetaHIT"],["MetaHIT"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 10, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm47": {'DS': [["MetaHIT"],["MetaHIT"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 100, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm48": {'DS': [["MetaHIT"],["MetaHIT"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 1000, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},

    "svm49": {'DS': [["LeChatelier"],["LeChatelier"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 1, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm49-shuffle": {'DS': [["LeChatelier"],["LeChatelier"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
                      'C': 1, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':1},
    "svm50": {'DS': [["LeChatelier"],["LeChatelier"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 10, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm51": {'DS': [["LeChatelier"],["LeChatelier"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 100, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},
    "svm52": {'DS': [["LeChatelier"],["LeChatelier"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 1000, 'KN': 'linear', 'GM': 'auto', 'KS': 8, 'SL':0},

    "svm53": {'DS': [["LeChatelier"],["LeChatelier"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 1, 'KN': 'linear', 'GM': 'auto', 'KS': 10, 'SL':0},
    "svm54": {'DS': [["LeChatelier"],["LeChatelier"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 10, 'KN': 'linear', 'GM': 'auto', 'KS': 10, 'SL':0},
    "svm55": {'DS': [["LeChatelier"],["LeChatelier"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 100, 'KN': 'linear', 'GM': 'auto', 'KS': 10, 'SL':0},
    "svm56": {'DS': [["LeChatelier"],["LeChatelier"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
              'C': 1000, 'KN': 'linear', 'GM': 'auto', 'KS': 10, 'SL':0},
    
    "svm1_norm": {'DS': [["Qin_et_al"],["Qin_et_al"]],  'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 10, 'KN': 'rbf', 'GM': 0.001, 'KS': 5, 'SL':0},
    "svm2_norm": {'DS': [["MetaHIT"],["MetaHIT"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 1000, 'KN': 'rbf', 'GM': 0.0001, 'KS': 5, 'SL':0},
    "svm3_norm": {'DS': [["Feng"],["Feng"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 1, 'KN': 'rbf', 'GM': 0.001, 'KS': 5, 'SL':0},
    "svm4_norm": {'DS': [["RA"],["RA"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 100, 'GM': 0.0001, 'KN': 'rbf', 'KS': 5, 'SL':0},
    "svm5_norm": {'DS': [["Zeller_2014"],["Zeller_2014"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 100, 'KN': 'rbf', 'GM': 0.0001, 'KS': 5, 'SL':0},
    "svm6_norm": {'DS': [["LiverCirrhosis"],["LiverCirrhosis"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 10, 'KN': 'rbf', 'GM': 0.001, 'KS': 5, 'SL':0},
    "svm7_norm": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 100, 'GM': 0.0001, 'KN': 'rbf', 'KS': 5, 'SL':0},
    "svm8_norm": {'DS': [["Karlsson_2013", "Qin_et_al"],["Karlsson_2013", "Qin_et_al"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 10, 'GM': 0.001, 'KN': 'rbf', 'KS': 5, 'SL':0},
    "svm9_norm": {'DS': [["Zeller_2014", "Feng"],["Zeller_2014", "Feng"]], 'CVT': 10,'N': 20,'M': "svm",'CL': [0, 1],
             'C': 10, 'KN': 'rbf', 'GM': 0.001, 'KS': 5, 'SL':0},
    
    "rf0": {'DS': [["Feng"],["Feng"]],  'CVT': 3,'N': 1,'M': "rf", "CL":[0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 100, 'NJ': 1, 'KS': 5, 'NR': 0, 'SL':0},
    "rf1": {'DS': [["Qin_et_al"],["Qin_et_al"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 400, 'NJ': 1, 'KS': 10, 'NR': 0, 'SL':0},
    "rf2": {'DS': [["MetaHIT"],["MetaHIT"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 400, 'NJ': 1, 'KS': 8, 'NR': 0, 'SL':0},
    "rf3": {'DS': [["Feng"],["Feng"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 400, 'NJ': 1, 'KS': 8, 'NR': 0, 'SL':0},
    "rf4": {'DS': [["RA"],["RA"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 500, 'NJ': 1, 'KS': 10, 'NR': 0, 'SL':0},
    "rf5": {'DS': [["Zeller_2014"],["Zeller_2014"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 500, 'NJ': 1, 'KS': 10, 'NR': 0, 'SL':0},
    "rf6": {'DS': [["LiverCirrhosis"],["LiverCirrhosis"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 500, 'NJ': 1, 'KS': 10, 'NR': 0, 'SL':0},
    "rf7": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 500, 'NJ': 1, 'KS': 10, 'NR': 0, 'SL':0},
    "rf8": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 400, 'NJ': 1, 'KS': 10, 'NR': 0, 'SL':0},
    "rf9": {'DS': [["LiverCirrhosis"],["LiverCirrhosis"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 500, 'NJ': 1, 'KS': 10, 'NR': 0, 'SL':0},
    "rf10": {'DS': [["MetaHIT"],["MetaHIT"]], 'CVT': 10,'N': 20,'M': "rf",'CL': [0, 1],
            'CR': 'gini', 'MD': None, 'MF': 'sqrt', 'MS': 2, 'NE': 500, 'NJ': 1, 'KS': 7, 'NR': 0, 'SL':0},

    "gb1": {'DS': [["Qin_et_al"],["Qin_et_al"]], 'CVT': 10,'N': 20,'M': "gb",'CL': [0, 1],
             'LR': 0.1, 'MD': None, 'MF': 'sqrt', 'ML': 5, 'MS': 2, 'NE': 500, 'SS': 0.8, 'KS': 5, 'SL':0},
    "gb2": {'DS': [["MetaHIT"],["MetaHIT"]], 'CVT': 10,'N': 20,'M': "gb",'CL': [0, 1],
            'LR': 0.1, 'MD': None, 'MF': 'sqrt', 'ML': 5, 'MS': 2, 'NE': 400, 'SS': 0.8, 'KS': 5, 'SL':0},
    "gb3": {'DS': [["Feng"],["Feng"]], 'CVT': 10,'N': 20,'M': "gb",'CL': [0, 1],
            'LR': 0.01, 'MD': None, 'MF': 'sqrt', 'ML': 1, 'MS': 2, 'NE': 500, 'SS': 0.8, 'KS': 5, 'SL':0},
    "gb4": {'DS': [["RA"],["RA"]], 'CVT': 10,'N': 20,'M': "gb",'CL': [0, 1],
            'LR': 0.05, 'MD': None, 'MF': 'sqrt', 'ML': 2, 'MS': 2, 'NE': 100, 'SS': 0.8, 'KS': 5, 'SL':0},
    "gb5": {'DS': [["Zeller_2014"],["Zeller_2014"]], 'CVT': 10,'N': 20,'M': "gb",'CL': [0, 1],
            'LR': 0.01, 'MD': None, 'MF': 'sqrt', 'ML': 2, 'MS': 2, 'NE': 500, 'SS': 0.8, 'KS': 5, 'SL':0},
    "gb6": {'DS': [["LiverCirrhosis"],["LiverCirrhosis"]], 'CVT': 10,'N': 20,'M': "gb",'CL': [0, 1],
            'LR': 0.05, 'MD': None, 'MF': 'sqrt', 'ML': 4, 'MS': 2, 'NE': 500, 'SS': 0.8, 'KS': 5, 'SL':0},
    "gb7": {'DS': [["Karlsson_2013"],["Karlsson_2013"]], 'CVT': 10,'N': 20,'M': "gb",'CL': [0, 1],
            'LR': 0.01, 'MD': None, 'MF': 'sqrt', 'ML': 4, 'MS': 2, 'NE': 200, 'SS': 0.8, 'KS':5, 'SL':0}
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
    parser.add_argument('-logiterfeats', type = bool, default = False, help = "Whether to log all feature importances from each iteration")
    parser.add_argument('-version', type = str, default = '1', help = "Version of the model being run")

    arg_vals = parser.parse_args()
    num_features = arg_vals.features
    log_iter_feats = arg_vals.logiterfeats
    version = arg_vals.version
    if log_iter_feats:
        num_features = -1
    
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
        data_set = param_grid["DS"][0]
        kmer_size = param_grid["KS"]
        
        # Retrieve diseased data and labels
        allowed_labels = ['0', '1']
        kmer_cnts, accessions, labels, domain_labels = load_kmer_cnts_pasolli_jf.load_kmers(kmer_size, data_set, allowed_labels)
        print("LOADED DATASET " + str(data_set[0]) + ": " + str(len(kmer_cnts)) + " SAMPLES")
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
        shuffle_labels = param_grid['SL']
        param_grid['V'] = version

        config_string = config_info(data_set[0], learn_type, param_grid, kmer_size)

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
            if shuffle_labels:
                np.random.shuffle(y)
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
                    if num_features != 0:
                        get_feature_importances(estimator, kmer_imps)
                    shap_values = []

                if learn_type == 'svm':
                    if num_features != 0:
                        background = np.zeros((1, len(kmers_no_comp)))
                        try:
                            estimator.predict_proba(background)
                            explainer = shap.KernelExplainer(estimator.predict_proba, background, link="logit")
                            shap_values = explainer.shap_values(x_test, nsamples=100)
                            shap_values = np.sum(np.absolute(shap_values[1]), axis=0)
                        except Exception as e:
                            print("Got exception: " + str(e) + " on " + str(data_set))
                            shap_values = []
                    else:
                        shap_values = []

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

                config_iter_fold_results[config_string][i][kfold] = [conf_mat, [fpr, tpr, roc_auc], classes, [y_test, y_test_pred], [accuracy, f1, precision, recall, roc_auc[1], roc_auc['macro']], shap_values]
                fold_results = np.array(config_iter_fold_results[config_string][i])
                kfold += 1
                logging.info("Completed fold " + str(kfold) + " of " + str(cv_testfolds) + " for iteration " + str(i) + " of " + str(n_iter))
                
            for config in config_iter_fold_results:
                # K-fold results
                fold_results = np.array(config_iter_fold_results[config][i])
                try:
                    shap_vals = np.sum(fold_results[:, 5], axis=0)
                except Exception as e:
                    print("Got exception " +  str(e) + " on " + str(data_set))
                    shap_vals = []

                # the config for this iteration
                config_iter = config + '_' + 'IT:' + str(i)

                # sum the confusion matrices over the folds
                conf_mat = np.sum(fold_results[:, 0], axis=0)

                # mean accuracy across K folds for this iteration
                mean_accuracy = np.mean([fold_results[k, 4][0] for k in range(len(fold_results))], axis=0)

                # save the results for this iteration so we can aggregate the overall results at the end
                if not config in config_results:
                    config_results[config] = []


                config_results[config].append([conf_mat, fold_results[:, 1], fold_results[:, 3], fold_results[:, 4], shap_vals])
                if log_iter_feats and (learn_type == 'rf' or learn_type == 'svm'):
                    if learn_type == 'rf':
                        imps = kmer_imps
                    else:
                        imps = shap_vals
                    print("DUMPING FEATURE IMPORTANCES FOR ITERATION " + str(i))
                    num_feature_imps = num_features
                    if (num_feature_imps == -1):
                        num_feature_imps = len(imps)
                    if imps is not None and num_feature_imps > 0:
                        file = open(graph_dir + "/feat_imps_" + config_iter + ".txt", "w")
                        file.write("Importances\tfor\t" + str(dataset) + "\t" + config_iter + "\n")
                        if learn_type == 'rf':
                            for j in range(num_feature_imps):
                                if imps[j] > 0:
                                    file.write(kmers_no_comp[j] + "\t" + str(imps[j] / ((i + 1) * cv_testfolds)) + "\n")
                        else:
                            for j in range(num_feature_imps):
                                if imps[j] > 0:
                                    file.write(kmers_no_comp[j] + "\t" + str(imps[j] / (len(y_train) + len(y_test))) + "\n")
                        print("END FEATURE IMPORTANCE DUMP FOR ITERATION " + str(i))
                        file.close()

                # Per-iteration plots across K folds
                if plot_iter:
                    # plot the confusion matrix
                    plot_confusion_matrix(conf_mat, config = config_iter)

                
        for config in config_results:
            # per iteration results
            iter_results = np.array(config_results[config])
            try:
                shap_vals = np.sum(iter_results[:, 4], axis=0)
            except Exception as e:
                print("Got exception " + str(e) + " on " + str(data_set))
                shap_vals = []


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

            if learn_type == 'rf' or learn_type == 'svm':
                if learn_type == 'rf':
                    imps = kmer_imps
                else:
                    imps = shap_vals
                print("SORTING FEATURE IMPORTANCES")
                num_feature_imps = num_features
                if (num_feature_imps == -1):
                    num_feature_imps = len(imps)
                if imps is not None and num_feature_imps > 0:
                    indices = np.argsort(imps)[::-1][0:num_feature_imps]
                    imps = imps[indices]
                    kmers_no_comp = [kmers_no_comp[i] for i in indices]
                    file = open(graph_dir + "/feat_imps_" + config + ".txt", "w")
                    file.write("Importances\tfor\t" + str(dataset) + "\t" + config + "\n")
                    if learn_type == 'rf':
                        for i in range(num_feature_imps):
                            if imps[i] > 0:
                                file.write(kmers_no_comp[i] + "\t" + str(imps[i] / (n_iter * cv_testfolds)) + "\n")
                    else:
                        for i in range(num_feature_imps):
                            if imps[i] > 0:
                                file.write(kmers_no_comp[i] + "\t" + str(imps[i] / (len(all_y_test))) + "\n")
                    print("END FEATURE IMPORTANCE DUMP")
                    file.close()
            
                

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
                   

