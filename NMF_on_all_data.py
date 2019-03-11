import os
import pandas as pd
import numpy as np
import pickle
import csv

from itertools import cycle, product
import argparse
import warnings

from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, RepeatedStratifiedKFold, train_test_split
#from sklearn import cross_validation, metrics
from sklearn.utils import shuffle
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

# import private scripts
import load_kmer_cnts_jf
import stats_utils_AEB

kmer_size = 7

data = pd.read_pickle("/pollard/home/abustion/deep_learning_microbiome/data_AEB/NMF_on_all_data/before_NMF_no_norm.pickle")

data_normalized = pd.read_pickle("/pollard/home/abustion/deep_learning_microbiome/data_AEB/NMF_on_all_data/before_NMF_with_norm.pickle")

factors=30
for n in range(2, factors + 1):
    data_NMF = stats_utils_AEB.NMF_factor(data, kmer_size, n_components = int(n), 
                                                     title=("ALL_DATA_no_norm_" + str(kmer_size) + "mers" 
                                                            + str(n) + "factors"))

factors=30
for n in range(2, factors + 1):
    data_NMF = stats_utils_AEB.NMF_factor(data_normalized, kmer_size, n_components = int(n), 
                                                     title=("ALL_DATA_no_norm_" + str(kmer_size) + "mers" 
                                                            + str(n) + "factors"))  