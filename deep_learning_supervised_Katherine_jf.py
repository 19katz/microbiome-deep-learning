# always run miniconda for keras:
# ./miniconda3/bin/python

import os
import numpy as np
import tensorflow as tf
import random as rn
from keras import backend as K
from keras.constraints import max_norm
from sklearn.utils import shuffle

import matplotlib
matplotlib.use('Agg') # this suppresses the console for plotting
import pylab
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.models import Input, Model, Sequential
from keras.callbacks import History, EarlyStopping
from keras import regularizers
from kmer_norms import norm_by_l1, norm_by_l2
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from itertools import cycle, product
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import re
from load_kmer_cnts_Katherine_jf import load_kmers
from exp_configs import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import shap
import argparse

from sklearn.decomposition import NMF
from sklearn.metrics import precision_recall_curve




# index of the data source name in the label list - used for more stratified splits between training and test samples and for plotting
source_ind = 2

# classification name - used only for plotting
class_name = None

# index of the true label in the label list - used for classifying and testing. It determines the y_train/y_test matrices
label_ind = None

# class names
classes = None

# number of classes
n_classes = None

# class name to index map - used for mapping class name to target vector and finding index of the color for plotting the class.
class_to_ind = None

# colors for plotting
colors = None

data_loader = None

# data source to marker dict for the currently active exp dataset
markers = None

graph_dir = os.environ['HOME'] + '/deep_learning_microbiome/analysis/kmers'

# plotting options for the autoencoder
plot_ae_fold = False
plot_ae_overall = True

# Turn off per-fold plotting when doing the grid search - otherwise, you'll get tons of plots.
plot_fold = False

# Per-iteration plotting
plot_iter = False

# Overall plotting - aggregate results across both folds and iteration
plot_overall = False

# controls transparency of the CI (Confidence Interval) band around the ROCs as in Pasolli
plot_alpha = 0.2

# factor for calculating band around the overall ROC with per fold ROCs (see Pasolli).
plot_factor = 2.26

# Add figure description for exp setup or not
add_fig_desc = False

plot_text_size = 10
plot_title_size = plot_text_size + 2

test_pct = 0.2

dataset_load_name = None


# map of <dataset_name, iteration index, K fold index> to list of <history>, <2d-codes-predicted>, etc,
# for storing per-fold results
global dataset_config_iter_fold_results
dataset_config_iter_fold_results = {}

dataset_name_dict = {'H': 'HMP', 'F': 'Feng', 'Z': 'Zeller_2014', 'R': 'RA', 'M': 'MetaHIT','L': 'LiverCirrhosis', 'K': 'Karlsson_2013', 'Q': 'Qin_et_al'}

# datasource name to marker map for plotting
datasource_marker = {'HMP': 'o', 'MetaHIT': '*', 'Qin_et_al': 'D', 'RA': '^', 'Feng': 'P', 'LiverCirrhosis': 'X', 'Zeller_2014':'s', 'Karlsson_2013': 'h'}

# dataset dictionary - this determines the specific model data and type of classification.
# Some of the datasets are not fully experimented with, e.g., obesity and BMI
dataset_dict = {
    # <dataset_name>: (<classification_name>, <function for loading the data(wrapped in lambda if it takes arguments)>, 
    # <index of the label in the list of labels>, <class name list>, <color list for plotting (in order of class names)>, <data source marker dict>)

    # For testing the autoencoder - because the K-fold cross validation code is shared with supervised learning
    # , it needs some positive labels to work, so we randomly generate some positives, which do not affect the
    # autoencoder experiment results.
    
    'SingleDiseaseMetaHIT': ('IBD', lambda kmer_size: load_kmers(kmer_size = kmer_size, data_sets = [ 'MetaHIT']), 3, [ 'Healthy', 'IBD', ],
                          ['green', 'red', ], datasource_marker, 'MetaHIT'),
    'SingleDiseaseQin': ('T2D', lambda kmer_size: load_kmers(kmer_size = kmer_size, data_sets = [ 'Qin_et_al']), 3, [ 'Healthy', 'T2D', ],
                         ['green', 'purple', ], datasource_marker, 'Qin_et_al'),
    
    'SingleDiseaseRA': ('RA', lambda kmer_size: load_kmers(kmer_size = kmer_size, data_sets = [ 'RA']), 3, [ 'Healthy', 'RA', ],
                        ['green', 'darkorange', ], datasource_marker, 'RA'),

    'RANoAdap': ('RA', lambda kmer_size: load_kmers(kmer_size = kmer_size, data_sets = [ 'RA_no_adapter']), 3, [ 'Healthy', 'RA', ],
                        ['green', 'darkorange', ], datasource_marker, 'RA_no_adapter'),

    'SingleDiseaseFeng': ('CRC', lambda kmer_size: load_kmers(kmer_size = kmer_size, data_sets = [ 'Feng']), 3, [ 'Healthy', 'CRC', ],
                        ['green', 'blue', ], datasource_marker, 'Feng'),
    

    'ZellerReduced': ('CRC', lambda kmer_size: load_kmers(kmer_size = kmer_size, data_sets = [ 'Zeller_2014']), 3, [ 'Healthy', 'CRC', ],
                        ['green', 'turquoise', ], datasource_marker, 'Zeller_2014'),
    
    'SingleDiseaseLiverCirrhosis': ('LC', lambda kmer_size: load_kmers(kmer_size = kmer_size, data_sets = [ 'LiverCirrhosis']), 3, [ 'Healthy', 'LC', ],
                        ['green', 'fucshia', ], datasource_marker, 'LiverCirrhosis'),

    'KarlssonReduced': ('T2D', lambda kmer_size: load_kmers(kmer_size = kmer_size, data_sets = [ 'Karlsson_2013']), 3, [ 'Healthy', 'T2D', ],
                        ['green', 'brown', ], datasource_marker, 'Karlsson_2013'),

    'KarlssonNoAdap': ('T2D', lambda kmer_size: load_kmers(kmer_size = kmer_size, data_sets = [ 'Karlsson_2013_no_adapter']), 3, [ 'Healthy', 'T2D', ],
                       ['green', 'brown', ], datasource_marker, 'Karlsson_2013_no_adapter'),

    'LeChatelier': ('Obese', lambda kmer_size: load_kmers(kmer_size = kmer_size, data_sets = [ 'LeChatelier']), 3, [ 'Healthy', 'Obese', ],
                        ['green', 'darkbrown', ], datasource_marker, 'LeChatelier'),

    'T2D-HMP': ('T2D', lambda kmer_size: load_kmers(kmer_size = kmer_size, data_sets = [ 'Qin_et_al', 'Karlsson_2013', 'HMP']), 3, [ 'Healthy', 'T2D', ],
                ['green', 'purple', ], datasource_marker),

    'RA-HMP': ('RA', lambda kmer_size:  load_kmers(kmer_size = kmer_size, data_sets = [ 'RA', 'HMP']), 3, [ 'Healthy', 'RA', ],
                ['green', 'darkorange', ], datasource_marker, None),
    'All-T2D': ('T2D', lambda kmer_size: load_kmers(kmer_size = kmer_size, data_sets = [ 'Karlsson_2013', 'Qin_et_al']), 3, [ 'Healthy', 'T2D', ],
            ['green', 'purple', ], datasource_marker, None),
    'All-CRC': ('CRC', lambda kmer_size: load_kmers(kmer_size = kmer_size, data_sets = [ 'Feng', 'Zeller_2014']), 3, [ 'Healthy', 'CRC', ],
            ['green', 'blue', ], datasource_marker, None)
    
}

# Specific model configs per dataset - used to evaluate individual exp configs after grid search
# Items are of the form <dataset_name>: [ <config_info1>, <config_info2>, ...] - config info is the key/value list returned by config_info() 
# that's embedded in plot file names and printed in log files.
#
# When a datasource has a corresponding entry in this dict, its config from this map overrides what's in exp_configs
dataset_configs = {}
dataset_k_fold = {}

# Normalization per sample to be experimented with
norm_func_dict = {
    "L1": norm_by_l1,
    "L2": norm_by_l2 
}
# For denoting code/decoding/output layer activation functions the same as the encoding activation function
# - used for testing whether the data has nonlinear structure.
SAME_AS_ENC = "asenc"

def set_config(exp_config, config_key, val):
    exp_config[config_key] = val

def set_dataset_exp_config(exp_config, dataset):
    exp_config[dataset_key] = dataset

# set global variables at the beginning of the file, for use in loading the data and creating one-hot
# encoding vectors
def set_config_global_vars(dataset):
    global source_ind, class_name, label_ind, classes, n_classes, class_to_ind, colors, markers, data_loader, dataset_load_name
    cls_name, data_load, lbl_ind, lbls, clrs, mkrs, d_name = dataset_dict[dataset]
    class_name = cls_name
    label_ind = lbl_ind
    classes = lbls
    n_classes = len(classes)
    class_to_ind = { classes[i]: i for i in range(n_classes) }
    colors = clrs
    markers = mkrs
    data_loader = data_load
    dataset_load_name = d_name

# The experiment config currently being trained/evaluated
global exp_config
exp_config = {}

# The training/test input and label matrices
x_train, y_train, info_train, x_test, y_test, info_test = None, None, None, None, None, None

# Turn on/off randomness in training.
def setup_randomness(no_random):
    if not no_random:
        return

    #print("Removing randomness")
    # From Keras FAQs: https://keras.io/getting-started/faq/

    # The below is necessary in Python 3.2.3 onwards to
    # have reproducible behavior for certain hash-based operations.
    # See these references for further details:
    # https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
    # https://github.com/fchollet/keras/issues/2280#issuecomment-306959926

    os.environ['PYTHONHASHSEED'] = '0'

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.

    np.random.seed(123)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.

    rn.seed(12345)

    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of
    # non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

    tf.set_random_seed(1234)

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

# Adds activation after a dense layer, and optionally adds batch normalization
# as well as dropout
def add_layer_common(autoencoder, activation):
    autoencoder.add(Activation(activation))
    if exp_config[batch_norm_key]:
        autoencoder.add(BatchNormalization())
    if not exp_config[dropout_pct_key] == 0:
        autoencoder.add(Dropout(float(exp_config[dropout_pct_key])))

# Maps class name to target vector, which is zero everywhere except 1 at the index of the true label
def class_to_target(cls):
    target = np.zeros(n_classes)
    target[class_to_ind[cls]] = 1.0
    return target

# Returns and stores the indices of train and test data for k-fold cross-validation given dataset, number of iterations/folds,
# kmer size, use of randomness, and whether to use nulls
def get_k_folds(dataset, num_iters, use_kfold, no_random, kmer_size, shuffle_labels, shuffle_abunds):
    if (dataset, num_iters, use_kfold, no_random, kmer_size, shuffle_labels, shuffle_abunds) in dataset_k_fold.keys():
        return dataset_k_fold[(dataset, num_iters, use_kfold, no_random, kmer_size, shuffle_labels, shuffle_abunds)]

    set_config_global_vars(dataset)
    x_y_info_train_test = []

    if dataset_load_name == None:
        data, orig_target = data_loader(kmer_size)
    else:
        data, orig_target = get_dataset(dataset_load_name, kmer_size)
    print("LOADED DATASET " + str(dataset) + ": " + str(len(data)) + " SAMPLES")
    orig_target = np.array(orig_target)

    for i in range(num_iters):
        # This is to optionally make the shuffle/fold deterministic
        setup_randomness(no_random)
        
        if (get_config_val(shuffle_labels_key, exp_config)):
            orig_target = np.copy(orig_target)
            np.random.shuffle(orig_target)
        elif (get_config_val(shuffle_abunds_key, exp_config)):
            data = np.copy(data)
            for r in data:
                np.random.shuffle(r)
            
        target = np.array([class_to_target(orig_target[i][label_ind]) for i in range(len(orig_target))])
        
        # Temp labels just for stratified shuffle split - we split proportionally by datasource+label+health
        # the 0 index is for diseased or not 
        target_labels = [(str(orig_target[i][0]) + orig_target[i][label_ind] + orig_target[i][source_ind]) for i in range(len(orig_target))]

        skf = None
        if use_kfold:
            # K folds - we shuffle only if not removing randomness
            skf = StratifiedKFold(n_splits=use_kfold, shuffle=(not no_random))
            skf = skf.split(data, target_labels)
        else:
            # single fold - the random state is fixed with the value of the fold index if no randomness,
            # otherwise np.random will be used by the function.
            skf = StratifiedShuffleSplit(target_labels, n_iter=1, test_size=test_pct, random_state=(i if no_random else None))

        x_y_info_train_test_it = []

        x_y_info_train_test_it.append([data, target, orig_target])

        # the current active fold
        # This loop is executed K times for K folds and only once for single split
        for train_idx, test_idx in skf:
            x_y_info_train_test_it.append([train_idx, test_idx])
        x_y_info_train_test.append(x_y_info_train_test_it)
    dataset_k_fold[(dataset, num_iters, use_kfold, no_random, kmer_size, shuffle_labels, shuffle_abunds)] = x_y_info_train_test
    return x_y_info_train_test

# Loops over all configs in grid search and executes each
# Stores each config in exp_config
def loop_over_configs(random=False):
    for config in ConfigIterator(random=random):
        global exp_config
        exp_config = config
        
        k_folds = get_k_folds(exp_config[dataset_key], exp_config[num_iters_key], exp_config[use_kfold_key],
                              exp_config[no_random_key], exp_config[kmer_size_key], exp_config[shuffle_labels_key],
                              exp_config[shuffle_abunds_key])
        
        exec_config(k_folds)


# Loops over certain supervised models in dataset_configs and executes each
# Stores each config in exp_config
def loop_over_dataset_configs():
    for dataset_name in dataset_configs:
        for config in dataset_configs[dataset_name]:
            setup_exp_config_from_config_info(config)
            set_dataset_exp_config(exp_config, dataset_name)
            k_folds = get_k_folds(exp_config[dataset_key], exp_config[num_iters_key], exp_config[use_kfold_key],
                                  exp_config[no_random_key], exp_config[kmer_size_key], exp_config[shuffle_labels_key],
                                  exp_config[shuffle_abunds_key])
            exec_config(k_folds)

dataset_kmer_labels = {}

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

def get_dataset(dataset_name, kmer_size):
    if (dataset_name, kmer_size) in dataset_kmer_labels:
        return dataset_kmer_labels[(dataset_name, kmer_size)]
    dataset_kmer_labels[(dataset_name, kmer_size)] = load_kmers(kmer_size, [dataset_name])
    return dataset_kmer_labels[(dataset_name, kmer_size)]

# Executes the config given training/test splits            
def exec_config(k_folds):
    global dataset_config_iter_fold_results
    config_results = {}
    
    set_config_global_vars(exp_config[dataset_key])
    x_y_info_train_test = k_folds
    dataset_config_iter_fold_results[exp_config[dataset_key]] = {}

    if log_iter_feats:
        kmers_no_comp = []
        print("GENERATING ALL KMERS CAPS")
        all_kmers_caps = [''.join(_) for _ in product(['A', 'C', 'G', 'T'], repeat = exp_config[kmer_size_key])]
        print("GENERATED ALL KMERS CAPS")
        for kmer in all_kmers_caps:
            if get_reverse_complement(kmer) not in kmers_no_comp:
                kmers_no_comp.append(kmer)
        
    for i in range(exp_config[num_iters_key]):
        exp_config[iter_key] = i
        exp_config[kfold_key] = 0
        x_y_info_train_test_it  = x_y_info_train_test[i]
        data, target, orig_target = x_y_info_train_test_it[0]
        for (train_idx, test_idx) in x_y_info_train_test_it[1:]:
            global x_train, y_train, info_train, x_test, y_test, info_test
            # set up the training/test sample and target matrices - the info matrix is used for plotting
            x_train, y_train, info_train = data[train_idx], target[train_idx], orig_target[train_idx]
            x_test, y_test, info_test = data[test_idx], target[test_idx], orig_target[test_idx]

            normalize_train_test()

            # the next fold
            exp_config[kfold_key] += 1

        # aggregate results across K folds for this iteration
        global config_iter_fold_results
        config_iter_fold_results = dataset_config_iter_fold_results[exp_config[dataset_key]]
        for config in config_iter_fold_results:
            # K-fold results
            fold_results = np.array(config_iter_fold_results[config][exp_config[iter_key]])

            # the config for this iteration
            config_iter = config + '_' + iter_key + ':' + str(exp_config[iter_key])
            # sum the confusion matrices over the folds
            conf_mat = np.sum(fold_results[:, 2], axis=0)
            shap_vals = np.sum(fold_results[:, 8], axis=0)

            # mean loss across K folds for this iteration
            loss = np.mean([fold_results[k, 0]['loss'] for k in range(len(fold_results))], axis=0)
            val_loss = np.mean([fold_results[k, 0]['val_loss'] for k in range(len(fold_results))], axis=0)

            # mean accuracy across K folds for this iteration
            acc = np.mean([fold_results[k, 0]['acc'] for k in range(len(fold_results))], axis=0)
            val_acc = np.mean([fold_results[k, 0]['val_acc'] for k in range(len(fold_results))], axis=0)

            # ending loss/acc and max validation acc/min validation loss indices - used for testing early stopping
            last_val_acc = val_acc[-1]
            last_val_loss = val_loss[-1]
            max_vai = np.argmax(val_acc)
            min_vli = np.argmin(val_loss)

            # log the results for offline analysis
            print(('{}\t{}\t{}\t{}\t{}\t{}\tkfold-iter-val-loss-acc for ' + class_name + ':{}').
                  format(last_val_acc, last_val_loss, max_vai, min_vli, val_acc[max_vai], val_loss[min_vli], config_iter))

            # save the results for this iteration so we can aggregate the overall results at the end
            if not config in config_results:
                config_results[config] = []
                
            # hist, [codes_pred, info_test], conf_mat, [fpr, tpr, roc_auc], classes, [y_test, y_test_pred],
            # [accuracy, f1, precision, recall, roc_auc[1], roc_auc['macro']], [ae_hist, ae_val_loss, ae_mse]
            config_results[config].append([val_acc, acc, val_loss, loss, conf_mat, fold_results[:, 3], fold_results[:, 5], fold_results[:, 6], fold_results[:, 7], shap_vals])

            print("DUMPING FEATURE IMPORTANCES FOR ITERATION " + str(i))
            if log_iter_feats:
                num_features = num_feature_imps
                if (num_features == -1):
                    num_features = len(shap_vals)
                file = open(graph_dir + "/feat_imps_" + name_file_from_config(exp_config, skip_keys=[kfold_key]) + ".txt", "w")
                file.write("Importances\tfor\t" + str(exp_config[dataset_key]) + "\t" + name_file_from_config(exp_config, skip_keys=[kfold_key]) + "\n")
                for j in range(num_features):
                    if shap_vals[j] > 0:
                        file.write(kmers_no_comp[j] + "\t" + str(shap_vals[j]/(len(y_test) + len(y_train))) + "\n")
                print("END FEATURE IMPORTANCE DUMP " + str(i))
                file.close()


            # Per-iteration plots across K folds
            if plot_iter:
                # plot the confusion matrix
                plot_confusion_matrix(conf_mat, config=config_iter)
                # plot loss vs epochs
                plot_loss_vs_epochs(loss, val_loss, config=config_iter)

                # plot accuracy vs epochs
                plot_acc_vs_epochs(acc, val_acc, config=config_iter)
    
    for config in config_results:
        # per iteration results
        iter_results = np.array(config_results[config])

        # sum the confusion matrices over iterations
        conf_mat = np.sum(iter_results[:, 4], axis=0)
        shap_vals = np.sum(iter_results[:, 9], axis=0)
        
        # mean results for loss/val loss/acc/val acc across iterations
        mean_results = np.mean(iter_results[:, [0,1,2,3]], axis=0)
            
        # mean loss across iterations
        loss = mean_results[3]
        val_loss = mean_results[2]

        # mean accuracy across iterations
        acc = mean_results[1]
        val_acc = mean_results[0]

        last_val_acc = val_acc[-1]
        last_val_loss = val_loss[-1]

        max_vai = np.argmax(val_acc)
        min_vli = np.argmin(val_loss)

        all_fold_roc_aucs = np.concatenate(iter_results[:, 5])
        all_y_test_pred = np.concatenate(iter_results[:, 6])
        
        all_y_test = np.array(np.concatenate([r[0] for r in all_y_test_pred]))
        all_y_pred = np.array(np.concatenate([r[1] for r in all_y_test_pred]))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        accs = dict()
        std_down = dict()
        std_up = dict()
        for i in range(n_classes):
            true_probs = [r[i] for r in all_y_test]
            pred_probs = [r[i] for r in all_y_pred]
            fpr[i], tpr[i], _ = roc_curve(true_probs, pred_probs)
            roc_auc[i] = auc(fpr[i], tpr[i])
            accs[i] = accuracy_score(np.round(true_probs), np.equal(np.argmax(all_y_pred, axis=1), i))

        fpr["micro"], tpr["micro"], _ = roc_curve(all_y_test.ravel(), all_y_pred.ravel())
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
            std_down[i] = np.maximum(tpr[i] - np.std(i_tpr[i], axis=0)*plot_factor/np.sqrt(get_config_val(use_kfold_key, exp_config)), 0)
            std_up[i] = np.minimum(tpr[i] + np.std(i_tpr[i], axis=0)*plot_factor/np.sqrt(get_config_val(use_kfold_key, exp_config)), 1.0)

        precision_graph, recall_graph, _ = precision_recall_curve(all_y_test[:,1], all_y_pred[:, 1])

        if plot_overall:
            # plot the confusion matrix
            plot_confusion_matrix(conf_mat, config=config)

            # plot loss vs epochs
            plot_loss_vs_epochs(loss, val_loss, config=config)
        
            # plot accuracy vs epochs
            plot_acc_vs_epochs(acc, val_acc, config=config)

            # plot the ROCs with AUCs/ACCs
            plot_roc_aucs(fpr, tpr, roc_auc, accs, std_down, std_up, config=config)

            if num_feature_imps != 0:
                print("SORTING FEATURE IMPORTANCES")
                num_features = num_feature_imps
                if (num_features == -1):
                    num_features = len(shap_vals)
                indices = np.argsort(shap_vals)[::-1][0:num_features]
                shap_vals = shap_vals[indices]
                kmers_no_comp = []
                all_kmers_caps = [''.join(_) for _ in product(['A', 'C', 'G', 'T'], repeat = exp_config[kmer_size_key])]
                for kmer in all_kmers_caps:
                    if get_reverse_complement(kmer) not in kmers_no_comp:
                        kmers_no_comp.append(kmer)
                kmers_no_comp = [kmers_no_comp[i] for i in indices]
                file = open(graph_dir + "/feat_imps_" + name_file_from_config(exp_config, skip_keys=[iter_key, kfold_key]) + ".txt", "w")
                file.write("Importances\tfor\t" + str(exp_config[dataset_key]) + "\t" + name_file_from_config(exp_config, skip_keys=[iter_key, kfold_key]) + "\n")
                for i in range(num_features):
                    if shap_vals[i] > 0:
                        file.write(kmers_no_comp[i] + "\t" + str(shap_vals[i]/(len(all_y_test))) + "\n")
                print("END FEATURE IMPORTANCE DUMP")
                file.close()

        '''
        print('kfold-overall-loss: ' + str(loss) + ' for ' + class_name + ':' + config)
        print('kfold-overall-val-loss: ' + str(val_loss) + ' for ' + class_name + ':' + config)
        print('kfold-overall-acc: ' + str(acc) + ' for ' + class_name + ':' + config)
        print('kfold-overall-val-acc: ' + str(val_acc) + ' for ' + class_name + ':' + config)
        print('kfold-overall-conf-mat: ' + str(conf_mat) + ' for ' + class_name + ':' + config)
        '''
        
        perf_metrics = np.vstack(np.concatenate(iter_results[:, 7]))
        perf_means = np.mean(perf_metrics, axis=0)
        perf_stds = np.std(perf_metrics, axis=0)

        if plot_overall:
            plot_precision_recall(precision_graph, recall_graph, perf_means[2], perf_means[1], config = config)
        
        # log the results for offline analysis
        print(('{}({})\t{}({})\t{}({})\t{}({})\t{}({})\t{}({})\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\tkfold-overall-perf-metrics for ' + class_name + ':{}').
              format(perf_means[0], perf_stds[0], perf_means[1], perf_stds[1], perf_means[2], perf_stds[2], perf_means[3], perf_stds[3], perf_means[4], perf_stds[4], perf_means[5], perf_stds[5], roc_auc[1], roc_auc['macro'], last_val_acc, last_val_loss, max_vai, min_vli, val_acc[max_vai], val_loss[min_vli], config))

        if get_config_val(use_ae_key, exp_config):           
            all_ae_results = np.vstack(np.concatenate(iter_results[:, 8]))
            # mean autoencoder loss vs epochs across iterations times K folds
            ae_loss = np.mean([ r[0]['loss'] for r in all_ae_results ], axis=0)
            ae_val_loss = np.mean([ r[0]['val_loss'] for r in all_ae_results ], axis=0)
            if plot_ae_overall:
                # plot loss vs epochs
                plot_loss_vs_epochs(ae_loss, ae_val_loss, config=config, name='ae_loss')
            ae_means = np.mean([ [r[1], r[2]] for r in all_ae_results ], axis=0)
            ae_stds = np.std([ [r[1], r[2]] for r in all_ae_results ], axis=0)
            ae_aggr_results = [ae_loss, ae_val_loss, ae_means, ae_stds]
            print(('{}({})\t{}({})\tkfold-overall-autoencoder-perf-metrics for ' + class_name + ':{}').
              format(ae_means[0], ae_stds[0], ae_means[1], ae_stds[1], config))
        else:
            ae_aggr_results = [9999, 9999, 9999, 9999]
        

        # dump the model results into a file for offline analysis and plotting, e.g., merging plots from
        # different model instances (real and null)
        aggr_results = [val_acc, acc, val_loss, loss, conf_mat, fpr, tpr, roc_auc, accs, std_down, std_up, perf_means, perf_stds]

        dataset_info = [source_ind, class_name, label_ind, classes, n_classes, class_to_ind, colors, markers]

        with open(graph_dir + "/aggr_results" + config +".pickle", "wb") as f:
            dump_dict = { "dataset_info": dataset_info, "results": [aggr_results, dataset_config_iter_fold_results[exp_config[dataset_key]][config]],
                          "ae_results": ae_aggr_results
                           }
            print("dumping")
            pickle.dump(dump_dict, f)
            print("finished dumping")
    dataset_config_iter_fold_results = {}

# Performs data normalization before creation, training, and testing of model
def normalize_train_test():
    norm_func = norm_func_dict[exp_config[norm_sample_key]]
    global x_train,  x_test
    x_train = (norm_func)(x_train) 
    x_test = (norm_func)(x_test)

    if exp_config[pca_dim_key] > 0:
        pca = PCA(n_components=exp_config[pca_dim_key])
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)
        pca = None

    elif exp_config[nmf_dim_key] > 0:
        nmf = NMF(n_components=exp_config[nmf_dim_key])
        x_train = nmf.fit_transform(x_train)
        x_test = nmf.transform(x_test)
        nmf = None
    
    layers = exp_config[layers_key]
    # exp with # of epochs for autoencoder training
    build_train_test(layers)

# Counts the number of layers of the autoencoder from output layer to the middle code layer
# - used for popping off the decoding layers after autoencoder training and before 
# supervised training
def cnt_to_coding_layer():
    num_layers = 2 * (len(exp_config[layers_key]) - 1) 
    if exp_config[batch_norm_key]:
        num_layers += len(exp_config[layers_key]) - 1
    if not exp_config[dropout_pct_key] == 0:
        num_layers += len(exp_config[layers_key]) - 1
    if not exp_config[input_dropout_pct_key] == 0:
        num_layers += 1
    return num_layers

def cnt_from_decoding():
    num_layers = 2 * (len(exp_config[layers_key]) - 1)
    if exp_config[batch_norm_key]:
        num_layers += len(exp_config[layers_key]) - 2
    if not exp_config[dropout_pct_key] == 0:
        num_layers += len(exp_config[layers_key]) - 2
    return num_layers

# Creates the supervised learning model based on information in exp_config
def build_train_test(layers):
    global x_train,  x_test
    """
    This uses the current active exp config (set up by the loops above) to first build and train the autoencoder,
    and then build and train/test the supervised model
    """

    # This is to optionally make the weight initialization deterministic
    setup_randomness(exp_config[no_random_key])

    # the autoencoder
    autoencoder = Sequential()
    if (not exp_config[input_dropout_pct_key] == 0):
        autoencoder.add(Dropout(float(exp_config[input_dropout_pct_key]), input_shape=(layers[0],)))

    # layers before the middle code layer - they share the same activation function
    for i in range(1, len(layers)-1):
        if not exp_config[max_norm_key] == 0:
            autoencoder.add(Dense(layers[i], input_dim=layers[i-1], kernel_constraint=max_norm(exp_config[max_norm_key]), kernel_regularizer=regularizers.l2(exp_config[act_reg_key]),
                                  activity_regularizer=regularizers.l1(exp_config[act_reg_key])))
        else:
            autoencoder.add(Dense(layers[i], input_dim=layers[i-1], kernel_regularizer=regularizers.l2(exp_config[act_reg_key]),
                                  activity_regularizer=regularizers.l1(exp_config[act_reg_key])))
        add_layer_common(autoencoder, exp_config[enc_act_key])


    # the middle code layer - it has an activation function independent of other encoding layers
    i = len(layers) - 1

    if not exp_config[max_norm_key] == 0:
        autoencoder.add(Dense(layers[i], input_dim=layers[i-1], kernel_constraint=max_norm(exp_config[max_norm_key]), kernel_regularizer=regularizers.l2(exp_config[act_reg_key]),
                                activity_regularizer=regularizers.l1(exp_config[act_reg_key])))
    else:
        autoencoder.add(Dense(layers[i], input_dim=layers[i-1], kernel_regularizer=regularizers.l2(exp_config[act_reg_key]),
                              activity_regularizer=regularizers.l1(exp_config[act_reg_key])))
    add_layer_common(autoencoder, exp_config[enc_act_key if exp_config[code_act_key] == SAME_AS_ENC else code_act_key])


    # the model from input to the code layer - used for plotting 2D graphs
    code_layer_model = Model(inputs=autoencoder.layers[0].input,
                                 outputs=autoencoder.layers[cnt_to_coding_layer() - 1].output)
    
    print("CODE LAYER MODEL FOR DIMENSIONALITY REDUCTION")
    code_layer_model.summary()

    # stats for the autoencoder - the 9s are for when autoencoder is (optionally) not used before supervised learning
    ae_val_loss, ae_loss, ae_mse = 999999, 999999, 999999
    ae_hist = []

    norm_input = exp_config[norm_input_key]
    if norm_input and (exp_config[after_ae_layers_key] is None or len(exp_config[after_ae_layers_key]) == 0):
        sample_mean = x_train.mean(axis=0)
        sample_std = x_train.std(axis=0)
        # Normalize both training and test samples with the training mean 
        x_train = (x_train - sample_mean)
        # test samples are normalized using only the mean of the training samples
        x_test = (x_test - sample_mean)

        if norm_input != 2:
            # Normalize both training and test samples with the training std
            x_train = x_train / sample_std
            # test samples are normalized using only the std of the training samples
            x_test = x_test / sample_std
        
    
    if get_config_val(use_ae_key, exp_config):
        print("AUTOENCODER")
        ae_datasets = exp_config[ae_datasets_key]

        # Use the autoencoder before supervised learning
        #
        # Add the decoding layers - from second to last layer to the second layer
        # because the last one is the "code" layer, which has no decoding counterpart.
        for i in range(len(layers)-2, 0, -1):
            autoencoder.add(Dense(layers[i], input_dim=layers[i+1]))
            add_layer_common(autoencoder, exp_config[enc_act_key if exp_config[dec_act_key] == SAME_AS_ENC else dec_act_key])

        # the last decoding layer - it has an activation function independent of other decoding layers
        i = 0
        autoencoder.add(Dense(layers[i], input_dim=layers[i+1]))
        autoencoder.add(Activation(exp_config[enc_act_key if exp_config[out_act_key] == SAME_AS_ENC else out_act_key]))
        autoencoder.compile(optimizer='adadelta', loss=exp_config[loss_func_key])
        print("FULL AUTOENCODER")
        autoencoder.summary()
            
        if auto_weights_file:
            print("LOADING AUTO WEIGHTS")
            autoencoder.load_weights(os.path.expanduser('~') + '/deep_learning_microbiome/scripts/' + auto_weights_file)
            ae_hist = {}
            keys = ['loss', 'val_loss']
            for key in keys:
                ae_hist[key] = [999999]

        else:
            print("TRAINING")
            datasets = list(ae_datasets)
            for i in range(len(ae_datasets)):
                datasets[i] = dataset_name_dict[ae_datasets[i]]

            kmers_train = None
            for i in range(len(datasets)):
                kmer_train, labels = get_dataset(datasets[i], exp_config[kmer_size_key])
                norm_func = norm_func_dict[exp_config[norm_sample_key]]
                kmer_train = (norm_func)(kmer_train)
                if i == 0:
                    kmers_train = kmer_train
                else:
                    kmers_train = np.concatenate((kmer_train, kmers_train))

            if not kmers_train is None:
                autoencoder.fit(kmers_train, kmers_train,
                                epochs=exp_config[auto_epochs_key],
                                batch_size=exp_config[batch_size_key],
                                # shuffle the training samples per epoch only if removing randomness is not specified
                                shuffle=(not get_config_val(no_random_key, exp_config)),
                                validation_data=None,
                                verbose=0)        

            history = History()
            
            callbacks = [history]
            # Support for early stopping - not used because Keras is too sensitive.
            if exp_config[early_stop_key]:
                callbacks.append(EarlyStopping(monitor='val_loss', patience=exp_config[patience_key], verbose=0))
            
            # train
            autoencoder.fit(x_train, x_train,
                            epochs=exp_config[auto_epochs_key],
                            batch_size=exp_config[batch_size_key],
                            # shuffle the training samples per epoch only if removing randomness is not specified
                            shuffle=(not get_config_val(no_random_key, exp_config)),
                            validation_data=(x_test, x_test),
                            verbose=0,
                            callbacks=callbacks)
            ae_hist = history.history
        if save_weights_auto:
            print("SAVING AUTO WEIGHTS")
            filename = "ae_weights_" + name_file_from_config(exp_config) + ".h5"
            autoencoder.save_weights(filename)
        
        # Plot training/test loss by epochs for the autoencoder
        if plot_ae_fold:
            plot_loss_vs_epochs(ae_hist['loss'], ae_hist['val_loss'], name='ae_loss')

        # Calculate the MSE on the test samples
        decoded_output = autoencoder.predict(x_test)
        ae_mse = mean_squared_error(decoded_output, x_test)

        ae_val_loss = ae_hist['val_loss'][-1]
        ae_loss = ae_hist['loss'][-1]

        # Which epoch achieved min loss - useful for manual early stopping
        min_vli = np.argmin(ae_hist['val_loss'])
        min_li = np.argmin(ae_hist['loss'])

        

        # Log the autoencoder stats/performance
        print('{},{},{}\t{},{},{}\t{}\t{}\t{}\tautoencoder-min-loss-general:{}'.
              format(ae_hist['val_loss'][min_vli], min_vli, ae_hist['loss'][min_vli],
                     ae_hist['loss'][min_li], min_li, ae_hist['val_loss'][min_li],
                     ae_val_loss, ae_loss, ae_mse, config_info(exp_config)))

        # if plot_fold and exp_config[enc_dim_key] > 1:
        #     fig = pylab.figure()
        #     pylab.title('AE2 VS AE1 After Autoencoder Training', size=plot_title_size)
        #     pylab.xlabel('AE1', size=plot_text_size)
        #     pylab.ylabel('AE2', size=plot_text_size)

        #     codes_pred = code_layer_model.predict(x_test)
        #     print("Autoencoder's test code shape for dimension {}: {}".format(exp_config[enc_dim_key], codes_pred.shape))
        #     plot_2d_codes(codes_pred, fig, 'ae_two_codes',
        #                   "This graph shows autoencoder's first 2 components of the encoding codes (the middle hidden layer output) with labels")

        # Pop the decoding layers before supervised learning
        
        for n in range(cnt_from_decoding()):
            autoencoder.pop()
        print("POPPED OFF DECODING LAYERS")
        autoencoder.summary()

        if exp_config[after_ae_layers_key] is not None and len(exp_config[after_ae_layers_key]) > 0:
            x_train = code_layer_model.predict(x_train)
            x_test = code_layer_model.predict(x_test)

            if norm_input and (exp_config[after_ae_layers_key] is None or len(exp_config[after_ae_layers_key]) == 0):
                sample_mean = x_train.mean(axis=0)
                sample_std = x_train.std(axis=0)
                # Normalize both training and test samples with the training mean 
                x_train = (x_train - sample_mean)
                # test samples are normalized using only the mean of the training samples
                x_test = (x_test - sample_mean)

                if norm_input != 2:
                    # Normalize both training and test samples with the training std
                    x_train = x_train / sample_std
                    # test samples are normalized using only the std of the training samples
                    x_test = x_test / sample_std

            print("CREATING NEW SUPERVISED MODEL USING AUTOENCODER OUTPUT AS INPUTS")
            autoencoder = None
            autoencoder = Sequential()
            autoencoder.add(Dense(exp_config[after_ae_layers_key][0], input_dim=exp_config[enc_dim_key]))
            autoencoder.add(Activation(exp_config[after_ae_act_key]))
            for dim in range(1, len(exp_config[after_ae_layers_key])):
                autoencoder.add(Dense(dim))
                autoencoder.add(Activation(exp_config[after_ae_act_key]))
            code_layer_model = Model(inputs=autoencoder.layers[0].input,
                             outputs=autoencoder.layers[-1].output)
            print("CODE LAYER MODEL FOR PLOTTING 2D CODES")
            code_layer_model.summary()
            
   
    
    # Add the classification layer for supervised learning - note that the loss function is always
    # categorical cross entropy - not what's in exp_configs, which is only used for the autoencoder.
    autoencoder.add(Dense(n_classes))
    autoencoder.add(Activation('softmax'))
    autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

    history = History()
    callbacks = [history]
    
    if exp_config[early_stop_key]:
        callbacks.append(EarlyStopping(monitor='val_acc', patience=exp_config[patience_key], verbose=0))

    if super_weights_file:
        print("LOADING SUPER WEIGHTS")
        autoencoder.load_weights(os.path.expanduser('~') + '/deep_learning_microbiome/scripts/' + super_weights_file)
        keys = ['val_acc', 'acc', 'val_loss', 'loss']
        hist = {}
        for key in keys:
            hist[key] = [999999]
    
    else:
        # Train
        weight = exp_config[class_weight_key]
        class_weight = None
        if weight != 0:
            class_weight = {}
            class_weight[0] = 1.0
            class_weight[1] = float(weight)
        autoencoder.fit(x_train, y_train,
                        epochs=exp_config[super_epochs_key],
                        batch_size=exp_config[batch_size_key],
                        shuffle=(not get_config_val(no_random_key, exp_config)),
                        verbose=0,
                        validation_data=(x_test, y_test),
                        class_weight=class_weight,
                        callbacks=callbacks)

        hist = history.history

    
    print("SUPERVISED MODEL WITH CLASSIFICATION LAYER")
    autoencoder.summary()

    if save_weights_super:
        print("SAVING SUPER WEIGHTS")
        filename = "super_weights_" + name_file_from_config(exp_config) + ".h5"
        autoencoder.save_weights(filename)


    # plot the 2D codes for the test samples
    codes_pred = code_layer_model.predict(x_test)
    if plot_fold and exp_config[enc_dim_key] > 1:
        plot_2d_codes(codes_pred)

    y_train_pred = autoencoder.predict(x_train)

    # predictions on the test samples
    y_test_pred = autoencoder.predict(x_test)

    # plot the confusion matrix
    # compare the true label index (with max value (1.0) in the target vector) against the predicted
    # label index (index of label with highest predicted probability)
    conf_mat = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_test_pred, axis=1), labels=range(n_classes))
    if plot_fold:
        plot_confusion_matrix(conf_mat)

    # printing the accuracy rates for diagnostics
    print("Total accuracy for " + str(len(y_test_pred)) + " test samples: " +
          str(np.mean(np.equal(np.argmax(y_test, axis=1), np.argmax(y_test_pred, axis=1)))))

    for cls in classes:
        idx = []
        for i in range(y_test_pred.shape[0]):
            if info_test[i][3] == cls:
                idx.append(i)
        if len(idx) == 0:
            continue
        idx = np.array(idx)
        print("Accuracy for total of " + str(len(idx)) + " " + cls + " samples: " +
              str(np.mean(np.equal(np.argmax(y_test[idx], axis=1), np.argmax(y_test_pred[idx], axis=1)))))

    # print('Predicted probabilities for training samples: ' +
    #       str([ (y_train[i], [y_train_pred[i][j] for j in range(y_train.shape[1]) ]) for i in range(len(y_train)) ]))
    # print('Predicted probabilities for testing samples: ' +
    #       str([ (y_test[i], [y_test_pred[i][j] for j in range(y_train.shape[1]) ]) for i in range(len(y_test)) ]))

    # Compute ROC curve and AUC for each class - http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    acc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_test_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        # Round float 1.0 to integer 1 and 0.0 to 0 in the target vectors, and 1 for max predicted prob
        # index being this one (i), 0 otherwise
        acc[i] = accuracy_score(np.round(y_test[:, i]), np.equal(np.argmax(y_test_pred, axis=1), i))
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_test_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    if plot_fold:
        # plot the ROCs with AUCs
        plot_roc_aucs(fpr, tpr, roc_auc, acc)

        # plot loss vs epochs
        plot_loss_vs_epochs(hist['loss'], hist['val_loss'])

        # plot accuracy vs epochs
        plot_acc_vs_epochs(hist['acc'], hist['val_acc'])

    # log the performance stats for sorting thru the grid search results
    ae_super_val_acc = hist['val_acc'][-1]
    ae_super_acc = hist['acc'][-1]
    ae_super_val_loss = hist['val_loss'][-1]
    ae_super_loss = hist['loss'][-1]

    max_vai = np.argmax(hist['val_acc'])
    max_ai = np.argmax(hist['acc'])
    min_vli = np.argmin(hist['val_loss'])
    min_li = np.argmin(hist['loss'])

    avg_perf = hist['val_acc'][max_vai] + hist['acc'][max_vai] + hist['acc'][max_ai] + hist['val_acc'][max_ai] + hist['val_acc'][min_vli] + hist['acc'][min_vli] +\
    hist['val_acc'][min_li] + hist['acc'][min_li] + ae_super_val_acc + ae_super_acc
    for i in [ 1, 'macro' ]:
        avg_perf += roc_auc[i]
    avg_perf /= 12

    print(('{}\t{},{},{},{},{}\t{},{},{},{},{}\t{},{},{},{},{}\t{},{},{},{},{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\tautoencoder-super-min-loss-max-acc for ' + class_name + ':{}').
          format(avg_perf, hist['val_acc'][max_vai], max_vai, hist['acc'][max_vai], hist['val_loss'][max_vai], hist['loss'][max_vai],
                 hist['acc'][max_ai], max_ai, hist['val_acc'][max_ai], hist['val_loss'][max_ai], hist['loss'][max_ai],
                 hist['val_loss'][min_vli], min_vli, hist['val_acc'][min_vli], hist['acc'][min_vli], hist['loss'][min_vli],
                 hist['loss'][min_li], min_li, hist['val_acc'][min_li], hist['acc'][min_li], hist['val_loss'][min_li],
                 ae_super_val_acc, ae_super_acc, ae_super_val_loss, ae_super_loss,
                 ae_val_loss, ae_loss, ae_mse, roc_auc[1], roc_auc['macro'], config_info(exp_config)))
    # calculate the accuracy/f1/precision/recall for this test fold - same way as in Pasolli
    test_true_label_inds = np.argmax(y_test, axis=1)
    test_pred_label_inds = np.argmax(y_test_pred, axis=1)
    accuracy = accuracy_score(test_true_label_inds, test_pred_label_inds)
    f1 = f1_score(test_true_label_inds, test_pred_label_inds, pos_label=None, average='weighted')
    precision = precision_score(test_true_label_inds, test_pred_label_inds, pos_label=None, average='weighted')
    recall = recall_score(test_true_label_inds, test_pred_label_inds, pos_label=None, average='weighted')

    print(('{}\t{}\t{}\t{}\t{}\t{}\tfold-perf-metrics for ' + class_name + ':{}').
          format(accuracy, f1, precision, recall, roc_auc[1], roc_auc['macro'], config_info(exp_config)))

    if num_feature_imps != 0:
        # select a set of background examples to take an expectation over
        background = np.zeros((1, autoencoder.layers[0].batch_input_shape[1]))
        e = shap.DeepExplainer(autoencoder, background)
        shap_values = e.shap_values(x_test)
        shap_values = np.sum(np.absolute(shap_values[1]), axis=0)

            
    else:
        shap_values = []

    # the config info for this exp but no fold/iter indices because we need to aggregate stats over them
    config_key = name_file_from_config(exp_config, skip_keys=[kfold_key, iter_key])
    global config_iter_fold_results
    config_iter_fold_results = dataset_config_iter_fold_results[exp_config[dataset_key]]
    if config_key not in config_iter_fold_results:
        config_iter_fold_results[config_key] = []
        
    # the iteration and fold indices
    iter_ind = exp_config[iter_key]
    fold_ind = exp_config[kfold_key]
    # extend the list for the iteration if necessary
    if len(config_iter_fold_results[config_key]) <= iter_ind:
        config_iter_fold_results[config_key].append([])
    # extend the list for the fold if necessary
    if len(config_iter_fold_results[config_key][iter_ind]) <= fold_ind:
        config_iter_fold_results[config_key][iter_ind].append([])
        
    config_iter_fold_results[config_key][iter_ind][fold_ind] = [hist, [codes_pred, info_test], conf_mat, [fpr, tpr, roc_auc], classes, [y_test, y_test_pred],
                                                                    [accuracy, f1, precision, recall, roc_auc[1], roc_auc['macro']], [ae_hist, ae_val_loss, ae_mse], shap_values]
        
            

    
# Plot the 2D codes in the layer right before the final classification layer
def plot_2d_codes(codes, name='two_codes', title='2D Codes Before Classfication Layer', 
                  xlabel='Code 1', ylabel='Code 2',  desc='', config=None, info=None):
    if info is None:
        info = info_test
    fig = pylab.figure()
    pylab.title(title, size=plot_title_size)
    pylab.xlabel(xlabel, size=plot_text_size)
    pylab.ylabel(ylabel, size=plot_text_size)
    pylab.tick_params(axis='both', labelsize=plot_text_size)

    handles = {}
    for i in range(len(codes)):
        src = info[i][source_ind]
        ind = class_to_ind[info[i][label_ind]]
        h = pylab.scatter([codes[i][0]], [codes[i][1]], marker=markers[src], color=colors[ind], facecolor='none')
        handles[src + ' - ' + info[i][label_ind]] = h
    keys = [ k for k in handles.keys() ]
    keys.sort()
    pylab.legend([handles[k] for k in keys], keys, prop={'size': plot_text_size})
    pylab.gca().set_position((.1, .7, 2.4, 1.8))
    add_figtexts_and_save(fig, name, desc, config=config)

# plot loss vs epochs
def plot_loss_vs_epochs(loss, val_loss, name='super_loss', title='Loss vs Epochs', 
                        xlabel='Epoch', ylabel='Loss', desc='', config=None):
    fig = pylab.figure()
    pylab.plot(loss)
    pylab.plot(val_loss)
    pylab.legend(['train ' + str(loss[-1]), 'test ' + str(val_loss[-1])], prop={'size': plot_text_size})
    pylab.title(title, size=plot_title_size)
    pylab.xlabel(xlabel, size=plot_text_size)
    pylab.ylabel(ylabel, size=plot_text_size)
    pylab.gca().set_position((.1, .7, .8, .6))
    add_figtexts_and_save(fig, name, desc, config=config)

# plot accuracy vs epochs
def plot_acc_vs_epochs(acc, val_acc, name='acc', title='Accuracy vs Epochs',
                       xlabel='Epoch', ylabel='Accuracy', desc='', config=None):
    fig = pylab.figure()
    pylab.plot(acc)
    pylab.plot(val_acc)
    pylab.legend(['train ' + str(acc[-1]), 'test ' + str(val_acc[-1])], loc='lower right', prop={'size': plot_text_size})
    pylab.title(title, size=plot_title_size)
    pylab.xlabel(xlabel, size=plot_text_size)
    pylab.ylabel(ylabel, size=plot_text_size)
    pylab.tick_params(axis='both', labelsize=plot_text_size)
    pylab.gca().set_position((.1, .7, .8, .6))
    add_figtexts_and_save(fig, name, desc, config=config)

    
# plot confusion matrix
def plot_confusion_matrix(cm, config=None, cmap=pylab.cm.Reds):
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
    add_figtexts_and_save(fig, 'cm', "Confusion matrix for predicting sample's " + class_name + " status using 5mers", y_off=1.3, config=config)

# plot roc-auc curves
def plot_roc_aucs(fpr, tpr, roc_auc, accs, std_down=None, std_up=None, config=None, name='roc_auc', title='ROC Curves with AUCs/ACCs', 
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

# plot precision-recall curves
def plot_precision_recall(precision, recall, average_precision, f1_score, name ='precision_recall', config = None):
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

# parse config info string back into exp configs - for evaluating models after grid search
def setup_exp_config_from_config_info(config):
    global exp_config
    exp_config = {}
    config_s = re.sub(r'LF:mean_squared_error', 'LF:kullback_leibler_divergence', config)
    changed = (config_s != config)
    cnt1 = 0
    for part in config_s.split('LF:kullback_leibler_divergence'):
        cnt1 += 1
        if not part:
            continue
        fields = part.split('_')
        for f in fields:
            if not f:
                continue
            k, v = f.split(':')
            v = v.split('-')
            cnt2 = 0
            for i in range(len(v)):
                cnt2 += 1
                if v[i].isdigit():
                    v[i] = int(v[i])

            # Network layer dimension specs are list of list of integers
            exp_config[k] = v[0] if (cnt2 <= 1 and k != after_ae_layers_key and k!= ae_datasets_key) else v
            if exp_config[k] == ['']:
                exp_config[k] = []
    # The loss function name 'kullback_leibler_divergence' mixes up with the key/value separation char '_', so handle
    # it here separately
    if cnt1 > 1:
        if changed:
            exp_config['LF'] = 'mean_squared_error'
        else:
            exp_config['LF'] = 'kullback_leibler_divergence'

    global version
    if version is not None:
        exp_config['V'] = version
    
    if ae_datasets_key not in exp_config:
        exp_config[ae_datasets_key] = exp_configs[ae_datasets_key][2]
    if after_ae_layers_key not in exp_config:
        exp_config[after_ae_layers_key] = exp_configs[after_ae_layers_key][2]
    if after_ae_act_key not in exp_config:
        exp_config[after_ae_act_key] = exp_configs[after_ae_act_key][2]
    if use_ae_key not in exp_config:
        exp_config[use_ae_key] = exp_configs[use_ae_key][2]
    if pca_dim_key not in exp_config:
        exp_config[pca_dim_key] = exp_configs[pca_dim_key][2]
    if max_norm_key not in exp_config:
        exp_config[max_norm_key] = exp_configs[max_norm_key][2]
    if dropout_pct_key not in exp_config:
        exp_config[dropout_pct_key] = exp_configs[dropout_pct_key][2]
    if input_dropout_pct_key not in exp_config:
        exp_config[input_dropout_pct_key] = exp_configs[input_dropout_pct_key][2]
    if class_weight_key not in exp_config:
        exp_config[class_weight_key] = exp_configs[class_weight_key][2]
    if nmf_dim_key not in exp_config:
        exp_config[nmf_dim_key] = exp_configs[nmf_dim_key][2]


# Add figure texts to plots that describe configs of the experiment that produced the plot
def add_figtexts_and_save(fig, name, desc, x_off=0.02, y_off=0.56, step=0.04, config=None):
    if add_fig_desc:
        pylab.figtext(x_off, y_off, desc)

        y_off -= step
        pylab.figtext(x_off, y_off,  get_config_desc(dataset_key) + ' --- ' + get_config_desc(norm_sample_key)  + ' --- ' + get_config_desc(no_random_key))

        y_off -= step
        pylab.figtext(x_off, y_off, get_config_desc(norm_input_key) + ' --- ' + get_config_desc(shuffle_labels_key))
        
        y_off -= step
        pylab.figtext(x_off, y_off, get_config_desc(layers_key) + ' --- ' + get_config_desc(enc_dim_key) + ' --- ' + get_config_desc(use_ae_key))

        y_off -= step
        pylab.figtext(x_off, y_off, get_config_desc(enc_act_key) + ' --- ' + get_config_desc(code_act_key))

        y_off -= step
        pylab.figtext(x_off, y_off, get_config_desc(dec_act_key) + ' --- ' + get_config_desc(out_act_key)
                      + ' --- ' + get_config_desc(use_kfold_key) + ' --- ' + get_config_desc(kfold_key))

        y_off -= step
        pylab.figtext(x_off, y_off, get_config_desc(loss_func_key) + ' --- ' + get_config_desc(batch_size_key) + ' --- ' + get_config_desc(shuffle_abunds_key))

        y_off -= step
        pylab.figtext(x_off, y_off, get_config_desc(auto_epochs_key) + ' --- ' + get_config_desc(super_epochs_key))

        y_off -= step
        pylab.figtext(x_off, y_off, get_config_desc(batch_norm_key) + " --- " + get_config_desc(early_stop_key) + " --- "
                      + get_config_desc(patience_key))

        y_off -= step
        pylab.figtext(x_off, y_off, 'Number of training samples: {}'.format(x_train.shape[0]) +
                      ' --- Number of test samples: {}'.format(x_test.shape[0]) + " --- " + get_config_desc(act_reg_key))

        y_off -= step
        pylab.figtext(x_off, y_off,  get_config_desc(dropout_pct_key) + " --- " + get_config_desc(backend_key)
                      + ' --- ' + get_config_desc(version_key) + ' --- ' + get_config_desc(num_iters_key) + ' --- ' + get_config_desc(iter_key))


    filename = graph_dir + '/' + name + (name_file_from_config(exp_config) if config is None else config) + '.png'
    pylab.savefig(filename , bbox_inches='tight')
    pylab.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= "Program to run linear machine learning models on kmer datasets")
    parser.add_argument('-featimps', type = int, default = -1, help = "Number of feature importances")
    parser.add_argument('-saveweightsauto', type = bool, default = False, help = "Whether to save model weights from the autoencoder")
    parser.add_argument('-saveweightssuper', type = bool, default = False, help = "Whether to save model weights from supervised learning")
    parser.add_argument('-autoweightsfile', type = str, default = '', help = "File to load autoencoder weights from")
    parser.add_argument('-superweightsfile', type = str, default = '', help = "File to load supervised weights from")
    parser.add_argument('-logiterfeats', type = bool, default = False, help = "Whether to log feature importances for all iterations")
    parser.add_argument('-version', type = str, default = None, help = "Version of the model being run")
    
    arg_vals = parser.parse_args()
    global num_feature_imps, log_iter_feats, version, save_weights_auto, save_weights_super, auto_weights_file, super_weights_file
    num_feature_imps = arg_vals.featimps
    save_weights_auto = arg_vals.saveweightsauto
    save_weights_super = arg_vals.saveweightssuper
    auto_weights_file = arg_vals.autoweightsfile
    super_weights_file = arg_vals.superweightsfile
    log_iter_feats = arg_vals.logiterfeats
    version = arg_vals.version
    if log_iter_feats:
        num_feature_imps = -1
    
    # the experiment mode can be one of SUPER_MODELS, AUTO_MODELS, SEARCH_SUPER_MODELS, SEARCH_AUTO_MODELS, OTHER
    # - see below
    exp_mode = "SUPER_MODELS"

    if exp_mode == "SUPER_MODELS":
        plot_ae_fold = False
        plot_ae_overall = False

        plot_fold = False
        plot_iter = False
        

        # Overall plotting - aggregate results across both folds and iteration
        plot_overall = True

        '''
        dataset_configs['LeChatelier'] = [
            #'LS:32896-2_EA:linear_CA:linear_DA:NA_OA:NA_ED:2_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:lechat_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_',
            'LS:32896-2_EA:linear_CA:linear_DA:NA_OA:NA_ED:2_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:lechat_AU:0_NR:0_SL:1_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_',
        ]
        '''

        #'''
        dataset_configs['KarlssonNoAdap'] = [
            #'LS:82-2_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:2_PC:82_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:noadap_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_',
            'LS:80-2_EA:linear_CA:linear_DA:NA_OA:NA_ED:2_PC:80_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:noadap_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_',
            #'LS:80-2_EA:linear_CA:linear_DA:NA_OA:NA_ED:2_PC:80_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:noadap_AU:0_NR:0_SL:1_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_',
        ]
        #'''

        '''
        dataset_configs['KarlssonNoAdap'] = [
            'LS:50-32_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:32_PC:50_AEP:50_SEP:200_BS:32_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:18_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_AD:_AL:_AA:None_NM:0_CW:0',
            ]
        '''

        '''
        dataset_configs['SingleDiseaseMetaHIT'] = [
            'LS:8192-8_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:8_PC:0_AEP:1_SEP:400_BS:16_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:18_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:_AL:_AA:None_NM:0_CW:0'
            ]
            
        '''


        '''
        dataset_configs['SingleDiseaseQin'] = [
            'DS:SingleDiseaseQin_LS:8192-3_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:4_PC:0_AEP:1_SEP:400_BS:16_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:18s_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:_AL:_AA:None_NM:0_CW:0'
            ]
        '''

        '''
        dataset_configs['SingleDiseaseQin'] = [
            'DS:SingleDiseaseQin_LS:8192-3_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:4_PC:0_AEP:1_SEP:400_BS:16_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:18_AU:0_NR:0_SL:1_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:_AL:_AA:None_NM:0_CW:0'
            ]
        '''

        '''
        dataset_configs['SingleDiseaseLiverCirrhosis'] = [
            #"LS:32896-4-32_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:32_PC:0_AEP:0_SEP:400_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:lifinal_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            #"LS:32896-4-4_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:4_PC:0_AEP:0_SEP:400_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:lifinal_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            #"LS:32896-6-2_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:2_PC:0_AEP:0_SEP:400_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:lifinal_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            "LS:32896-8-4_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:4_PC:0_AEP:0_SEP:400_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:lifinal_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            "LS:32896-4_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:4_PC:0_AEP:0_SEP:400_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:zell_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            "LS:32896-8_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:8_PC:0_AEP:0_SEP:400_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:zell_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            "LS:32896-6-8_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:8_PC:0_AEP:0_SEP:400_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:lifinal_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            "LS:32896-6-4_EA:linear_CA:sigmoid_DA:NA_OA:NA_ED:4_PC:0_AEP:0_SEP:400_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:lifinal_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            "LS:32896-4-64_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:64_PC:0_AEP:0_SEP:400_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:lifinal_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
        ]
        '''

        '''
        dataset_configs['SingleDiseaseLiverCirrhosis'] = [
            'DS:SingleDiseaseLiverCirrhosis_LS:32896-8-2_EA:sigmoid_CA:softmax_DA:NA_OA:NA_ED:2_PC:0_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:18_AU:0_NR:0_SL:1_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:None_NM:0_CW:0'
            ]
        '''

        '''
        dataset_configs['ZellerReduced'] = [
            "LS:32896-2_EA:linear_CA:linear_DA:NA_OA:NA_ED:2_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:zefinal_AU:0_NR:0_SL:1_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            "LS:32896-2_EA:linear_CA:linear_DA:NA_OA:NA_ED:2_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:zefinal_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            "LS:32896-4_EA:linear_CA:linear_DA:NA_OA:NA_ED:4_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:zefinal_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            "LS:32896-4_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:4_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:zefinal_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            "LS:32896-2_EA:tanh_CA:tanh_DA:NA_OA:NA_ED:2_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:zefinal_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            "LS:32896-16_EA:tanh_CA:tanh_DA:NA_OA:NA_ED:16_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:zefinal_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
        ]
        '''

        '''
        #dataset_configs['ZellerReduced'] = [
            #"LS:32896-2_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:2_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:zefinal_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            #"LS:32896-32_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:32_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:zefinal_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            #"LS:32896-8_EA:tanh_CA:tanh_DA:NA_OA:NA_ED:8_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:zefinal_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            #"LS:32896-2_EA:relu_CA:relu_DA:NA_OA:NA_ED:2_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:zefinal_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
        #]
        '''

        '''
        dataset_configs['ZellerReduced'] = [
            #"LS:32896-32_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:32_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:zefinal_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:4_AD:_AL:_AA:linear_NM:0_CW:0_",
            #"LS:32896-32_EA:linear_CA:linear_DA:NA_OA:NA_ED:32_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:zefinal_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:4_AD:_AL:_AA:linear_NM:0_CW:0_",

            #"LS:32896-32_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:32_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0.25_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:zefinal_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            "LS:32896-8_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:8_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0.35_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:zefinal_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:4_AD:_AL:_AA:linear_NM:0_CW:0_",
        ]
        '''

        '''
        dataset_configs['ZellerReduced'] = [
            'DS:ZellerReduced_LS:512-32_EA:softmax_CA:softmax_DA:NA_OA:NA_ED:32_PC:0_AEP:50_SEP:400_BS:16_LF:kullback_leibler_divergence_BN:0_DP:0.75_IDP:0.75_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:18_AU:0_NR:0_SL:1_SA:0_UK:10_ITS:20_KS:5_MN:0_AD:_AL:_AA:None_NM:0_CW:0'
        ]
        '''


        '''
        dataset_configs['SingleDiseaseFeng'] = [
            'DS:SingleDiseaseFeng_LS:40-64_EA:relu_CA:relu_DA:NA_OA:NA_ED:64_PC:40_AEP:50_SEP:200_BS:32_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:18_AU:0_NR:0_SL:1_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:None_NM:0_CW:0'
        ]
        '''

        '''
        dataset_configs['SingleDiseaseRA'] = [
            'DS:SingleDiseaseRA_LS:512-2-2_EA:sigmoid_CA:softmax_DA:NA_OA:NA_ED:2_PC:0_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:16_AU:0_NR:0_SL:1_SA:0_UK:10_ITS:20_KS:5_MN:0_AD:_AL:_AA:None_NM:0_CW:0'
        ]
        '''
        
        '''
        dataset_configs['SingleDiseaseRA'] = [
            #"LS:32896-2_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:2_PC:0_AEP:0_SEP:400_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:rasearch_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            # "LS:32896-8_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:8_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:rasearch_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            # "LS:32896-8_EA:tanh_CA:tanh_DA:NA_OA:NA_ED:8_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:rasearch_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            # "LS:32896-64_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:64_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:rasearch_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            # "LS:32896-2_EA:tanh_CA:tanh_DA:NA_OA:NA_ED:2_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:rasearch_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            # "LS:32896-4_EA:linear_CA:linear_DA:NA_OA:NA_ED:4_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:rasearch_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            # "LS:32896-4_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:4_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:rasearch_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            #"LS:32896-2_EA:linear_CA:linear_DA:NA_OA:NA_ED:2_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:rafinal_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            #"LS:32896-2_EA:linear_CA:linear_DA:NA_OA:NA_ED:2_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:rafinal_AU:0_NR:0_SL:1_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",

            #"LS:32896-2_EA:linear_CA:linear_DA:NA_OA:NA_ED:2_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:18_AU:0_NR:0_SL:1_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",

            # "LS:32896-32_EA:tanh_CA:tanh_DA:NA_OA:NA_ED:32_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:rasearch_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            #"LS:32896-1_EA:linear_CA:linear_DA:NA_OA:NA_ED:1_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:rasearch_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            #"LS:32896-3_EA:linear_CA:linear_DA:NA_OA:NA_ED:3_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:rasearch_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
        ]
        '''

        '''
        dataset_configs['RANoAdap'] = [
            "LS:32896-2_EA:linear_CA:linear_DA:NA_OA:NA_ED:2_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:rafinal_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            #"LS:32896-2_EA:linear_CA:linear_DA:NA_OA:NA_ED:2_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:rafinal_AU:0_NR:0_SL:1_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",

        ]
        '''
        
        '''
        dataset_configs['KarlssonReduced'] = [
            'DS:KarlssonReduced_LS:50-32_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:32_PC:50_AEP:50_SEP:200_BS:32_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:memtest_AU:0_NR:0_SL:1_SA:0_UK:10_ITS:20_KS:10_MN:0_AD:_AL:_AA:None_NM:0_CW:0'
            ]
        '''

        
        

        '''
        dataset_configs['SingleDiseaseRA'] = [
            "LS:2080-64_EA:sigmoid_CA:sigmoid_DA:linear_OA:linear_ED:64_PC:0_AEP:50_SEP:400_BS:32_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:16_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_MN:0_AD:_",
            ]
        '''

        '''
        dataset_configs['SingleDiseaseRA'] = [

            "LS:2080-32_EA:tanh_CA:tanh_DA:relu_OA:relu_ED:32_PC:0_AEP:50_SEP:400_BS:8_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:16_AU:1_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_MN:0_AD:R-F-L-Z-M_",
            "LS:2080-50_EA:relu_CA:relu_DA:softmax_OA:softmax_ED:50_PC:0_AEP:50_SEP:200_BS:16_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:16_AU:1_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_MN:0_AD:R-F-L-Z-M_",
            "LS:2080-64_EA:sigmoid_CA:sigmoid_DA:linear_OA:linear_ED:64_PC:0_AEP:50_SEP:400_BS:32_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:16_AU:1_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_MN:0_AD:R-F-L-Z-M_",
            "LS:2080-64_EA:linear_CA:linear_DA:softmax_OA:softmax_ED:64_PC:0_AEP:50_SEP:400_BS:16_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:16_AU:1_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_MN:0_AD:R-F-L-Z-M_",
            "LS:8192-32_EA:sigmoid_CA:sigmoid_DA:relu_OA:relu_ED:32_PC:0_AEP:50_SEP:200_BS:16_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:16_AU:1_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:R-F-L-Z-M_",
            "LS:8192-32_EA:tanh_CA:tanh_DA:tanh_OA:tanh_ED:32_PC:0_AEP:50_SEP:400_BS:32_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:16_AU:1_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:R-F-L-Z-M_",
            "LS:8192-32_EA:linear_CA:linear_DA:relu_OA:relu_ED:32_PC:0_AEP:50_SEP:400_BS:16_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:16_AU:1_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:R-F-L-Z-M_",
            "LS:8192-64_EA:sigmoid_CA:sigmoid_DA:relu_OA:relu_ED:64_PC:0_AEP:50_SEP:400_BS:32_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:16_AU:1_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:R-F-L-Z-M_",
            "LS:8192-64_EA:linear_CA:linear_DA:linear_OA:linear_ED:64_PC:0_AEP:50_SEP:200_BS:32_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:16_AU:1_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:R-F-L-Z-M_",
            "LS:8192-4096-64_EA:linear_CA:sigmoid_DA:linear_OA:softmax_ED:64_PC:0_AEP:50_SEP:400_BS:8_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:16_AU:1_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:R-F-L-Z-M_",

            ]
        '''

        '''
        dataset_configs['SingleDiseaseQin'] = [
            "LS:8192-2-4_EA:sigmoid_CA:sigmoid_DA:softmax_OA:sigmoid_ED:4_AEP:1_SEP:400_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:18_AE:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7"

            ]
        '''

        '''

        dataset_configs['SingleDiseaseFeng'] = [
            "LS:40-64_EA:relu_CA:relu_DA:NA_OA:NA_ED:64_PC:40_AEP:50_SEP:200_BS:32_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:18_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_",
            ]
        '''

        '''
        dataset_configs['SingleDiseaseLiverCirrhosis'] = [
            "LS:32896-8-2_EA:sigmoid_CA:softmax_DA:NA_OA:NA_ED:2_PC:0_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:18_AE:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_",
            ]
        '''

        '''
        dataset_configs['ZellerReduced'] = [
            "LS:512-32_EA:softmax_CA:softmax_DA:sigmoid_OA:sigmoid_ED:32_PC:0_AEP:50_SEP:400_BS:16_LF:kullback_leibler_divergence_BN:0_DP:0.75_IDP:0.75_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:18_AE:1_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:5_MN:0_",
            ]
        '''

        '''
        dataset_configs['SingleDiseaseRA'] = [
            "LS:512-2-2_EA:sigmoid_CA:softmax_DA:linear_OA:tanh_ED:2_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:18_AE:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:5_",
            ]
        '''

        '''

        dataset_configs['KarlssonReduced'] = [
            "LS:50-32_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:32_PC:50_AEP:50_SEP:200_BS:32_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:18_AE:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
            ]
        '''

        '''
        dataset_configs['SingleDiseaseMetaHIT'] = [
            "LS:8192-40_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:40_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:18_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:_AL:_AA:linear_NM:0_CW:0_",
            ]
        '''

        '''
        dataset_configs['SingleDiseaseQin'] = [
            "LS:8192-8_EA:sigmoid_CA:sigmoid_DA:tanh_OA:tanh_ED:8_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:18_AE:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
            ]
        '''
        

    
        

        '''
        dataset_configs['SingleDiseaseMetaHIT'] = [
                                                            "LS:8192-512_EA:relu_CA:relu_DA:NA_OA:NA_ED:512_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0.5_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:14_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:4_AD:_AL:_AA:linear_NM:0_CW:1.5_",
"LS:32896-1024_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:1024_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0.5_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:15_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:3_AD:_AL:_AA:linear_NM:0_CW:0_",
"LS:8192-1024_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:1024_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0.5_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:15_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:1_AD:_AL:_AA:linear_NM:0_CW:2_",
"LS:32896-512_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:512_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0.25_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:15_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:1_AD:_AL:_AA:linear_NM:0_CW:2.5_",
"LS:32896-512_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:512_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0.25_IDP:0.25_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:15_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:2_AD:_AL:_AA:linear_NM:0_CW:2.5_",
"LS:8192-512_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:512_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0.5_IDP:0.25_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:15_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:3_AD:_AL:_AA:linear_NM:0_CW:1.5_",
"LS:8192-512_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:512_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:15_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:2_AD:_AL:_AA:linear_NM:0_CW:3.5_",
"LS:8192-512_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:512_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0.25_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:15_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:1_AD:_AL:_AA:linear_NM:0_CW:1.5_",
"LS:32896-1024_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:1024_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0.25_IDP:0.25_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:15_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:2_AD:_AL:_AA:linear_NM:0_CW:0_",
"LS:32896-512_EA:tanh_CA:tanh_DA:NA_OA:NA_ED:512_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0.5_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:15_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:1_KS:8_MN:4_AD:_AL:_AA:linear_NM:0_CW:2_",
                                                            ]
        '''


        '''
        dataset_configs['SingleDiseaseLiverCirrhosis'] = ["LS:512-2_EA:sigmoid_CA:sigmoid_DA:linear_OA:linear_ED:2_PC:0_AEP:50_SEP:4_BS:8_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:1_NR:0_SL:0_SA:0_UK:3_ITS:1_MN:0_KS:5_"
                                                                       ]
        '''

        '''
        dataset_configs['SingleDiseaseMetaHIT'] = [
                                                    "LS:8192-40_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:40_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:12_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:_AL:_AA:linear_CW:1.1875",
                                                    "LS:8192-40_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:40_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:12_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:_AL:_AA:linear_CW:1.3125",
                                                    "LS:8192-40_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:40_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:12_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:_AL:_AA:linear_CW:1.4375",
                                                    ]

        '''
        
        '''
        dataset_configs['SingleDiseaseMetaHIT'] = [
                                                    "LS:8192-40_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:40_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:12_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:_AL:_AA:linear_CW:1.25",
                                                    "LS:8192-40_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:40_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:12_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:_AL:_AA:linear_CW:1.125",
                                                    "LS:8192-40_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:40_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:12_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:_AL:_AA:linear_CW:1.375",
                                                    ]

        '''
        
        '''
        dataset_configs['SingleDiseaseMetaHIT'] = [

                                                    "LS:8192-40_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:40_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:12_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:_AL:_AA:linear_CW:1.5",
                                                    "LS:8192-40_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:40_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:12_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:_AL:_AA:linear_CW:2",
                                                    "LS:8192-40_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:40_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:12_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:_AL:_AA:linear_CW:2.5",
                                                    "LS:8192-40_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:40_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:12_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:_AL:_AA:linear_CW:3",
                                                    "LS:8192-40_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:40_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:12_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:_AL:_AA:linear_CW:3.5",
                                                    "LS:8192-40_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:40_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:12_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:_AL:_AA:linear_CW:4",


                                                    ]
        '''

        '''
        dataset_configs['SingleDiseaseMetaHIT'] = [
                "LS:8192-512_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:512_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0.75_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:13_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:1_AD:_AL:_AA:linear_NM:0_",
                "LS:8192-128_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:128_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0.35_IDP:0.35_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:13_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:3_AD:_AL:_AA:linear_NM:0_",
                "LS:8192-128_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:128_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0.25_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:13_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:_AL:_AA:linear_NM:0_",
                "LS:8192-16_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:16_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0.25_IDP:0.75_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:13_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:2_AD:_AL:_AA:linear_NM:0_",
                "LS:8192-1024_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:1024_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0.35_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:13_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:1_AD:_AL:_AA:linear_NM:0_",
                "LS:8192-16_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:16_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:13_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:1_AD:_AL:_AA:linear_NM:0_",
                "LS:8192-1024_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:1024_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0.35_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:13_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:3_AD:_AL:_AA:linear_NM:0_",
                "LS:8192-32_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:32_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0.75_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:13_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:3_AD:_AL:_AA:linear_NM:0_",
                "LS:8192-16_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:16_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0.75_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:13_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:2_AD:_AL:_AA:linear_NM:0_",
                "LS:8192-16_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:16_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0.35_IDP:0.35_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:13_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:3_AD:_AL:_AA:linear_NM:0_",
                        ]
        '''

        '''
        dataset_configs['SingleDiseaseLiverCirrhosis'] = ["LS:32896-8-2_EA:sigmoid_CA:softmax_DA:linear_OA:linear_ED:2_PC:0_AEP:0_SEP:400_BS:8_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:2_ES:0_PA:2_NO:L1_BE:theano_V:6_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:_AL:_AA:None_"
                                                        ]
        '''

        '''
        dataset_configs['SingleDiseaseQin'] = ["LS:8192-2-4_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:4_PC:0_AEP:1_SEP:400_BS:16_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:2_ES:0_PA:2_NO:L1_BE:tensorflow_V:6_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:_AL:_AA:None_"]

        '''
        
        '''
        dataset_configs['SingleDiseaseMetaHIT'] = ["LS:8192-40_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:40_PC:0_AEP:0_SEP:200_BS:32_LF:kullback_leibler_divergence_BN:0_DP:0.5_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:12_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:_AL:_AA:linear_",
                                                    "LS:8192-80_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:80_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0.5_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:12_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:_AL:_AA:linear_",
                                                    "LS:8192-40_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:40_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:12_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:_AL:_AA:linear_",
                                                    "LS:8192-40_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:40_PC:0_AEP:0_SEP:200_BS:16_LF:kullback_leibler_divergence_BN:0_DP:0.5_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:12_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:_AL:_AA:linear_",
                                                    "LS:80-40_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:40_PC:80_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:12_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_AD:_AL:_AA:linear_",
                                                    "LS:2080-1040-20_EA:linear_CA:tanh_DA:NA_OA:NA_ED:20_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0.25_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:12_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_MN:0_AD:_AL:_AA:linear_",
                                                    "LS:8192-20_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:20_PC:0_AEP:0_SEP:200_BS:32_LF:kullback_leibler_divergence_BN:0_DP:0.5_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:12_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:_AL:_AA:linear_",
                                                    "LS:8192-20_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:20_PC:0_AEP:0_SEP:400_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0.5_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:12_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:_AL:_AA:linear_",
                                                    "LS:8192-20_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:20_PC:0_AEP:0_SEP:400_BS:16_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:12_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:_AL:_AA:linear_",
                                                    "LS:8192-80_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:80_PC:0_AEP:0_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:12_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:_AL:_AA:linear_",
                                                    ]
        '''

        '''
        dataset_configs['SingleDiseaseMetaHIT'] = ["LS:8192-8_EA:sigmoid_CA:sigmoid_DA:softmax_OA:softmax_ED:8_AEP:1_SEP:400_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:11_AE:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_"
                                                    ]
        
        '''

        '''
        dataset_configs['SingleDiseaseLiverCirrhosis'] = ["LS:32896-8-2_EA:sigmoid_CA:softmax_DA:NA_OA:NA_ED:2_PC:0_AEP:0_SEP:400_BS:8_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:0_ES:0_PA:2_NO:L1_BE:theano_V:6_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_MN:0_KS:8_"]
        '''
        
        '''
        dataset_configs['SingleDiseaseQin'] = [
                                                "LS:8192-2-4_EA:sigmoid_CA:sigmoid_DA:softmax_OA:sigmoid_ED:4_AEP:1_SEP:400_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:0_ES:0_PA:2_NO:L1_BE:tensorflow_V:6_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",]
        '''

        '''
        dataset_configs['SingleDiseaseMetaHIT'] = [
                                                    'LS:8192-8_EA:sigmoid_CA:sigmoid_DA:softmax_OA:softmax_ED:8_AEP:1_SEP:400_BS:16_LF:mean_squared_error_BN:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:10_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_DP:0_IDP:0',
                                                    ]
        '''
        '''
        dataset_configs['SingleDiseaseFeng'] = ['LS:80-32_EA:sigmoid_CA:sigmoid_DA:softmax_OA:softmax_ED:32_PC:0_AEP:2_SEP:2_BS:32_LF:kullback_leibler_divergence_BN:0_DP:0.5_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:7_AU:0_NR:0_SL:0_SA:0_UK:3_ITS:1_KS:5_MN:0_AD:_AL:_AA:linear_NM:80']
        '''

        '''
        dataset_configs['SingleDiseaseFeng'] = ['LS:32896-64_EA:sigmoid_CA:sigmoid_DA:softmax_OA:softmax_ED:64_PC:0_AEP:200_SEP:400_BS:32_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:6_AU:1_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:H-Q-L-Z-M-K_AL:40_AA:sigmoid_']
        '''
        
        '''
        dataset_configs['ZellerReduced'] = ['LS:512-32_EA:linear_CA:linear_DA:softmax_OA:softmax_ED:32_PC:0_AEP:200_SEP:400_BS:16_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:6_AU:1_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:5_MN:0_AD:H-Q-L-F-M-K_AL:_AA:linear_']
        '''
        '''
        dataset_configs['SingleDiseaseLiverCirrhosis'] = ["LS:32896-8-2_EA:sigmoid_CA:softmax_DA:NA_OA:NA_ED:2_PC:0_AEP:0_SEP:400_BS:8_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:6_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_MN:0_KS:8_"]
        '''
                                                          
        '''
        dataset_configs['SingleDiseaseLiverCirrhosis'] = ["LS:512-2_EA:sigmoid_CA:sigmoid_DA:linear_OA:linear_ED:2_PC:0_AEP:1_SEP:1_BS:8_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:1_NR:0_SL:0_SA:0_UK:3_ITS:1_MN:0_KS:5_"
                                                                       ]
        '''
        
        '''
        dataset_configs['SingleDiseaseLiverCirrhosis'] = ["LS:32896-8_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:8_PC:0_AEP:0_SEP:400_BS:8_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_MN:0_KS:8_"
                                                                ]
        '''
        
        '''
        dataset_configs['SingleDiseaseLiverCirrhosis'] = ["LS:8192-2-2_EA:tanh_CA:tanh_DA:sigmoid_OA:sigmoid_ED:2_PC:0_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0",
"LS:8192-2_EA:sigmoid_CA:sigmoid_DA:linear_OA:linear_ED:2_PC:0_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_MN:0_KS:7_",
"LS:32896-8-2_EA:sigmoid_CA:softmax_DA:linear_OA:linear_ED:2_PC:0_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_MN:0_KS:8_",
"LS:8192-2_EA:sigmoid_CA:sigmoid_DA:sigmoid_OA:sigmoid_ED:2_PC:0_AEP:1_SEP:200_BS:16_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_MN:0_KS:7_",
"LS:8192-2-8_EA:softmax_CA:sigmoid_DA:sigmoid_OA:softmax_ED:8_PC:0_AEP:1_SEP:400_BS:32_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_MN:0_KS:7_",
"LS:2080-1040-520-260-130-61_EA:linear_CA:sigmoid_DA:NA_OA:NA_ED:61_PC:0_AEP:1_SEP:200_BS:16_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_MN:0_",
"LS:8192-2-4_EA:tanh_CA:tanh_DA:linear_OA:linear_ED:4_PC:0_AEP:1_SEP:200_BS:8_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_MN:0_KS:7_",
"LS:8192-2-16_EA:tanh_CA:relu_DA:tanh_OA:sigmoid_ED:16_PC:0_AEP:1_SEP:400_BS:32_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_MN:0_KS:7_",
"LS:32896-8-4_EA:linear_CA:sigmoid_DA:NA_OA:NA_ED:4_PC:0_AEP:1_SEP:200_BS:32_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_",
"LS:8192-2_EA:softmax_CA:softmax_DA:tanh_OA:tanh_ED:2_PC:0_AEP:1_SEP:200_BS:8_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_MN:0_KS:7_"
]
        
        '''
        '''
        dataset_configs['KarlssonReduced'] = ["LS:50-2_EA:linear_CA:linear_DA:NA_OA:NA_ED:2_PC:50_AEP:50_SEP:400_BS:32_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
"LS:50-32_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:32_PC:50_AEP:50_SEP:200_BS:32_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
"LS:40-16_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:16_PC:40_AEP:50_SEP:200_BS:32_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
"LS:50-64_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:64_PC:50_AEP:50_SEP:200_BS:32_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
"LS:50-16_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:16_PC:50_AEP:50_SEP:200_BS:32_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
"LS:50-64_EA:relu_CA:relu_DA:NA_OA:NA_ED:64_PC:50_AEP:50_SEP:200_BS:16_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
"LS:50-8_EA:softmax_CA:softmax_DA:NA_OA:NA_ED:8_PC:50_AEP:50_SEP:400_BS:32_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
"LS:50-8_EA:linear_CA:linear_DA:NA_OA:NA_ED:8_PC:50_AEP:50_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
"LS:50-4_EA:tanh_CA:tanh_DA:NA_OA:NA_ED:4_PC:50_AEP:50_SEP:400_BS:32_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
"LS:40-16_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:16_PC:40_AEP:50_SEP:400_BS:32_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_"
                                             ]
        '''
        '''
        dataset_configs['SingleDiseaseMetaHIT'] = ["LS:524800-100_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:100_PC:0_AEP:50_SEP:200_BS:16_LF:mean_squared_error_BN:0_DP:0.75_IDP:0.75_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
"LS:70-4_EA:linear_CA:linear_DA:NA_OA:NA_ED:4_PC:70_AEP:50_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
"LS:524800-100_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:100_PC:0_AEP:50_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0.35_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
"LS:8192-4_EA:tanh_CA:tanh_DA:relu_OA:relu_ED:4_PC:0_AEP:1_SEP:200_BS:8_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_MN:0_KS:7_",
"LS:70-32_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:32_PC:70_AEP:50_SEP:400_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
"LS:70-4_EA:linear_CA:linear_DA:NA_OA:NA_ED:4_PC:70_AEP:50_SEP:400_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
"LS:8192-8_EA:sigmoid_CA:sigmoid_DA:sigmoid_OA:sigmoid_ED:8_PC:0_AEP:1_SEP:200_BS:16_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_MN:0_KS:7_",
"LS:70-16_EA:relu_CA:relu_DA:NA_OA:NA_ED:16_PC:70_AEP:50_SEP:200_BS:16_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
"LS:8192-2-16_EA:linear_CA:linear_DA:linear_OA:softmax_ED:16_PC:0_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_MN:0_KS:7_",
"LS:70-4_EA:tanh_CA:tanh_DA:NA_OA:NA_ED:4_PC:70_AEP:50_SEP:400_BS:32_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_"
                                                   ]
        
        '''
        '''
        dataset_configs['SingleDiseaseQin'] = ["LS:8192-16_EA:sigmoid_CA:sigmoid_DA:tanh_OA:tanh_ED:16_PC:0_AEP:1_SEP:400_BS:32_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_MN:0_KS:7_",
"LS:2080-1040-996_EA:relu_CA:linear_DA:NA_OA:NA_ED:996_PC:0_AEP:1_SEP:400_BS:16_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_MN:0_",
"LS:8192-2-2_EA:softmax_CA:sigmoid_DA:linear_OA:relu_ED:2_PC:0_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_MN:0_KS:7_",
"LS:8192-2-2_EA:softmax_CA:tanh_DA:softmax_OA:softmax_ED:2_PC:0_AEP:1_SEP:200_BS:16_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_MN:0_KS:7_",
"LS:2080-2-4_EA:linear_CA:sigmoid_DA:relu_OA:tanh_ED:4_PC:0_AEP:1_SEP:400_BS:32_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_MN:0_KS:6_"
                                               ]
        '''
        '''
        dataset_configs['ZellerReduced'] = [
            "LS:8192-64_EA:tanh_CA:tanh_DA:linear_OA:linear_ED:64_PC:0_AEP:50_SEP:200_BS:16_LF:mean_squared_error_BN:0_DP:0_IDP:0.75_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:1_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0",
            "LS:40-50_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:50_PC:40_AEP:50_SEP:400_BS:8_LF:mean_squared_error_BN:0_DP:0.25_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_MN:0_",
            "LS:50-8_EA:softmax_CA:softmax_DA:NA_OA:NA_ED:8_PC:50_AEP:50_SEP:400_BS:16_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_",
            "LS:512-32_EA:softmax_CA:softmax_DA:sigmoid_OA:sigmoid_ED:32_PC:0_AEP:50_SEP:400_BS:16_LF:kullback_leibler_divergence_BN:0_DP:0.75_IDP:0.75_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:1_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:5_MN:0",
        ]
        '''

        '''
        dataset_configs['SingleDiseaseFeng'] = [
            "LS:40-64_EA:relu_CA:relu_DA:NA_OA:NA_ED:64_PC:40_AEP:50_SEP:200_BS:32_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_",
            ]
        '''

        '''
        dataset_configs['ZellerReduced'] = ["LS:40-50_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:50_PC:40_AEP:50_SEP:200_BS:8_LF:mean_squared_error_BN:0_DP:0.25_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_MN:0_",
"LS:40-50_EA:relu_CA:relu_DA:NA_OA:NA_ED:50_PC:40_AEP:50_SEP:400_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0.5_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_MN:0_",
"LS:512-32_EA:softmax_CA:softmax_DA:sigmoid_OA:sigmoid_ED:32_PC:0_AEP:50_SEP:400_BS:16_LF:kullback_leibler_divergence_BN:0_DP:0.75_IDP:0.75_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:1_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:5_MN:0_",
"LS:40-64_EA:relu_CA:relu_DA:NA_OA:NA_ED:64_PC:40_AEP:50_SEP:200_BS:16_LF:mean_squared_error_BN:0_DP:0.25_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:5_MN:0_",
"LS:50-32_EA:linear_CA:linear_DA:NA_OA:NA_ED:32_PC:50_AEP:50_SEP:400_BS:32_LF:kullback_leibler_divergence_BN:0_DP:0.35_IDP:0.25_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_MN:0_",
"LS:512-32_EA:softmax_CA:softmax_DA:linear_OA:linear_ED:32_PC:0_AEP:50_SEP:200_BS:32_LF:mean_squared_error_BN:0_DP:0.35_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:1_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:5_MN:0_",
"LS:8192-32_EA:relu_CA:relu_DA:relu_OA:relu_ED:32_PC:0_AEP:50_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0.35_IDP:0.25_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:1_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_",
"LS:50-64_EA:linear_CA:linear_DA:NA_OA:NA_ED:64_PC:50_AEP:50_SEP:400_BS:16_LF:mean_squared_error_BN:0_DP:0.5_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_MN:0_",
"LS:512-32_EA:sigmoid_CA:sigmoid_DA:softmax_OA:softmax_ED:32_PC:0_AEP:50_SEP:400_BS:32_LF:kullback_leibler_divergence_BN:0_DP:0.5_IDP:0.75_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:1_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:5_MN:0_",
"LS:50-64_EA:linear_CA:linear_DA:NA_OA:NA_ED:64_PC:50_AEP:50_SEP:200_BS:16_LF:mean_squared_error_BN:0_DP:0.35_IDP:0.5_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_MN:0_",]
        '''
        '''
        dataset_configs['ZellerReduced'] = ["LS:70-64_EA:relu_CA:relu_DA:NA_OA:NA_ED:64_PC:70_AEP:50_SEP:200_BS:16_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_",
"LS:40-64_EA:tanh_CA:tanh_DA:NA_OA:NA_ED:64_PC:40_AEP:50_SEP:200_BS:8_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_",
"LS:40-4_EA:tanh_CA:tanh_DA:NA_OA:NA_ED:4_PC:40_AEP:50_SEP:400_BS:16_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_",
"LS:40-64_EA:relu_CA:relu_DA:NA_OA:NA_ED:64_PC:40_AEP:50_SEP:200_BS:32_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_",
"LS:40-16_EA:softmax_CA:softmax_DA:NA_OA:NA_ED:16_PC:40_AEP:50_SEP:400_BS:16_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_",
"LS:50-8_EA:softmax_CA:softmax_DA:NA_OA:NA_ED:8_PC:50_AEP:50_SEP:200_BS:16_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_",
"LS:40-64_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:64_PC:40_AEP:50_SEP:400_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_",
"LS:40-64_EA:linear_CA:linear_DA:NA_OA:NA_ED:64_PC:40_AEP:50_SEP:200_BS:16_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_",
"LS:40-16_EA:tanh_CA:tanh_DA:NA_OA:NA_ED:16_PC:40_AEP:50_SEP:200_BS:32_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_",
"LS:40-32_EA:tanh_CA:tanh_DA:NA_OA:NA_ED:32_PC:40_AEP:50_SEP:400_BS:32_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_"]
        '''
        '''
        dataset_configs['SingleDiseaseFeng'] = ["LS:40-4_EA:linear_CA:linear_DA:NA_OA:NA_ED:4_PC:40_AEP:50_SEP:400_BS:32_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_",
"LS:70-2_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:2_PC:70_AEP:50_SEP:400_BS:8_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
"LS:70-8_EA:tanh_CA:tanh_DA:NA_OA:NA_ED:8_PC:70_AEP:50_SEP:200_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_",
"LS:70-4_EA:softmax_CA:softmax_DA:NA_OA:NA_ED:4_PC:70_AEP:50_SEP:400_BS:32_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
"LS:70-8_EA:relu_CA:relu_DA:NA_OA:NA_ED:8_PC:70_AEP:50_SEP:400_BS:16_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
"LS:70-4_EA:relu_CA:relu_DA:NA_OA:NA_ED:4_PC:70_AEP:50_SEP:400_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
"LS:70-16_EA:linear_CA:linear_DA:NA_OA:NA_ED:16_PC:70_AEP:50_SEP:200_BS:32_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
"LS:70-8_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:8_PC:70_AEP:50_SEP:400_BS:8_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
"LS:70-16_EA:linear_CA:linear_DA:NA_OA:NA_ED:16_PC:70_AEP:50_SEP:400_BS:16_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
"LS:50-16_EA:relu_CA:relu_DA:NA_OA:NA_ED:16_PC:50_AEP:50_SEP:200_BS:32_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_"
                                                ]
        '''
        '''
        dataset_configs['SingleDiseaseRA'] = [
"LS:70-16_EA:relu_CA:relu_DA:NA_OA:NA_ED:16_PC:70_AEP:50_SEP:400_BS:8_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
"LS:70-16_EA:relu_CA:relu_DA:NA_OA:NA_ED:16_PC:70_AEP:50_SEP:400_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
"LS:512-2-4_EA:relu_CA:sigmoid_DA:tanh_OA:linear_ED:4_PC:0_AEP:1_SEP:200_BS:16_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_MN:0_KS:5_",
"LS:524800-4_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:4_PC:0_AEP:50_SEP:200_BS:32_LF:mean_squared_error_BN:0_DP:0.5_IDP:0.75_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
"LS:2080-4_EA:sigmoid_CA:sigmoid_DA:relu_OA:relu_ED:4_PC:0_AEP:1_SEP:200_BS:16_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_MN:0_KS:6_",
"LS:50-16_EA:relu_CA:relu_DA:NA_OA:NA_ED:16_PC:50_AEP:50_SEP:400_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
"LS:70-16_EA:relu_CA:relu_DA:NA_OA:NA_ED:16_PC:70_AEP:50_SEP:200_BS:32_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:10_MN:0_",
"LS:2080-2-4_EA:sigmoid_CA:sigmoid_DA:relu_OA:tanh_ED:4_PC:0_AEP:1_SEP:200_BS:16_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_MN:0_KS:6_",
"LS:512-2-2_EA:sigmoid_CA:softmax_DA:linear_OA:tanh_ED:2_PC:0_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_MN:0_KS:5_"
                                              ]
        '''

        
        '''
        dataset_configs['SingleDiseaseLiverCirrhosis'] = ['LS:8192-50_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:50_AEP:1_SEP:200_BS:8_LF:mean_squared_error_BN:0_DP:0.5_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0',
                                                          'LS:8192-50_EA:sigmoid_CA:sigmoid_DA:NA_OA:NA_ED:50_AEP:1_SEP:200_BS:16_LF:mean_squared_error_BN:0_DP:0_IDP:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0',
                                                          ]
        '''
        '''
        dataset_configs['SingleDiseaseKarlsson'] = [
            'LS:8192-4_EA:tanh_CA:tanh_DA:linear_OA:linear_ED:4_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7',
            ]
        '''
        
        '''
        dataset_configs['SingleDiseaseQin'] = [
                                                "LS:32896-2-4_EA:sigmoid_CA:sigmoid_DA:softmax_OA:sigmoid_ED:4_AEP:1_SEP:400_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8__",
                                                "LS:32896-8_EA:sigmoid_CA:sigmoid_DA:tanh_OA:tanh_ED:8_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8__",
                                                "LS:32896-4_EA:tanh_CA:tanh_DA:linear_OA:linear_ED:4_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8__",
                                                "LS:32896-4_EA:tanh_CA:tanh_DA:sigmoid_OA:sigmoid_ED:4_AEP:1_SEP:200_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8__",
                                                "LS:32896-2-4_EA:linear_CA:sigmoid_DA:relu_OA:sigmoid_ED:4_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_",

                                                ]
        '''                                                       
        '''
        dataset_configs['SingleDiseaseRA'] = [
                                                "LS:32896-2-16_EA:softmax_CA:softmax_DA:sigmoid_OA:sigmoid_ED:16_AEP:1_SEP:200_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_",
                                                "LS:32896-2_EA:tanh_CA:tanh_DA:linear_OA:linear_ED:2_AEP:1_SEP:400_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_",
                                                "LS:32896-16448-4_EA:linear_CA:sigmoid_DA:relu_OA:linear_ED:4_AEP:1_SEP:200_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_",
                                                "LS:32896-2-2_EA:sigmoid_CA:softmax_DA:linear_OA:tanh_ED:2_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_",
                                                "LS:32896-2-2_EA:tanh_CA:tanh_DA:softmax_OA:sigmoid_ED:2_AEP:1_SEP:200_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_"
                                                ]
        '''
        '''
        dataset_configs['SingleDiseaseFeng'] = [
                                                      "LS:32896-16_EA:sigmoid_CA:sigmoid_DA:linear_OA:linear_ED:16_AEP:1_SEP:200_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8__",
"LS:32896-16_EA:tanh_CA:tanh_DA:tanh_OA:tanh_ED:16_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8__",
"LS:32896-2-4_EA:linear_CA:sigmoid_DA:relu_OA:tanh_ED:4_AEP:1_SEP:200_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8__",
"LS:32896-2-4_EA:linear_CA:sigmoid_DA:relu_OA:tanh_ED:4_AEP:1_SEP:200_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8__",
                                                 "LS:32896-16_EA:sigmoid_CA:sigmoid_DA:sigmoid_OA:sigmoid_ED:16_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8__"
                                               
]
        '''
        '''
        dataset_configs['SingleDiseaseZeller'] = [
                                                "LS:32896-16_EA:tanh_CA:tanh_DA:sigmoid_OA:sigmoid_ED:16_AEP:1_SEP:200_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_",
"LS:32896-2_EA:sigmoid_CA:sigmoid_DA:linear_OA:linear_ED:2_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_",
"LS:32896-8_EA:tanh_CA:tanh_DA:linear_OA:linear_ED:8_AEP:1_SEP:200_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_",
"LS:32896-16448-16_EA:sigmoid_CA:sigmoid_DA:sigmoid_OA:relu_ED:16_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_",
"LS:32896-2_EA:sigmoid_CA:sigmoid_DA:linear_OA:linear_ED:2_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_"
                                                ]
        '''
        
        '''
        dataset_configs['SingleDiseaseLiverCirrhosis'] = [
                                                        "LS:32896-2-8_EA:softmax_CA:sigmoid_DA:sigmoid_OA:softmax_ED:8_AEP:1_SEP:400_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_"
                                                        "LS:32896-2-4_EA:sigmoid_CA:tanh_DA:softmax_OA:softmax_ED:4_AEP:1_SEP:200_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_",
                                                        "LS:32896-2-2_EA:softmax_CA:sigmoid_DA:sigmoid_OA:softmax_ED:2_AEP:1_SEP:400_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_",
                                                        "LS:32896-2_EA:tanh_CA:tanh_DA:tanh_OA:tanh_ED:2_AEP:1_SEP:200_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_",
                                                        "LS:32896-2-2_EA:sigmoid_CA:sigmoid_DA:softmax_OA:tanh_ED:2_AEP:1_SEP:400_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_",
                                                        "LS:32896-2_EA:tanh_CA:tanh_DA:tanh_OA:tanh_ED:2_AEP:1_SEP:200_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_"
                                                         ]
        '''
        '''
        dataset_configs['SingleDiseaseKarlsson'] = [
                                                    "LS:32896-16_EA:relu_CA:relu_DA:tanh_OA:tanh_ED:16_AEP:1_SEP:400_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8__",
                                                    "LS:32896-2-8_EA:relu_CA:linear_DA:linear_OA:sigmoid_ED:8_AEP:1_SEP:400_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8__",
                                                    "LS:32896-16448-2_EA:sigmoid_CA:tanh_DA:tanh_OA:softmax_ED:2_AEP:1_SEP:400_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8__",
                                                    "LS:32896-2-16_EA:softmax_CA:linear_DA:tanh_OA:linear_ED:16_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8__",
                                                    "LS:32896-16448-16_EA:linear_CA:sigmoid_DA:sigmoid_OA:linear_ED:16_AEP:1_SEP:400_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8__"
                                                    ]

        '''

        '''
        dataset_configs['SingleDiseaseQin'] = [
                                                #"LS:8192-4_EA:tanh_CA:tanh_DA:sigmoid_OA:sigmoid_ED:4_AEP:1_SEP:200_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                #"LS:8192-2-4_EA:sigmoid_CA:sigmoid_DA:softmax_OA:sigmoid_ED:4_AEP:1_SEP:400_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                "LS:8192-8_EA:sigmoid_CA:sigmoid_DA:tanh_OA:tanh_ED:8_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                "LS:8192-2-4_EA:linear_CA:sigmoid_DA:relu_OA:sigmoid_ED:4_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                "LS:8192-8_EA:linear_CA:linear_DA:relu_OA:relu_ED:8_AEP:1_SEP:200_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                "LS:8192-2_EA:tanh_CA:tanh_DA:sigmoid_OA:sigmoid_ED:2_AEP:1_SEP:400_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                "LS:8192-2_EA:sigmoid_CA:sigmoid_DA:relu_OA:relu_ED:2_AEP:1_SEP:200_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                "LS:8192-4_EA:tanh_CA:tanh_DA:sigmoid_OA:sigmoid_ED:4_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                #"LS:8192-4_EA:tanh_CA:tanh_DA:linear_OA:linear_ED:4_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_"
                                                ]
        '''

        '''
        dataset_configs['SingleDiseaseRA'] = [#"LS:512-2_EA:tanh_CA:tanh_DA:linear_OA:linear_ED:2_AEP:1_SEP:400_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:5_",
                                                #"LS:512-2-2_EA:sigmoid_CA:softmax_DA:linear_OA:tanh_ED:2_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:5_",
                                                #"LS:512-2-16_EA:softmax_CA:softmax_DA:sigmoid_OA:sigmoid_ED:16_AEP:1_SEP:200_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:5_",
                                                "LS:512-2-2_EA:tanh_CA:tanh_DA:softmax_OA:sigmoid_ED:2_AEP:1_SEP:200_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:5_",
                                                "LS:8192-2-8_EA:tanh_CA:softmax_DA:sigmoid_OA:linear_ED:8_AEP:1_SEP:200_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                "LS:2080-4_EA:tanh_CA:tanh_DA:softmax_OA:softmax_ED:4_AEP:1_SEP:200_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_",
                                                "LS:512-2_EA:sigmoid_CA:sigmoid_DA:relu_OA:relu_ED:2_AEP:1_SEP:200_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:5_",
                                                "LS:8192-2-8_EA:sigmoid_CA:sigmoid_DA:tanh_OA:tanh_ED:8_AEP:1_SEP:200_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                "LS:512-256-128-8_EA:tanh_CA:sigmoid_DA:sigmoid_OA:softmax_ED:8_AEP:1_SEP:200_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:5_",
                                                "LS:2080-1040-520-16_EA:linear_CA:tanh_DA:sigmoid_OA:sigmoid_ED:16_AEP:1_SEP:200_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_"]
        '''

        '''
        dataset_configs['SingleDiseaseFeng'] = [#"LS:8192-16_EA:tanh_CA:tanh_DA:tanh_OA:tanh_ED:16_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                #"LS:8192-16_EA:sigmoid_CA:sigmoid_DA:linear_OA:linear_ED:16_AEP:1_SEP:200_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                "LS:8192-2-4_EA:linear_CA:sigmoid_DA:relu_OA:tanh_ED:4_AEP:1_SEP:200_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                "LS:8192-16_EA:sigmoid_CA:sigmoid_DA:linear_OA:linear_ED:16_AEP:1_SEP:200_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                "LS:512-256-128-16_EA:linear_CA:softmax_DA:tanh_OA:sigmoid_ED:16_AEP:1_SEP:200_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:5_",
                                                "LS:8192-4096-16_EA:sigmoid_CA:sigmoid_DA:sigmoid_OA:softmax_ED:16_AEP:1_SEP:400_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                "LS:8192-2-16_EA:linear_CA:tanh_DA:tanh_OA:tanh_ED:16_AEP:1_SEP:200_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                "LS:8192-2_EA:linear_CA:linear_DA:tanh_OA:tanh_ED:2_AEP:1_SEP:200_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                "LS:8192-2_EA:linear_CA:linear_DA:tanh_OA:tanh_ED:2_AEP:1_SEP:400_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                "LS:8192-2-16_EA:linear_CA:linear_DA:relu_OA:relu_ED:16_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_"
                                                ]
        '''
        '''
        dataset_configs['SingleDiseaseZeller'] = [#"LS:512-256-16_EA:sigmoid_CA:sigmoid_DA:sigmoid_OA:relu_ED:16_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:5_",
                                                #"LS:2080-16_EA:tanh_CA:tanh_DA:sigmoid_OA:sigmoid_ED:16_AEP:1_SEP:200_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_",
                                                #"LS:2080-8_EA:tanh_CA:tanh_DA:linear_OA:linear_ED:8_AEP:1_SEP:200_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_",
                                                "LS:512-2_EA:sigmoid_CA:sigmoid_DA:linear_OA:linear_ED:2_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:5_",
                                                "LS:2080-16_EA:sigmoid_CA:sigmoid_DA:sigmoid_OA:sigmoid_ED:16_AEP:1_SEP:200_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_",
                                                "LS:512-256-16_EA:linear_CA:sigmoid_DA:linear_OA:tanh_ED:16_AEP:1_SEP:400_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:5_",
                                                "LS:2080-16_EA:sigmoid_CA:sigmoid_DA:softmax_OA:softmax_ED:16_AEP:1_SEP:200_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_",
                                                "LS:512-2-2_EA:linear_CA:linear_DA:relu_OA:softmax_ED:2_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:5_",
                                                "LS:2080-16_EA:linear_CA:linear_DA:sigmoid_OA:sigmoid_ED:16_AEP:1_SEP:200_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_",
                                                "LS:512-8_EA:sigmoid_CA:sigmoid_DA:relu_OA:relu_ED:8_AEP:1_SEP:200_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:5_"]
        '''
        '''
        dataset_configs['SingleDiseaseLiverCirrhosis'] = [#"LS:2080-2-2_EA:softmax_CA:sigmoid_DA:sigmoid_OA:softmax_ED:2_AEP:1_SEP:400_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_",
                                                            #"LS:2080-2-2_EA:sigmoid_CA:sigmoid_DA:softmax_OA:tanh_ED:2_AEP:1_SEP:400_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_",
                                                            #"LS:2080-2-4_EA:sigmoid_CA:tanh_DA:softmax_OA:softmax_ED:4_AEP:1_SEP:200_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_",
                                                            "LS:2080-2_EA:tanh_CA:tanh_DA:tanh_OA:tanh_ED:2_AEP:1_SEP:200_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_",
                                                            "LS:2080-2-2_EA:tanh_CA:linear_DA:softmax_OA:softmax_ED:2_AEP:1_SEP:400_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_",
                                                            "LS:2080-4_EA:sigmoid_CA:sigmoid_DA:tanh_OA:tanh_ED:4_AEP:1_SEP:400_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_",
                                                            "LS:8192-2-4_EA:softmax_CA:softmax_DA:tanh_OA:relu_ED:4_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                            "LS:8192-2-4_EA:sigmoid_CA:sigmoid_DA:relu_OA:linear_ED:4_AEP:1_SEP:200_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                            "LS:2080-8_EA:sigmoid_CA:sigmoid_DA:relu_OA:relu_ED:8_AEP:1_SEP:400_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_",
                                                            "LS:2080-1040-520-8_EA:sigmoid_CA:softmax_DA:softmax_OA:tanh_ED:8_AEP:1_SEP:400_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:6_"]
        '''
        '''
        dataset_configs['SingleDiseaseKarlsson'] = [#"LS:8192-16_EA:relu_CA:relu_DA:tanh_OA:tanh_ED:16_AEP:1_SEP:400_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                    "LS:8192-4096-2_EA:sigmoid_CA:tanh_DA:tanh_OA:softmax_ED:2_AEP:1_SEP:400_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                    "LS:8192-4096-2048-8_EA:tanh_CA:sigmoid_DA:sigmoid_OA:relu_ED:8_AEP:1_SEP:400_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                    "LS:8192-4096-2048-8_EA:relu_CA:relu_DA:sigmoid_OA:linear_ED:8_AEP:1_SEP:200_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                    "LS:8192-4096-2048-1024-2_EA:sigmoid_CA:sigmoid_DA:relu_OA:relu_ED:2_AEP:1_SEP:400_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                    "LS:8192-4096-2048-1024-16_EA:sigmoid_CA:linear_DA:sigmoid_OA:linear_ED:16_AEP:1_SEP:200_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                    "LS:8192-4096-4_EA:sigmoid_CA:sigmoid_DA:softmax_OA:linear_ED:4_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                    "LS:8192-4096-2048-1024-2_EA:tanh_CA:linear_DA:sigmoid_OA:softmax_ED:2_AEP:1_SEP:400_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                    "LS:8192-4096-2048-1024-4_EA:relu_CA:sigmoid_DA:softmax_OA:relu_ED:4_AEP:1_SEP:200_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_",
                                                    "LS:8192-2-2_EA:linear_CA:linear_DA:sigmoid_OA:relu_ED:2_AEP:1_SEP:200_BS:32_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_"]
        '''


        #dataset_configs['SingleDiseaseMetaHIT'] = ['LS:8192-8_EA:sigmoid_CA:sigmoid_DA:softmax_OA:softmax_ED:8_AEP:1_SEP:400_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_UK:10_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7']
        '''
        dataset_configs['SingleDiseaseQin'] = ['LS:8192-4_EA:tanh_CA:tanh_DA:linear_OA:linear_ED:4_AEP:1_SEP:400_BS:8_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7',
                                               'LS:8192-2-4_EA:sigmoid_CA:sigmoid_DA:softmax_OA:sigmoid_ED:4_AEP:1_SEP:400_BS:16_LF:mean_squared_error_BN:0_DO:0_AR:0_NI:1_ES:0_PA:2_NO:L1_BE:tensorflow_V:5_AU:0_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7',
                                               ]
        # This produced 0.759 AUC and 0.693 ACC in average with 20 iterations and full randomness (shuffling and weight initialization)
        # - a drop from the previous model as a result of avoiding overfitting.
        # the first model is the real one and the second is identical except with labels shuffled (for testing AUC statistical significance as in Pasolli)
        
        dataset_configs['SingleDiseaseQin'] = ['LS:512-2_ED:2_EA:linear_CA:tanh_DA:linear_OA:linear_LF:kullback_leibler_divergence_AEP:50_SEP:300_NO:L1_NI:1_BS:16_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AU:0_UK:10_ITS:20_IT:0_NR:1_SL:0_ES:0_SA:0_PA:2_KS:5',
                                               #'LS:512-2_ED:2_EA:linear_CA:tanh_DA:linear_OA:linear_LF:kullback_leibler_divergence_AEP:50_SEP:3_NO:L1_NI:1_BS:16_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AU:0_UK:3_ITS:20_IT:0_NR:1_SL:1_ES:0_SA:0_PA:2_KS:5',
                                               #'LS:512-2_ED:2_EA:linear_CA:tanh_DA:linear_OA:linear_LF:kullback_leibler_divergence_AEP:50_SEP:3_NO:L1_NI:1_BS:32_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AU:0_UK:3_ITS:20_IT:0_NR:1_SL:0_ES:0_SA:0_PA:2_KS:5',

                                              ]
        
        dataset_configs['SingleDiseaseKarlsson'] = ['LS:512-2_ED:2_EA:linear_CA:tanh_DA:linear_OA:linear_LF:kullback_leibler_divergence_AEP:50_SEP:300_NO:L1_NI:1_BS:32_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AU:1_UK:10__NR:1',# 'LS:1024-2_ED:2_EA:linear_CA:tanh_DA:linear_OA:linear_LF:kullback_leibler_divergence_AEP:50_SEP:300_NO:L1_NI:1_BS:32_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AU:1_UK:10__NR:1_SL:1'
                                              ]

        # This produced 0.604 AUC and 0.601 ACC in average with 20 iterations and full randomness (shuffling and weight initialization)
        # but the std is too big (around 0.1 for both), suggesting again that the RA data is not amenable to this technique.
        dataset_configs['SingleDiseaseRA'] = ['LS:512-2_ED:2_EA:linear_CA:linear_DA:linear_OA:linear_LF:kullback_leibler_divergence_AEP:50_SEP:100_NO:L1_NI:1_BS:32_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AU:1_UK:10__NR:1']
        #'LS:1024-2_ED:2_EA:linear_CA:linear_DA:linear_OA:linear_LF:kullback_leibler_divergence_AEP:50_SEP:100_NO:L1_NI:1_BS:32_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AU:1_UK:10__NR:1_SL:1'
        #]
        '''
       
        
        loop_over_dataset_configs()
        
    elif exp_mode == "AUTO_MODELS":
        plot_ae_fold = False
        plot_ae_overall = True

        plot_fold = False
        plot_iter = False
        plot_overall = False

        # Models chosen after unsupervised grid search whose performance stats were reported in the Siemens paper.
        # These come in pairs - one for the real model, the other for the null/shuffled one
        #
        # You can add any specific models to be tested by adding them to their respective dataset_config list
        
        #dataset_configs['SingleDiseaseMetaHIT'] = ['LS:8192-8_EA:linear_CA:linear_DA:softmax_OA:softmax_ED:8_PC:0_AEP:200_SEP:1_BS:16_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:0_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:1_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:H-Q-F-L-Z-K_AL:_AA:linear_']
        #dataset_configs['SingleDiseaseQin'] = ['LS:8192-8_EA:sigmoid_CA:sigmoid_DA:softmax_OA:softmax_ED:8_PC:0_AEP:200_SEP:1_BS:8_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:0_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:1_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:7_MN:0_AD:H-F-L-Z-M-K_AL:_AA:linear_']
        #dataset_configs['SingleDiseaseFeng'] = ['LS:32896-64_EA:sigmoid_CA:sigmoid_DA:softmax_OA:softmax_ED:64_PC:0_AEP:200_SEP:1_BS:32_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:0_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:1_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:8_MN:0_AD:H-Q-L-Z-M-K_AL:_AA:linear_']
        dataset_configs['ZellerReduced'] = ['LS:512-32_EA:linear_CA:linear_DA:softmax_OA:softmax_ED:32_PC:0_AEP:200_SEP:1_BS:16_LF:kullback_leibler_divergence_BN:0_DP:0_IDP:0_AR:0_NI:0_ES:0_PA:2_NO:L1_BE:theano_V:5_AU:1_NR:0_SL:0_SA:0_UK:10_ITS:20_KS:5_MN:0_AD:H-Q-L-F-M-K_AL:_AA:linear_']

        loop_over_dataset_configs()
    elif exp_mode == "SEARCH_SUPER_MODELS":
        plot_ae_fold = False
        plot_ae_overall = False

        plot_fold = False
        plot_iter = False
        plot_overall = True

        exp_configs[shuffle_labels_key][0] = [0]

        # edit exp_configs to your needs
        loop_over_configs(random=True)
        
    elif exp_mode == "SEARCH_AUTO_MODELS":
        plot_ae_fold = False
        plot_ae_overall = True

        plot_fold = False
        plot_iter = False
        plot_overall = False
        

        # For searching unsupervised models - normalization across samples needs to be off
        exp_configs[norm_input_key][0] = [0]
        # set the supervised epochs to be 1 to make the ROC calculations happy and to avoid wasting time on supervised
        # learning. We have to use the supervised part only because the K-fold cross-validation code was tangled with
        # supervised learning
        exp_configs[super_epochs_key][0] = [1]
        # edit exp_configs to your needs
        loop_over_configs(random=True)
        
    else:
        # anything goes, just edit exp_configs to your needs - it can be for a single specific model, or for
        # arbitrary grid search
        loop_over_configs()

