# always run miniconda for keras:
# ./miniconda3/bin/python

import os
import numpy as np
import tensorflow as tf
import random as rn
from keras import backend as K

import matplotlib
matplotlib.use('Agg') # this suppresses the console for plotting
import pylab
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.models import Input, Model, Sequential
from keras.callbacks import History, EarlyStopping
from keras import regularizers
from load_kmer_cnts import load_kmer_cnts_for_hmp, load_kmer_cnts_for_metahit,\
    load_kmer_cnts_for_ra, load_kmer_cnts_for_t2d, load_kmer_cnts_for_ra_with_labels,\
    load_kmer_cnts_for_metahit_with_labels, load_kmer_cnts_for_t2d_with_labels,\
    load_kmer_cnts_for_hmp_with_labels, load_hmp_metahit_with_labels, load_ra_t2d_with_labels,\
    load_metahit_with_obesity_labels, load_metahit_with_bmi_labels, load_all_kmer_cnts_with_labels,\
    load_single_disease_plus_healthy_others, load_kmer_cnts_for_hmp_with_random_labels
from kmer_norms import norm_by_l1, norm_by_l2
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from itertools import cycle, product
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import re

# Experiment config keys - these are used to iterate thru the experiment configs and uniquely name 
# the plot files associated to the configs.
auto_epochs_key = 'AEP'
super_epochs_key = 'SEP'
batch_size_key = 'BS'
loss_func_key = 'LF'
enc_dim_key = 'ED'
enc_act_key = 'EA'
code_act_key = 'CA'
dec_act_key = 'DA'
out_act_key = 'OA'
layers_key = 'LS'
batch_norm_key = 'BN'
dropout_key = 'DO'
act_reg_key = 'AR'
norm_input_key = 'NI'
early_stop_key = 'ES'
patience_key = 'PA'
dataset_key = 'DS'
norm_sample_key = 'NO'
backend_key = 'BE'
version_key = 'V'
use_ae_key = 'AE'
use_kfold_key = 'UK'
kfold_key = 'KF'
no_random_key = 'NR'
iter_key = 'IT'
shuffle_labels_key = 'SL'
num_iters_key = 'ITS'
shuffle_abunds_key = 'SA'

input_dim = 1024

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

# datasource name to marker map for plotting
datasource_marker = {'HMP': 'o', 'MetaHIT': '*', 'T2D': 'D', 'RA': '^'}

# dataset dictionary - this determines the specific model data and type of classification.
# Some of the datasets are not fully experimented with, e.g., obesity and BMI
dataset_dict = {
    # <dataset_name>: (<classification_name>, <function for loading the data(wrapped in lambda if it takes arguments)>, 
    # <index of the label in the list of labels>, <class name list>, <color list for plotting (in order of class names)>, <data source marker dict>)

    'AllContinent': ('Continent', load_all_kmer_cnts_with_labels, 1, [ 'North America', 'Europe', 'Asia', ],
                     ['green', 'red', 'blue'], datasource_marker),
    
    'AllContinentLabelShuffled': ('Continent-LabelShuffled', lambda: load_all_kmer_cnts_with_labels(shuffle_labels=True), 1, [ 'North America', 'Europe', 'Asia', ],
                                  ['green', 'red', 'blue'], datasource_marker),
    
    'AllCountry': ('Country', load_all_kmer_cnts_with_labels, 3, [ 'United States', 'Denmark', 'Spain', 'China', ],
                   ['green', 'red', 'darkorange', 'blue'], datasource_marker),
    
    'AllCountryLabelShuffled': ('Country-LabelShuffled', lambda: load_all_kmer_cnts_with_labels(shuffle_labels=True), 3, [ 'United States', 'Denmark', 'Spain', 'China', ],
                                ['green', 'red', 'darkorange', 'blue'], datasource_marker),
    
    'AllHealth': ('Disease', load_all_kmer_cnts_with_labels, 6, [ 'Healthy', 'IBD', 'RA', 'T2D', ],
                  ['green', 'red', 'darkorange', 'purple'], datasource_marker),

    'AllHealthBinary': ('Diseased', load_all_kmer_cnts_with_labels, 0, [ '0', '1', ],
                        ['green', 'red', ], datasource_marker),

    # For testing the autoencoder - because the K-fold cross validation code is shared with supervised learning
    # , it needs some positive labels to work, so we randomly generate some positives, which do not affect the
    # autoencoder experiment results.
    'HMP': ('HMP', load_kmer_cnts_for_hmp_with_random_labels, 6, [ 'Healthy', 'IBD', ],
            ['green', 'red', ], datasource_marker),
    
    'SingleDiseaseIBD': ('IBD', load_kmer_cnts_for_metahit_with_labels, 6, [ 'Healthy', 'IBD', ],
                          ['green', 'red', ], datasource_marker),
    'SingleDiseaseIBDLabelShuffled': ('IBD-LabelShuffled', lambda: load_kmer_cnts_for_metahit_with_labels(shuffle_labels=True), 6, [ 'Healthy', 'IBD', ],
                                      ['green', 'red', ], datasource_marker),
    'SingleDiseaseT2D': ('T2D', load_kmer_cnts_for_t2d_with_labels, 6, [ 'Healthy', 'T2D', ],
                         ['green', 'purple', ], datasource_marker),
    'SingleDiseaseT2DLabelShuffled': ('T2D-LabelShuffled', lambda: load_kmer_cnts_for_t2d_with_labels(shuffle_labels=True), 6, [ 'Healthy', 'T2D', ],
                                      ['green', 'purple', ], datasource_marker),

    'T2D-HMP': ('T2D', lambda: load_single_disease_plus_healthy_others('T2D'), 6, [ 'Healthy', 'T2D', ],
                ['green', 'purple', ], datasource_marker),

    'RA-HMP': ('RA', lambda: load_single_disease_plus_healthy_others('RA'), 6, [ 'Healthy', 'RA', ],
                ['green', 'purple', ], datasource_marker),

    'RA-T2Dh': ('RA', lambda: load_single_disease_plus_healthy_others('RA', healthy_others=['T2D']), 6, [ 'Healthy', 'RA', ],
                ['green', 'purple', ], datasource_marker),

    'SingleDiseaseRA': ('RA', load_kmer_cnts_for_ra_with_labels, 6, [ 'Healthy', 'RA', ],
                        ['green', 'darkorange', ], datasource_marker),
    
    'AllHealthObese': ('Disease (with obesity replacing IBD for MetaHIT)', lambda: load_all_kmer_cnts_with_labels(metahit_obesity=True), 6, [ 'Healthy', 'Obese', 'RA', 'T2D', ],
                       ['green', 'red', 'darkorange', 'purple'], datasource_marker),

    'MetaHIT-BMI': ('BMI', load_metahit_with_bmi_labels, 7, [ 'obese', 'over_weight', 'normal_weight', 'under_weight', ],
                    ['green', 'red', 'purple', 'darkorange', ], datasource_marker),
}

# Specific model configs per dataset - used to evaluate individual exp configs after grid search
# Items are of the form <dataset_name>: [ <config_info1>, <config_info2>, ...] - config info is the key/value list returned by config_info() 
# that's embedded in plot file names and printed in log files.
#
# When a datasource has a corresponding entry in this dict, its config from this map overrides what's in exp_configs
dataset_config = {}

# Normalization per sample to be experimented with
norm_func_dict = {
    "L1": norm_by_l1,
    "L2": norm_by_l2 
}

# For denoting code/decoding/output layer activation functions the same as the encoding activation function
# - used for testing whether the data has nonlinear structure.
SAME_AS_ENC = "asenc"

# This is the experiment setup - for grid search of model candidates. It can also be used to evalue individual models.
# Most of the time, this is the only place that needs to be modified.
#
# <exp_config_key>: [ <list of values to be experimented with for this key>, <format string for the exp config value> ]
# Boolean values use 1 for True and 0 for False to avoid exceeding plot file name length limit on Linux.
exp_configs = {
                # Datasets to use
                dataset_key:       [ [
                                       # 'AllContinent',
                                       # 'AllCountry',
                                       # 'AllHealth',
                                       'SingleDiseaseIBD',
                                       'SingleDiseaseT2D',
                                       'SingleDiseaseRA',
                                       'HMP',
                                     ], 'Dataset: {}'],
                norm_sample_key:   [ [
                                       'L1',
                                       # 'L2'
                                     ], 'Normalize each sample with: {}' ],
                # 1 for supervised and 0 for autoencoder only -- CHANGE IT BACK TO 1 FOR ANY SUPERVISED LEARNING!!!
                norm_input_key:    [ [0], 'Normalize across samples (each component with zero mean/unit std across training samples): {}' ],

                # Deep net structure
                # The last entry (-1 is the placeholder) of the layer list is for code layer dimensions - this is so we don't
                # have to list too many network layer lists when we vary only the code layer dimension.
                layers_key:        [ [
                                       [input_dim, -1],
                                       #[input_dim, input_dim // 2, -1],
                                       [input_dim, 2, -1],
                                       #[input_dim, input_dim // 2, input_dim // 4, -1],
                                       #[input_dim, input_dim // 2, input_dim // 4, input_dim // 8, -1],
                                       #[input_dim, input_dim // 2, input_dim // 4, input_dim // 8, input_dim // 16, -1],
                                     ], "Layers for autoencoder's first half : {}" ],
                enc_dim_key:       [ [2],  'Encoding dimensions: {}' ],

                enc_act_key:       [ [
                                         'sigmoid',
                                         'relu',
                                         'linear',
                                         'softmax',
                                         'tanh',
                                     ], 'Encoding activation: {}' ],
                code_act_key:      [ [
                                         SAME_AS_ENC,
                                         # 'linear',
                                         # 'softmax',
                                         # 'sigmoid',
                                         # 'relu',
                                         # 'tanh',

                                     ], 'Code (last encoding) layer activation: {}' ],
    
                # Decoding activations are fixed as linear as they are popped off anyway
                # after autoencoder training
                dec_act_key:       [ [
                                         SAME_AS_ENC,
                                         #'linear',
                                         #'sigmoid',
                                         #'relu',
                                         # 'softmax',
                                         #'tanh',
                                     ], 'Decoding layer activation: {}' ],
                out_act_key:       [ [

                                         SAME_AS_ENC,
                                         #'linear',
                                         #'sigmoid',
                                         #'relu',
                                         #'softmax',
                                         #'tanh',
                                     ], 'Last decoding layer activation: {}' ],
                loss_func_key :    [ [
                                         'mean_squared_error',
                                         'kullback_leibler_divergence'
                                     ], 'Autoencoder loss function: {}' ],
                # boolean for whether to use autoencoder for pretraining before supervised learning
                use_ae_key:    [ [1], 'Use autoencoder pretraining for supervised learning: {}' ],


                # Training options
                auto_epochs_key :  [ [200], 'Max number of epochs for autoencoder training: {}' ],
                super_epochs_key : [ [1], 'Max number of epochs for supervised training: {}' ],
                batch_size_key:    [ [32], 'Batch size used during training: {}' ],
                # two booleans
                batch_norm_key:    [ [0], 'Use batch normalization: {}' ],
                dropout_key:       [ [0], 'Use dropout: {}' ],
                act_reg_key:       [ [0], 'Activation regularization (for sparsity): {}' ],
                # boolean
                early_stop_key:    [ [0],  'Use early stopping: {}' ],
                patience_key:      [ [2], 'Early stopping patience (consecutive degradations): {}' ],
                use_kfold_key:    [ [10], 'Stratified K folds (0 means one random shuffle with stratified 80/20 split): {}' ],
                kfold_key:    [ [0], 'K fold index for the current fold: {}' ],
                # boolean for whether no randomness should be used
                no_random_key:    [ [0], "Eliminate randomness in training: {}" ],
                # number of iterations
                num_iters_key:    [ [20], "Number of iterations: {}" ],
                # the current iteration index
                iter_key:    [ [0], "Iteration: {}" ],
                # boolean
                shuffle_labels_key:    [ [0],  'Shuffle labels (for supervised null): {}' ],
                # boolean
                shuffle_abunds_key:    [ [0],  'Shuffle abundances (for unsupervised null): {}' ],

                # misc
                backend_key:   [ [K.backend()], 'Backend: {}' ],
                version_key:   [ ['2'], 'Version (catching all other unnamed configs): {}' ],
                         
            }

def set_config(config_key, val_list):
    exp_configs[config_key][0] = val_list

test_pct = 0.2
drop_pct = 0.5
input_drop_pct = 0

# The experiment config currently being trained/evaluated
exp_config = {}

# The training/test input and label matrices
x_train, y_train, info_train, x_test, y_test, info_test = None, None, None, None, None, None

# Turn on/off randomness in training.
def setup_randomness():
    exp_config[no_random_key] = exp_configs[no_random_key][0][0]
    if not exp_config[no_random_key]:
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
    if exp_config[batch_norm_key]:
        autoencoder.add(BatchNormalization())
    autoencoder.add(Activation(activation))
    if exp_config[dropout_key]:
        autoencoder.add(Dropout(drop_pct))

# index of the data source name in the label list - used for more stratified splits between training and test samples and for plotting
source_ind = 5

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

# data source to marker dict for the currently active exp dataset
markers = None

# map of <dataset_name, iteration index, K fold index> to list of <history>, <2d-codes-predicted>, etc,
# for storing per-fold results
dataset_config_iter_fold_results = {}

# map of config to list of <avg_val_acc>, <avg_acc>, <avg_val_loss>, <avg_loss>, <conf_mat>, etc, per iteration
# for calculating average results over all runs
config_results = {}

# Maps class name to target vector, which is zero everywhere except 1 at the index of the true label
def class_to_target(cls):
    target = np.zeros((n_classes,))
    target[class_to_ind[cls]] = 1.0
    return target

# The loop_over_* functions implement the grid search. They go through each of the exp config keys and
# set the current exp_config for each key to each of the possible values for that key in order, and 
# , at the end of these loops, perform the training/testing under the then 'active' config setup.
#
# I found out that sklearn has support for grid search after implementing this, but I stuck with this because
# , to use sklearn, a specific interface has to be implemented.

# loops over the datasets
def loop_over_datasets():
    for dataset_name in exp_configs[dataset_key][0]:
        cls_name, data_loader, lbl_ind, lbls, clrs, mkrs = dataset_dict[dataset_name]

        # set the current active dataset
        exp_config[dataset_key] = dataset_name
        dataset_config_iter_fold_results[dataset_name] = {}

        global class_name, label_ind, classes, n_classes, class_to_ind, colors, markers

        class_name = cls_name
        label_ind = lbl_ind
        classes = lbls
        n_classes = len(classes)
        class_to_ind = { classes[i]: i for i in range(n_classes) }
        colors = clrs
        markers = mkrs

        # For dumping and replotting model results
        dataset_info = [source_ind, class_name, label_ind, classes, n_classes, class_to_ind, colors, markers]

        # load the samples and their labels
        data, orig_target = data_loader()
        # for autoencoder null
        data_shuffled = np.array(data.values)

        # Temp labels just for stratified shuffle split - we split proportionally by datasource+label+health
        # the 0 index is for diseased or not 
        target_labels = [(orig_target[i][0] + orig_target[i][label_ind] + orig_target[i][source_ind]) for i in range(len(orig_target))]
        orig_target_random = np.array(orig_target)
        orig_target = np.array(orig_target)

        # The target matrix for training/testing - note that only the label index is used this time
        target = np.array([ class_to_target(orig_target[i][label_ind]) for i in range(len(orig_target)) ])

        exp_config[num_iters_key] = exp_configs[num_iters_key][0][0]
        for i in range(exp_config[num_iters_key]):
            exp_config[iter_key] = i
            
            # This is to optionally make the shuffle/fold deterministic
            setup_randomness()

            # shuffle labels for supervised null
            np.random.shuffle(orig_target_random)
            target_random = np.array([ class_to_target(orig_target_random[k][label_ind]) for k in range(len(orig_target_random)) ])

            # shuffle kmer counts for unsupervised null
            for r in data_shuffled:
                np.random.shuffle(r)

            skf = None
            if get_config_val(use_kfold_key):
                # K folds - we shuffle only if not removing randomness
                skf = StratifiedKFold(n_splits=get_config_val(use_kfold_key), shuffle=(not get_config_val(no_random_key)))
                skf = skf.split(data.values, target_labels)
            else:
                # single fold - the random state is fixed with the value of the fold index if no randomness,
                # otherwise np.random will be used by the function.
                skf = StratifiedShuffleSplit(target_labels, n_iter=1, test_size=test_pct, random_state=(i if get_config_val(no_random_key) else None))

            # the current active fold
            exp_config[kfold_key] = 0

            # This loop is executed K times for K folds and only once for single split
            for train_idx, test_idx in skf:
                global x_train, y_train, info_train, x_test, y_test, info_test

                # set up the training/test sample and target matrices - the info matrix is used for plotting
                x_train, y_train, info_train = data.values[train_idx], target[train_idx], orig_target[train_idx]
                x_test, y_test, info_test = data.values[test_idx], target[test_idx], orig_target[test_idx]

                try:
                    # There are specific model configs for this dataset in dataset_config, so use them
                    for config in dataset_config[dataset_name]:
                        # Don't use the randomness and the number of folds from the config because,
                        # as in Pasolli, we removed randomness and used 5 folds for grid search,
                        # but allow randomness and use 10 folds for evaluation.
                        prev_no_random = get_config_val(no_random_key)
                        prev_kfolds = get_config_val(use_kfold_key)
                        prev_shuffle_labels = get_config_val(shuffle_labels_key)
                        prev_shuffle_abunds = get_config_val(shuffle_abunds_key)

                        # Parse the config info and set exp_configs accordingly
                        setup_exp_configs_from_config_info(config)

                        # Because the above will use the config info to set up randomness and # of folds,
                        # we set them back to their values before the call (what're in exp_configs)
                        set_config(no_random_key, [prev_no_random])
                        set_config(use_kfold_key, [prev_kfolds])

                        # use shuffled labels for supervised null. As in Pasolli, the null uses the same K folds as the normal
                        # model for comparison purpose. That's why label shuffling is here and not done as a separate dataset as
                        # originally implemented (even though that also adequately showed statistical significance of the models).
                        if get_config_val(shuffle_labels_key):
                            # remember the unshuffled targets and set the train/test targets to the shuffled ones
                            y_train_old, y_test_old, info_train_old, info_test_old = y_train, y_test, info_train, info_test
                            y_train, y_test, info_train, info_test = target_random[train_idx], target_random[test_idx], orig_target_random[train_idx], orig_target_random[test_idx]

                            loop_over_norm_funcs()

                            # revert back to the unshuffled targets for normal models
                            y_train, y_test, info_train, info_test = y_train_old, y_test_old, info_train_old, info_test_old
                        elif get_config_val(shuffle_abunds_key):
                            # we don't shuffle both kmer counts and labels at the same time
                            # remember the unshuffled samples and set the train/test samples to the shuffled ones
                            x_train_old, x_test_old = x_train, x_test
                            x_train, x_test = data_shuffled[train_idx], data_shuffled[test_idx]
                            
                            loop_over_norm_funcs()

                            # revert back to the unshuffled samples for normal models
                            x_train, x_test = x_train_old, x_test_old
                        else:
                            loop_over_norm_funcs()
                        set_config(shuffle_labels_key, [prev_shuffle_labels])
                        set_config(shuffle_abunds_key, [prev_shuffle_abunds])
                except KeyError:
                    # No specific configs for the dataset, so we loop thru exp_configs - often used for grid search
                    loop_over_norm_funcs()

                # the next fold
                exp_config[kfold_key] += 1

            # aggregate results across K folds for this iteration
            config_iter_fold_results = dataset_config_iter_fold_results[dataset_name]
            for config in config_iter_fold_results:
                # K-fold results
                fold_results = np.array(config_iter_fold_results[config][exp_config[iter_key]])

                # the config for this iteration
                config_iter = config + '_' + iter_key + ':' + str(exp_config[iter_key])

                # sum the confusion matrices over the folds
                conf_mat = np.sum(fold_results[:, 2], axis=0)

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

                config_results[config].append([val_acc, acc, val_loss, loss, conf_mat, fold_results[:, 3], fold_results[:, 5], fold_results[:, 6], fold_results[:, 7]])

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
                std_down[i] = np.maximum(tpr[i] - np.std(i_tpr[i], axis=0)*plot_factor/np.sqrt(get_config_val(use_kfold_key)), 0)
                std_up[i] = np.minimum(tpr[i] + np.std(i_tpr[i], axis=0)*plot_factor/np.sqrt(get_config_val(use_kfold_key)), 1.0)

            if plot_overall:
                # plot the confusion matrix
                plot_confusion_matrix(conf_mat, config=config)

                # plot loss vs epochs
                plot_loss_vs_epochs(loss, val_loss, config=config)
            
                # plot accuracy vs epochs
                plot_acc_vs_epochs(acc, val_acc, config=config)

                # plot the ROCs with AUCs/ACCs
                plot_roc_aucs(fpr, tpr, roc_auc, accs, std_down, std_up, config=config)

            print('tkfold-overall-loss: ' + str(loss) + ' for ' + class_name + ':' + config)
            print('tkfold-overall-val-loss: ' + str(val_loss) + ' for ' + class_name + ':' + config)
            print('tkfold-overall-acc: ' + str(acc) + ' for ' + class_name + ':' + config)
            print('tkfold-overall-val-acc: ' + str(val_acc) + ' for ' + class_name + ':' + config)
            print('tkfold-overall-conf-mat: ' + str(conf_mat) + ' for ' + class_name + ':' + config)

            perf_metrics = np.vstack(np.concatenate(iter_results[:, 7]))
            perf_means = np.mean(perf_metrics, axis=0)
            perf_stds = np.std(perf_metrics, axis=0)
            
            # log the results for offline analysis
            print(('{}({})\t{}({})\t{}({})\t{}({})\t{}({})\t{}({})\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\tkfold-overall-perf-metrics for ' + class_name + ':{}').
                  format(perf_means[0], perf_stds[0], perf_means[1], perf_stds[1], perf_means[2], perf_stds[2], perf_means[3], perf_stds[3], perf_means[4], perf_stds[4], perf_means[5], perf_stds[5], roc_auc[1], roc_auc['macro'], last_val_acc, last_val_loss, max_vai, min_vli, val_acc[max_vai], val_loss[min_vli], config))

            all_ae_results = np.vstack(np.concatenate(iter_results[:, 8]))
            # mean autoencoder loss vs epochs across iterations times K folds
            ae_loss = np.mean([ r[0]['loss'] for r in all_ae_results ], axis=0)
            ae_val_loss = np.mean([ r[0]['val_loss'] for r in all_ae_results ], axis=0)
            if plot_ae_overall:
                # plot loss vs epochs
                plot_loss_vs_epochs(ae_loss, ae_val_loss, config=config, name='ae_general_loss_vs_epochs')
            ae_means = np.mean([ [r[1], r[2]] for r in all_ae_results ], axis=0)
            ae_stds = np.std([ [r[1], r[2]] for r in all_ae_results ], axis=0)
            print(('{}({})\t{}({})\t{}({})\t{}({})\t{}({})\t{}({})\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\tkfold-overall-perf-metrics for ' + class_name + ':{}').
                  format(perf_means[0], perf_stds[0], perf_means[1], perf_stds[1], perf_means[2], perf_stds[2], perf_means[3], perf_stds[3], perf_means[4], perf_stds[4], perf_means[5], perf_stds[5], roc_auc[1], roc_auc['macro'], last_val_acc, last_val_loss, max_vai, min_vli, val_acc[max_vai], val_loss[min_vli], config))


            print(('{}({})\t{}({})\tkfold-overall-autoencoder-perf-metrics for ' + class_name + ':{}').
                  format(ae_means[0], ae_stds[0], ae_means[1], ae_stds[1], config))

            # dump the model results into a file for offline analysis and plotting, e.g., merging plots from
            # different model instances (real and null)
            aggr_results = [val_acc, acc, val_loss, loss, conf_mat, fpr, tpr, roc_auc, accs, std_down, std_up, perf_means, perf_stds]
            ae_aggr_results = [ae_loss, ae_val_loss, ae_means, ae_stds]
            with open("aggr_results" + config +".pickle", "wb") as f:
                dump_dict = { "dataset_info": dataset_info, "results": [aggr_results, dataset_config_iter_fold_results[dataset_name][config]],
                              "ae_results": ae_aggr_results }
                pickle.dump(dump_dict, f)
            
def loop_over_norm_funcs():
    """
    loops over the choices for per-sample normalization - L1/L2
    and optionally normalize across samples. Note that normalization across samples (per kmer)
    does not use the test samples. The test samples are normalized using only the mean and std of the
    *traning* samples. This is the key to the predictive/generation power of the models:
    test samples are not used at all in training the models - only for cross validation.
    """
    for norm_name in exp_configs[norm_sample_key][0]:
        norm_func = norm_func_dict[norm_name]
        exp_config[norm_sample_key] = norm_name
        global x_train,  x_test
        # Per-sample normalization
        x_train = (norm_func)(x_train) 
        x_test = (norm_func)(x_test)

        for norm_input in exp_configs[norm_input_key][0]:
            exp_config[norm_input_key] = norm_input
            if norm_input:
                # the mean and std are calculated using only the training samples -
                # this is the key!
                sample_mean = x_train.mean(axis=0)
                sample_std = x_train.std(axis=0)

                # Normalize both training and test samples with the training mean and std
                x_train = (x_train - sample_mean) / sample_std
                # test samples are normalized using only the mean and std of the training samples
                x_test = (x_test - sample_mean) / sample_std

            loop_over_act_loss_funcs()

# loops over the activations and loss functions and sets the current active activations/loss function
def loop_over_act_loss_funcs():
    for encoding_activation in exp_configs[enc_act_key][0]:
        exp_config[enc_act_key] = encoding_activation
        for code_activation in exp_configs[code_act_key][0]:
            exp_config[code_act_key] = code_activation
            for decoding_activation in exp_configs[dec_act_key][0]:
                exp_config[dec_act_key] = decoding_activation
                for output_activation in exp_configs[out_act_key][0]:
                    exp_config[out_act_key] = output_activation
                    for loss_func in exp_configs[loss_func_key][0]:
                        exp_config[loss_func_key] = loss_func
                            
                        loop_over_batch_norm_early_stop_batch_size()

# loops over other exp cofigs: batch normalization, dropout, early stopping, batch size, etc.
def loop_over_batch_norm_early_stop_batch_size():
    for batch_norm in exp_configs[batch_norm_key][0]:
        exp_config[batch_norm_key] = batch_norm
        for dropout in exp_configs[dropout_key][0]:
            exp_config[dropout_key] = dropout
            for early_stop in exp_configs[early_stop_key][0]:
                exp_config[early_stop_key] = early_stop
                for patience in exp_configs[patience_key][0]:
                    exp_config[patience_key] = patience
                    for batch_size in exp_configs[batch_size_key][0]:
                        exp_config[batch_size_key] = batch_size
                    
                        loop_over_layers()

        
last_loss = []
mean_squared_errors = []

# loops over the list of the network layer structures
def loop_over_layers():
    for layers in exp_configs[layers_key][0]:
        exp_config[layers_key] = layers
            
        # The last entry of the layers is for code layer dimension (middle layer of the autoencoder)
        for enc_dim in exp_configs[enc_dim_key][0]:
            exp_config[enc_dim_key] = enc_dim
            # set the last entry in the layer dimension list to the code layer dimension being iterated thru
            # - this is so we don't have to list too many network layer lists when we vary only the code layer
            # dimension.
            layers[-1] = enc_dim

            # exp with # of epochs for autoencoder training
            for auto_epochs in exp_configs[auto_epochs_key][0]:
                exp_config[auto_epochs_key] = auto_epochs
                # exp with # of epochs for supervised training
                for super_epochs in exp_configs[super_epochs_key][0]:
                    exp_config[super_epochs_key] = super_epochs

                    # Do the real work of training and testing
                    build_train_test(layers)

# Counts the number of layers of the autoencoder from output layer to the middle code layer
# - used for popping off the decoding layers after autoencoder training and before 
# supervised training
def cnt_to_coding_layer():
    mult = 2
    if exp_config[batch_norm_key]:
        mult += 1
    if exp_config[dropout_key]:
        mult += 1
    return mult * (len(exp_config[layers_key]) - 2) + 2

def build_train_test(layers):
    """
    This uses the current active exp config (set up by the loops above) to first build and train the autoencoder,
    and then build and train/test the supervised model
    """

    # This is to optionally make the weight initialization deterministic
    setup_randomness()

    # (optionally) use activation regulerization
    exp_config[act_reg_key] = exp_configs[act_reg_key][0][0]

    # the autoencoder
    autoencoder = Sequential()

    # layers before the middle code layer - they share the same activation function
    for i in range(1, len(layers)-1):
        autoencoder.add(Dense(layers[i], input_dim=layers[i-1], kernel_regularizer=regularizers.l2(exp_config[act_reg_key]),
                              activity_regularizer=regularizers.l1(exp_config[act_reg_key])))
        add_layer_common(autoencoder, exp_config[enc_act_key])

    # the middle code layer - it has an activation function independent of other encoding layers
    i = len(layers) - 1
    autoencoder.add(Dense(layers[i], input_dim=layers[i-1], kernel_regularizer=regularizers.l2(exp_config[act_reg_key]),
                          activity_regularizer=regularizers.l1(exp_config[act_reg_key])))
    autoencoder.add(Activation(exp_config[enc_act_key if exp_config[code_act_key] == SAME_AS_ENC else code_act_key]))

    # the model from input to the code layer - used for plotting 2D graphs
    code_layer_model = Model(inputs=autoencoder.layers[0].input,
                             outputs=autoencoder.layers[cnt_to_coding_layer() - 1].output)

    print("Code layer model:")
    code_layer_model.summary()

    # stats for the autoencoder - the 9s are for when autoencoder is (optionally) not used before supervised learning
    ae_val_loss, ae_loss, ae_mse = 999999, 999999, 999999
    ae_hist = []
    if get_config_val(use_ae_key):
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

        # Compile the autoencoder model
        autoencoder.compile(optimizer='adadelta', loss=exp_config[loss_func_key])
    
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
                        shuffle=(not get_config_val(no_random_key)),
                        verbose=0,
                        validation_data=(x_test, x_test),
                        callbacks=callbacks)

        ae_hist = history.history
        # Plot training/test loss by epochs for the autoencoder
        if plot_ae_fold:
            plot_loss_vs_epochs(ae_hist['loss'], ae_hist['val_loss'], name='ae_general_loss_vs_epochs')

        # Calculate the MSE on the test samples
        decoded_output = autoencoder.predict(x_test)
        ae_mse = mean_squared_error(decoded_output, x_test)
        mean_squared_errors.append(ae_mse)

        ae_val_loss = ae_hist['val_loss'][-1]
        ae_loss = ae_hist['loss'][-1]
        last_loss.append(ae_val_loss)

        # Which epoch achieved min loss - useful for manual early stopping
        min_vli = np.argmin(ae_hist['val_loss'])
        min_li = np.argmin(ae_hist['loss'])

        # Log the autoencoder stats/performance
        print('{},{},{}\t{},{},{}\t{}\t{}\t{}\tautoencoder-min-loss-general:{}'.
              format(ae_hist['val_loss'][min_vli], min_vli, ae_hist['loss'][min_vli],
                     ae_hist['loss'][min_li], min_li, ae_hist['val_loss'][min_li],
                     ae_val_loss, ae_loss, ae_mse, config_info()))

        # if plot_fold and exp_config[enc_dim_key] > 1:
        #     fig = pylab.figure()
        #     pylab.title('AE2 VS AE1 After Autoencoder Training', size=plot_title_size)
        #     pylab.xlabel('AE1', size=plot_text_size)
        #     pylab.ylabel('AE2', size=plot_text_size)

        #     codes_pred = code_layer_model.predict(x_test)
        #     print("Autoencoder's test code shape for dimension {}: {}".format(exp_config[enc_dim_key], codes_pred.shape))
        #     plot_2d_codes(codes_pred, fig, 'ae_general_two_codes',
        #                   "This graph shows autoencoder's first 2 components of the encoding codes (the middle hidden layer output) with labels")

        # Pop the decoding layers before supervised learning
        autoencoder.summary()
        for n in range(cnt_to_coding_layer()):
            autoencoder.pop()

    # Add the classification layer for supervised learning - note that the loss function is always
    # categorical cross entropy - not what's in exp_configs, which is only used for the autoencoder.
    autoencoder.add(Dense(n_classes, input_dim=layers[-1]))
    autoencoder.add(Activation('softmax'))
    autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    autoencoder.summary()

    history = History()
    callbacks = [history]
    if exp_config[early_stop_key]:
        callbacks.append(EarlyStopping(monitor='val_acc', patience=exp_config[patience_key], verbose=0))

    # Train
    autoencoder.fit(x_train, y_train,
                    epochs=exp_config[super_epochs_key],
                    batch_size=exp_config[batch_size_key],
                    shuffle=(not get_config_val(no_random_key)),
                    verbose=0,
                    validation_data=(x_test, y_test),
                    callbacks=callbacks)

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
            if info_test[i][4] == cls:
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

    hist = history.history

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
                 ae_val_loss, ae_loss, ae_mse, roc_auc[1], roc_auc['macro'], config_info()))

    # calculate the accuracy/f1/precision/recall for this test fold - same way as in Pasolli
    test_true_label_inds = np.argmax(y_test, axis=1)
    test_pred_label_inds = np.argmax(y_test_pred, axis=1)
    accuracy = accuracy_score(test_true_label_inds, test_pred_label_inds)
    f1 = f1_score(test_true_label_inds, test_pred_label_inds, pos_label=None, average='weighted')
    precision = precision_score(test_true_label_inds, test_pred_label_inds, pos_label=None, average='weighted')
    recall = recall_score(test_true_label_inds, test_pred_label_inds, pos_label=None, average='weighted')

    print(('{}\t{}\t{}\t{}\t{}\t{}\tfold-perf-metrics for ' + class_name + ':{}').
          format(accuracy, f1, precision, recall, roc_auc[1], roc_auc['macro'], config_info()))

    # the config info for this exp but no fold/iter indices because we need to aggregate stats over them
    config_key = config_info(skip_keys=[kfold_key, iter_key])
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
                                                                [accuracy, f1, precision, recall, roc_auc[1], roc_auc['macro']], [ae_hist, ae_val_loss, ae_mse]]

        
# Plot the 2D codes in the layer right before the final classification layer
def plot_2d_codes(codes, name='ae_super_general_two_codes', title='2D Codes Before Classfication Layer', 
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
def plot_loss_vs_epochs(loss, val_loss, name='ae_super_general_loss_vs_epochs', title='Loss vs Epochs', 
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
def plot_acc_vs_epochs(acc, val_acc, name='ae_super_general_accu_vs_epochs', title='Accuracy vs Epochs',
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

    
# Plot individual test sample's predicted probabilities - not tested
def plot_probs(probs_pred, fig, name, desc):
    handles = {}
    sample_info = {'HMP': {}, 'IBD': {}, 'T2D': {}, 'RA': {}, 'Obese': {}}
    for i in range(len(probs_pred)):
        ind = (1 if info_test[i][0] == '1' else 0)
        src = info_test[i][4]
        stat = ('Yes' if ind == 1 else 'Healthy')
        info = [ [probs_pred[i][0]], markers[src], colors[ind] ]
        if stat in sample_info[src]:
            sample_info[src][stat].append(info)
        else:
            sample_info[src][stat] = [info]
            
    keys = [ k for k in sample_info.keys() ]
    keys.sort()

    i = 1
    for k in keys:
        for stat in ['Yes', 'No']:
            if not stat in sample_info[k]:
                continue
            for info in sample_info[k][stat]:
                h = pylab.scatter([i], info[0], marker=info[1], color=info[2], facecolor='none')
                handles[k + ' - ' + stat] = h
                i += 1

    keys = [ k for k in handles.keys() ]
    keys.sort()
    pylab.legend([handles[k] for k in keys], keys, prop={'size': plot_text_size})
    pylab.gca().set_position((.1, .7, 2.4, 1.8))
    add_figtexts_and_save(fig, name, desc)

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
    add_figtexts_and_save(fig, 'ae_super_general_confusion_mat', "Confusion matrix for predicting sample's " + class_name + " status using 5mers", y_off=1.3, config=config)

def plot_roc_aucs(fpr, tpr, roc_auc, accs, std_down=None, std_up=None, config=None, name='ae_super_general_roc_auc', title='ROC Curves with AUCs/ACCs', 
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

# Get the config value for the given config key - used in identifying model in grid search
# as well as unquiely naming the plots for the models
def get_config_val(config_key):
    if config_key in exp_config:
        config = exp_config[config_key]
    else:
        config = exp_configs[config_key][0][0]

    if type(config) is list:
        config = '-'.join([ str(c) for c in config])
    return config

# Get the config description for the given config key - used in figure descriptions
def get_config_desc(config_key):
    return exp_configs[config_key][1].format(get_config_val(config_key))

# Get the config info for the current active experiment - for identifying model in grid search
# as well as uniquely naming plots for the models - has the option to skip the fold and iteration keys
# for aggregating results across them.
config_keys = [ dataset_key, layers_key, enc_dim_key, enc_act_key, code_act_key, dec_act_key, out_act_key, loss_func_key,
                auto_epochs_key, super_epochs_key, norm_sample_key, norm_input_key, batch_size_key, batch_norm_key,
                # Thes two are not used yet, so skip them to save file name length
                # early_stop_key, patience_key,
                dropout_key, act_reg_key, backend_key, version_key, use_ae_key, no_random_key,
                # kfold_key should be the last and iter_key second to last
                use_kfold_key, num_iters_key, shuffle_labels_key, shuffle_abunds_key, iter_key, kfold_key ]
def config_info(skip_keys=[]):        
    config_info = ''
    for k in config_keys:
        # skip the specified keys, used for skipping the fold and iteration indices (for aggregating results across them)
        if not k in skip_keys:
            config_info += '_' + k + ':' + str(get_config_val(k))
    return config_info

# parse config info string back into exp configs - for evaluating models after grid search
def setup_exp_configs_from_config_info(config):
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
            exp_configs[k][0] = v if cnt2 <= 1 else [v]
    # The loss function name 'kullback_leibler_divergence' mixes up with the key/value separation char '_', so handle
    # it here separately
    if cnt1 > 1:
        if changed:
            exp_configs['LF'][0] = ['mean_squared_error']
        else:
            exp_configs['LF'][0] = ['kullback_leibler_divergence']

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
        pylab.figtext(x_off, y_off,  get_config_desc(dropout_key) + " --- " + get_config_desc(backend_key)
                      + ' --- ' + get_config_desc(version_key) + ' --- ' + get_config_desc(num_iters_key) + ' --- ' + get_config_desc(iter_key))


    filename = graph_dir + '/' + name + (config_info() if config is None else config) + '.svg'
    pylab.savefig(filename , bbox_inches='tight')
    pylab.close(fig)

if __name__ == '__main__':
    # the experiment mode can be one of SUPER_MODELS, AUTO_MODELS, SEARCH_SUPER_MODELS, SEARCH_AUTO_MODELS, OTHER
    # - see below
    exp_mode = "AUTO_MODELS"

    if exp_mode == "SUPER_MODELS":
        plot_ae_fold = False
        plot_ae_overall = False

        plot_fold = False
        plot_iter = False

        # Overall plotting - aggregate results across both folds and iteration
        plot_overall = True

        # For testing supervised models - normalization across samples needs to be on
        set_config(norm_input_key, [1])

        set_config(dataset_key, ['AllContinent', 'AllCountry', 'SingleDiseaseIBD', 'SingleDiseaseT2D', 'SingleDiseaseRA'])

        # Models chosen after supervised grid search whose performance stats wre reported in the Siemens paper.
        # These come in pairs - one for the real model, the other for the null/shuffled one
        #
        # You can add any specific models to be tested by add ing them to their respective dataset_config list


        # This produced 0.998 AUC and 0.987 ACC in average with 20 iterations and full randomness (shuffling and weight initialization)
        # the first model is the real one and the second is identical except with labels shuffled (for testing AUC statistical significance as in Pasolli)
        dataset_config['AllContinent'] = ['LS:1024-2_ED:2_EA:linear_CA:tanh_DA:linear_OA:linear_LF:kullback_leibler_divergence_AEP:50_SEP:200_NO:L1_NI:1_BS:16_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AE:1_UK:5__NR:1',
                                          'LS:1024-2_ED:2_EA:linear_CA:tanh_DA:linear_OA:linear_LF:kullback_leibler_divergence_AEP:50_SEP:200_NO:L1_NI:1_BS:16_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AE:1_UK:5__NR:1_SL:1'
        ]

        # This produced 0.989 AUC and 0.939 ACC in average with 20 iterations and full randomness (shuffling and weight initialization)
        # the first model is the real one and the second is identical except with labels shuffled (for testing AUC statistical significance as in Pasolli)
        dataset_config['AllCountry'] = ['LS:1024-2-2_ED:2_EA:tanh_CA:linear_DA:linear_OA:linear_LF:kullback_leibler_divergence_AEP:50_SEP:200_NO:L1_NI:1_BS:16_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AE:1_UK:5__NR:1', 'LS:1024-2-2_ED:2_EA:tanh_CA:linear_DA:linear_OA:linear_LF:kullback_leibler_divergence_AEP:50_SEP:200_NO:L1_NI:1_BS:16_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AE:1_UK:5__NR:1_SL:1']
        
        # This produced 0.947 AUC and 0.914 ACC in average with 20 iterations and full randomness (shuffling and weight initialization)
        # the first model is the real one and the second is identical except with labels shuffled (for testing AUC statistical significance as in Pasolli)
        dataset_config['SingleDiseaseIBD'] = ['LS:1024-2_ED:2_EA:linear_CA:tanh_DA:linear_OA:linear_LF:kullback_leibler_divergence_AEP:50_SEP:300_NO:L1_NI:1_BS:16_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AE:1_UK:__NR:1',
                                              'LS:1024-2_ED:2_EA:linear_CA:tanh_DA:linear_OA:linear_LF:kullback_leibler_divergence_AEP:50_SEP:300_NO:L1_NI:1_BS:16_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AE:1_UK:5__NR:1_SL:1'
        ]

        # This produced 0.759 AUC and 0.693 ACC in average with 20 iterations and full randomness (shuffling and weight initialization)
        # - a drop from the previous model as a result of avoiding overfitting.
        # the first model is the real one and the second is identical except with labels shuffled (for testing AUC statistical significance as in Pasolli)
        dataset_config['SingleDiseaseT2D'] = ['LS:1024-2_ED:2_EA:linear_CA:tanh_DA:linear_OA:linear_LF:kullback_leibler_divergence_AEP:50_SEP:300_NO:L1_NI:1_BS:32_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AE:1_UK:5__NR:1', 'LS:1024-2_ED:2_EA:linear_CA:tanh_DA:linear_OA:linear_LF:kullback_leibler_divergence_AEP:50_SEP:300_NO:L1_NI:1_BS:32_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AE:1_UK:5__NR:1_SL:1']

        # This produced 0.604 AUC and 0.601 ACC in average with 20 iterations and full randomness (shuffling and weight initialization)
        # but the std is too big (around 0.1 for both), suggesting again that the RA data is not amenable to this technique.
        dataset_config['SingleDiseaseRA'] = ['LS:1024-2_ED:2_EA:linear_CA:linear_DA:linear_OA:linear_LF:kullback_leibler_divergence_AEP:50_SEP:100_NO:L1_NI:1_BS:32_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AE:1_UK:5__NR:1',
                                             'LS:1024-2_ED:2_EA:linear_CA:linear_DA:linear_OA:linear_LF:kullback_leibler_divergence_AEP:50_SEP:100_NO:L1_NI:1_BS:32_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AE:1_UK:5__NR:1_SL:1'
        ]

        loop_over_datasets()
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

        # For testing autoencoder - normalization across samples is turned off
        # make your your config string has NI:0
        exp_configs[norm_input_key] = [0]
        # set the supervised epochs to be 1 to make the ROC calculations happy and to avoid wasting time on supervised
        # learning. We have to use the supervised part only because the K-fold cross-validation code was tangled with
        # supervised learning. Make sure your config string has SEP:1 to avoid wasting computing time
        set_config(super_epochs_key, [1])
        
        set_config(dataset_key, ['HMP', 'SingleDiseaseIBD', 'SingleDiseaseT2D', 'SingleDiseaseRA'])
        
        # MSE=1.625*10^(-8) after 20 runs of 10-fold CV
        dataset_config['SingleDiseaseT2D'] = ['DS:SingleDiseaseT2D_LS:1024-2-2_ED:2_EA:tanh_CA:asenc_DA:asenc_OA:asenc_LF:mean_squared_error_AEP:200_SEP:1_NO:L1_NI:0_BS:32_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AE:1_NR:0_UK:10_ITS:10_SL:0_SA:0',
                                              'DS:SingleDiseaseT2D_LS:1024-2-2_ED:2_EA:tanh_CA:asenc_DA:asenc_OA:asenc_LF:mean_squared_error_AEP:200_SEP:1_NO:L1_NI:0_BS:32_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AE:1_NR:0_UK:10_ITS:10_SL:0_SA:1',                                          
                                          ]

        # MSE=1.577*10^(-8) after 20 runs of 10-fold CV
        dataset_config['SingleDiseaseRA'] = ['DS:SingleDiseaseRA_LS:1024-2-2_ED:2_EA:softmax_CA:softmax_DA:softmax_OA:softmax_LF:kullback_leibler_divergence_AEP:200_SEP:1_NO:L1_NI:0_BS:32_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AE:1_NR:0_UK:10_ITS:1_SL:0_SA:0',
                                             'DS:SingleDiseaseRA_LS:1024-2-2_ED:2_EA:softmax_CA:softmax_DA:softmax_OA:softmax_LF:kullback_leibler_divergence_AEP:200_SEP:1_NO:L1_NI:0_BS:32_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AE:1_NR:0_UK:10_ITS:1_SL:0_SA:1',
                                         ]

        # MSE=2.176*10^(-8) after 20 runs of 10-fold CV
        dataset_config['SingleDiseaseIBD'] = ['DS:SingleDiseaseIBD_LS:1024-2-2_ED:2_EA:softmax_CA:softmax_DA:softmax_OA:softmax_LF:kullback_leibler_divergence_AEP:200_SEP:1_NO:L1_NI:0_BS:32_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AE:1_NR:0_UK:10_ITS:1_SL:0_SA:0',
                                              'DS:SingleDiseaseIBD_LS:1024-2-2_ED:2_EA:softmax_CA:softmax_DA:softmax_OA:softmax_LF:kullback_leibler_divergence_AEP:200_SEP:1_NO:L1_NI:0_BS:32_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AE:1_NR:0_UK:10_ITS:1_SL:0_SA:1',
                                          ]

        # MSE=2.617*10^(-8) after 20 runs of 10-fold CV
        dataset_config['HMP'] = ['DS:HMP_LS:1024-2-2_ED:2_EA:softmax_CA:softmax_DA:softmax_OA:softmax_LF:kullback_leibler_divergence_AEP:200_SEP:1_NO:L1_NI:0_BS:32_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AE:1_NR:0_UK:10_ITS:1_SL:0_SA:0',
                                 'DS:HMP_LS:1024-2-2_ED:2_EA:softmax_CA:softmax_DA:softmax_OA:softmax_LF:kullback_leibler_divergence_AEP:200_SEP:1_NO:L1_NI:0_BS:32_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AE:1_NR:0_UK:10_ITS:1_SL:0_SA:1',
                             ]

        loop_over_datasets()
    elif exp_mode == "SEARCH_SUPER_MODELS":
        plot_ae_fold = False
        plot_ae_overall = False

        plot_fold = False
        plot_iter = False
        plot_overall = True

        # For searching supervised models - normalization across samples needs to be on
        set_config(norm_input_key, [1])

        # edit exp_configs to your needs
        loop_over_datasets()
    elif exp_mode == "SEARCH_AUTO_MODELS":
        plot_ae_fold = False
        plot_ae_overall = True

        plot_fold = False
        plot_iter = False
        plot_overall = False

        # For searching unsupervised models - normalization across samples needs to be off
        set_config(norm_input_key, [0])
        # set the supervised epochs to be 1 to make the ROC calculations happy and to avoid wasting time on supervised
        # learning. We have to use the supervised part only because the K-fold cross-validation code was tangled with
        # supervised learning
        set_config(super_epochs_key, [1])

        # edit exp_configs to your needs
        loop_over_datasets()
    else:
        # anything goes, just edit exp_configs to your needs - it can be for a single specific model, or for
        # arbitrary grid search
        loop_over_datasets()

