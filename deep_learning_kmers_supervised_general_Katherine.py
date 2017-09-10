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
    load_metahit_with_obesity_labels, load_metahit_with_bmi_labels, load_all_kmer_cnts_with_labels
from kmer_norms import norm_by_l1, norm_by_l2
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import StratifiedKFold
from itertools import cycle, product
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

input_dim = 1024

graph_dir = '~/deep_learning_microbiome/analysis/kmers'

# Turn off plotting when doing the grid search - otherwise, you'll get tons of plots.
plot = True

# Number of iterations - increase this to get more reliable experiment results.
n_iters = 1

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
    
    'SingleDiseaseIBD': ('IBD', load_kmer_cnts_for_metahit_with_labels, 6, [ 'Healthy', 'IBD', ],
                          ['green', 'red', ], datasource_marker),
    'SingleDiseaseIBDLabelShuffled': ('IBD-LabelShuffled', lambda: load_kmer_cnts_for_metahit_with_labels(shuffle_labels=True), 6, [ 'Healthy', 'IBD', ],
                                      ['green', 'red', ], datasource_marker),
    'SingleDiseaseT2D': ('T2D', load_kmer_cnts_for_t2d_with_labels, 6, [ 'Healthy', 'T2D', ],
                         ['green', 'purple', ], datasource_marker),
    'SingleDiseaseT2DLabelShuffled': ('T2D-LabelShuffled', lambda: load_kmer_cnts_for_t2d_with_labels(shuffle_labels=True), 6, [ 'Healthy', 'T2D', ],
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

# This is the experiment setup - for grid search of model candidates. It can also be used to evalue individual models.
# Most of the time, this is the only place that needs to be modified.
#
# <exp_config_key>: [ <list of values to be experimented with for this key>, <format string for the exp config value> ]
# Boolean values use 1 for True and 0 for False to avoid exceeding plot file name length limit on Linux.
exp_configs = {
                # Datasets to use
                dataset_key:       [ [
                                       'AllContinent',
                                       'AllCountry',
                                       'AllHealth',
                                       'SingleDiseaseIBD',
                                       'SingleDiseaseT2D',
                                       'SingleDiseaseRA',
                                     ], 'Dataset: {}'],
                norm_sample_key:   [ [
                                       'L1',
                                       # 'L2'
                                     ], 'Normalize each sample with: {}' ],
                norm_input_key:    [ [1], 'Normalize across samples (each component with zero mean/unit std across training samples): {}' ],

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
                                         #'sigmoid',
                                         # 'relu',
                                         'linear',
                                         # 'softmax',
                                         # 'tanh',
                                     ], 'Encoding activation: {}' ],
                code_act_key:      [ [
                                         # 'linear',
                                         # 'softmax',
                                         # 'sigmoid',
                                         # 'relu',
                                         'tanh',

                                     ], 'Code (last encoding) layer activation: {}' ],
    
                # Decoding activations are fixed as linear as they are popped off anyway
                # after autoencoder training
                dec_act_key:       [ [
                                         'linear',
                                         #'sigmoid',
                                         #'relu',
                                         #'softmax',
                                         #'tanh',
                                     ], 'Decoding layer activation: {}' ],
                out_act_key:       [ [
                                         'linear',
                                         #'sigmoid',
                                         #'relu',
                                         #'softmax',
                                         #'tanh',
                                     ], 'Last decoding layer activation: {}' ],
                loss_func_key :    [ [
                                         #'mean_squared_error',
                                         'kullback_leibler_divergence'
                                     ], 'Autoencoder loss function: {}' ],
                # boolean for whether to use autoencoder for pretraining before supervised learning
                use_ae_key:    [ [1], 'Use autoencoder pretraining for supervised learning: {}' ],


                # Training options
                auto_epochs_key :  [ [50], 'Max number of epochs for autoencoder training: {}' ],
                super_epochs_key : [ [200], 'Max number of epochs for supervised training: {}' ],
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
                # the current iteration index
                iter_key:    [ [0], "Iteration: {}" ],

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

# numer of classes
n_classes = None

# class name to index map - used for mapping class name to target vector and finding index of the color for plotting the class.
class_to_ind = None

# colors for plotting
colors = None

# data source to marker dict for the currently active exp dataset
markers = None

# TODO: for aggregating across K folds and iterations
val_losses, val_accs, macro_aucs = None, None, None

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
        global class_name, label_ind, classes, n_classes, class_to_ind, colors, markers

        class_name = cls_name
        label_ind = lbl_ind
        classes = lbls
        n_classes = len(classes)
        class_to_ind = { classes[i]: i for i in range(n_classes) }
        colors = clrs
        markers = mkrs

        # load the samples and their labels
        data, orig_target = data_loader()

        # Temp labels just for stratified shuffle split - we split proportionally by datasource+label+health
        # the 0 index is for diseased or not 
        target_labels = [(orig_target[i][0] + orig_target[i][label_ind] + orig_target[i][source_ind]) for i in range(len(orig_target))]
        orig_target = np.array(orig_target)

        # The target matrix for training/testing - note that only the label index is used this time
        target = np.array([ class_to_target(orig_target[i][label_ind]) for i in range(len(orig_target)) ])

        for i in range(n_iters):
            exp_config[iter_key] = i
            
            # This is to optionally make the shuffle/fold deterministic
            setup_randomness()

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

            global val_losses, val_accs, macro_aucs
            val_losses, val_accs, macro_aucs = [], [], []

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
                        # as in Pasoli, we removed randomness and used 5 folds for grid search,
                        # but allow randomness and use 10 folds for evaluation.
                        prev_no_random = get_config_val(no_random_key)
                        prev_kfolds = get_config_val(use_kfold_key)

                        # Parse the config info and set exp_configs accordingly
                        setup_exp_configs_from_config_info(config)

                        # Because the above will use the config info to set up randomness and # of folds,
                        # we set them back to their values before the call (what're in exp_configs)
                        set_config(no_random_key, [prev_no_random])
                        set_config(use_kfold_key, [prev_kfolds])

                        loop_over_norm_funcs()
                except KeyError:
                    # No speficic configs for the dataset, so we loop thru exp_configs - often used for grid search
                    loop_over_norm_funcs()

                # the next fold
                exp_config[kfold_key] += 1

            # print(('{}\t{}\t{}\t{}\t{}\t{}\tautoencoder-super-kfold-val-loss-acc-auc for ' + class_name + ':{}').
            #       format(np.mean(val_losses), np.std(val_losses), np.mean(val_accs), np.std(val_accs), np.mean(macro_aucs), np.std(macro_aucs), config_info()))


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
    autoencoder.add(Activation(exp_config[code_act_key]))

    # the model from input to the code layer - used for plotting 2D graphs
    code_layer_model = Model(inputs=autoencoder.layers[0].input,
                             outputs=autoencoder.layers[cnt_to_coding_layer() - 1].output)

    print("Code layer model:")
    code_layer_model.summary()

    # stats for the autoencoder - the 9s are for when autoencoder is (optionally) not used before supervised learning
    ae_val_loss, ae_loss, ae_mse = 999999, 999999, 999999

    if get_config_val(use_ae_key):
        # Use the autoencoder before supervised learning
        #
        # Add the decoding layers - from second to last layer to the second layer
        # because the last one is the "code" layer, which has no decoding counterpart.
        for i in range(len(layers)-2, 0, -1):
            autoencoder.add(Dense(layers[i], input_dim=layers[i+1]))
            add_layer_common(autoencoder, exp_config[dec_act_key])

        # the last decoding layer - it has an activation function independent of other decoding layers
        i = 0
        autoencoder.add(Dense(layers[i], input_dim=layers[i+1]))
        autoencoder.add(Activation(exp_config[out_act_key]))

        # Compile the auencoder model
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

        # Plot training/test loss by epochs for the autoencoder
        if plot:
            fig = pylab.figure()
            pylab.plot(history.history['loss'])
            pylab.plot(history.history['val_loss'])
            #pylab.yticks(list(pylab.yticks()[0]) + [history.history['loss'][-1], history.history['val_loss'][-1]])
            pylab.legend(['train', 'test'], loc='upper right')
            #pylab.legend(['train ' + str(history.history['loss'][-1]), 'test ' + str(history.history['val_loss'][-1])], loc='upper right')
            pylab.title('Loss vs Epoch')
            pylab.xlabel('Number of Epochs')
            pylab.ylabel('Loss')
            pylab.gca().set_position((.1, .7, .8, .6))
            add_figtexts_and_save(fig, 'ae_general_loss_vs_epochs', "Autoencoder's loss vs epochs before being applied to supervised learning")

        # Calculate the MSE on the test samples
        decoded_output = autoencoder.predict(x_test)
        ae_mse = mean_squared_error(decoded_output, x_test)
        mean_squared_errors.append(ae_mse)

        hist = history.history
        ae_val_loss = hist['val_loss'][-1]
        ae_loss = hist['loss'][-1]
        last_loss.append(ae_val_loss)

        # Which epoch achieved min loss - useful for manual early stopping
        min_vli = np.argmin(hist['val_loss'])
        min_li = np.argmin(hist['loss'])

        # Log the autoencoder stats/performance
        print('{},{},{}\t{},{},{}\t{}\t{}\t{}\tautoencoder-min-loss-general:{}'.
              format(hist['val_loss'][min_vli], min_vli, hist['loss'][min_vli],
                     hist['loss'][min_li], min_li, hist['val_loss'][min_li],
                     ae_val_loss, ae_loss, ae_mse, config_info()))

        # if plot and exp_config[enc_dim_key] > 1:
        #     fig = pylab.figure()
        #     pylab.title('AE2 VS AE1 After Autoencoder Training')
        #     pylab.xlabel('AE1')
        #     pylab.ylabel('AE2')

        #     codes_pred = code_layer_model.predict(x_test)
        #     print("Autoencoder's test code shape for dimension {}: {}".format(exp_config[enc_dim_key], codes_pred.shape))
        #     plot_2d_codes(codes_pred, fig, 'ae_general_two_codes',
        #                   "This graph shows autoencoder's first 2 components of the encoding codes (the middle hidden layer output) with labels")

        # Pop the decoding layers before supervised learning
        autoencoder.summary()
        for n in range(cnt_to_coding_layer()):
            autoencoder.pop()

    # Add the classification layer for supervised elarning - note that the loss function is always
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
    if plot and exp_config[enc_dim_key] > 1:
        fig = pylab.figure()
        pylab.title('AE2 VS AE1 After Supervised Training On ' + class_name)
        pylab.xlabel('AE1')
        pylab.ylabel('AE2')

        codes_pred = code_layer_model.predict(x_test)
        print("Autoencoder's test code shape for dimension {}: {}".format(exp_config[enc_dim_key], codes_pred.shape))
        plot_2d_codes(codes_pred, fig, 'ae_super_general_two_codes',
                      "This graph shows the first 2 components of the encoding codes after supervised training")

    y_train_pred = autoencoder.predict(x_train)

    # predictions on the test samples
    y_test_pred = autoencoder.predict(x_test)

    # plot the confusion matrix
    if plot:
        # compare the true label index (with max value (1.0) in the target vector) against the predicted
        # label index (index of label with highest predicted probability)
        conf_mat = confusion_matrix(np.argmax(y_test, axis=-1), np.argmax(y_test_pred, axis=-1))
        plot_confusion_matrix(conf_mat)

    # printing the accuracy rates for diagnostics
    print("Total accuracy for " + str(len(y_test_pred)) + " test samples: " +
          str(np.mean(np.equal(np.argmax(y_test, axis=-1), np.argmax(y_test_pred, axis=-1)))))

    for cls in classes:
        idx = []
        for i in range(y_test_pred.shape[0]):
            if info_test[i][4] == cls:
                idx.append(i)
        if len(idx) == 0:
            continue
        idx = np.array(idx)
        print("Accuracy for total of " + str(len(idx)) + " " + cls + " samples: " +
              str(np.mean(np.equal(np.argmax(y_test[idx], axis=-1), np.argmax(y_test_pred[idx], axis=-1)))))

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
        acc[i] = accuracy_score(np.round(y_test[:, i]), np.equal(np.argmax(y_test_pred, axis=-1), i))
        

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

    if plot:
        fig = pylab.figure()
        lw = 2
        if n_classes > 2:
            pylab.plot(fpr["micro"], tpr["micro"],
                       label='micro-average ROC (AUC = {0:0.4f})'
                       ''.format(roc_auc["micro"]),
                       color='deeppink', linestyle=':', linewidth=4)

            pylab.plot(fpr["macro"], tpr["macro"],
                       label='macro-average ROC (AUC = {0:0.4f})'
                       ''.format(roc_auc["macro"]),
                       color='navy', linestyle=':', linewidth=4)

            colors = cycle(['green', 'red', 'purple', 'darkorange'])
            for i, color in zip(range(n_classes), colors):
                pylab.plot(fpr[i], tpr[i], color=color, lw=lw,
                           label='ROC of {0} (AUC = {1:0.4f}, acc={2:0.4f})'
                           ''.format(classes[i], roc_auc[i], acc[i]))
        else:
            # plot just one curve for binary case 
            pylab.plot(fpr[1], tpr[1], color="darkorange", lw=lw,
                       label='ROC of {0} (AUC = {1:0.4f}, acc={2:0.4f})'
                       ''.format(classes[1], roc_auc[1], acc[1]))

        # plot the ROC curves with AUCs
        pylab.plot([0, 1], [0, 1], 'k--', lw=lw)
        pylab.xlim([0.0, 1.0])
        pylab.ylim([0.0, 1.05])
        pylab.xlabel('False Positive Rate')
        pylab.ylabel('True Positive Rate')
        pylab.title('Sample ' + class_name + ' Prediction Using 5-kmers')
        pylab.legend(loc="lower right")
        pylab.gca().set_position((.1, .7, .8, .6))
        add_figtexts_and_save(fig, 'ae_super_general_roc_auc', "ROC/AUC plots for predicting sample's " + class_name + " using 5mers")

        # plot loss vs epochs
        fig = pylab.figure()
        pylab.plot(history.history['loss'])
        pylab.plot(history.history['val_loss'])
        pylab.legend(['train', 'test'], loc='upper right')
        pylab.legend(['train ' + str(history.history['loss'][-1]), 'test ' + str(history.history['val_loss'][-1])])
        pylab.title('Loss vs Epoch')
        pylab.xlabel('Number of Epochs')
        pylab.ylabel('Loss')
        pylab.gca().set_position((.1, .7, .8, .6))
        add_figtexts_and_save(fig, 'ae_super_general_loss_vs_epochs', "Supervised " + class_name + " learning loss vs epochs after autoencoder learning the encoding codes")

        # plot accuracy vs epochs
        fig = pylab.figure()
        pylab.plot(history.history['acc'])
        pylab.plot(history.history['val_acc'])
        pylab.legend(['train ' + str(history.history['acc'][-1]), 'test ' + str(history.history['val_acc'][-1])], loc='lower right')
        pylab.title('Accuracy vs Epoch')
        pylab.xlabel('Number of Epochs')
        pylab.ylabel('Accuracy')
        pylab.gca().set_position((.1, .7, .8, .6))
        add_figtexts_and_save(fig, 'ae_super_general_accu_vs_epochs', "Supervised " + class_name + " learning accuracy vs epochs after autoencoder learning the encoding codes")

        # plot raw probabilities per test sample (not tested - commented out)
        fig = pylab.figure()
        pylab.title('Predicted Test Sample Probabilities With Labels')
        pylab.xlabel('Test Sample')
        pylab.ylabel('Predicted Probability')

        # plot_probs(y_test_pred, fig, 'ae_super_general_pred_probs',
        #            "This graph shows predicted probabilities with labels for test samples after supervised training")

    # log the performance stats for sorting thru the grid search results
    hist = history.history
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
    for i in [ 'micro', 'macro' ]:
        avg_perf += roc_auc[i]
    avg_perf /= 12

    print(('{}\t{},{},{},{},{}\t{},{},{},{},{}\t{},{},{},{},{}\t{},{},{},{},{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\tautoencoder-super-min-loss-max-acc for ' + class_name + ':{}').
          format(avg_perf, hist['val_acc'][max_vai], max_vai, hist['acc'][max_vai], hist['val_loss'][max_vai], hist['loss'][max_vai],
                 hist['acc'][max_ai], max_ai, hist['val_acc'][max_ai], hist['val_loss'][max_ai], hist['loss'][max_ai],
                 hist['val_loss'][min_vli], min_vli, hist['val_acc'][min_vli], hist['acc'][min_vli], hist['loss'][min_vli],
                 hist['loss'][min_li], min_li, hist['val_acc'][min_li], hist['acc'][min_li], hist['val_loss'][min_li],
                 ae_super_val_acc, ae_super_acc, ae_super_val_loss, ae_super_loss,
                 ae_val_loss, ae_loss, ae_mse, roc_auc['micro'], roc_auc['macro'], config_info()))

    # TODO: across K folds and iterations stats and plots
    #
    # val_losses.append(ae_super_val_loss)
    # val_accs.append(ae_super_val_acc)
    # macro_aucs.append(roc_auc['macro'])
    # if exp_config[kfold_key] == get_config_val(use_kfold_key):
        
# Plot the 2D codes in the layer right before the final classificaion layer
def plot_2d_codes(codes, fig, name, desc):
    handles = {}
    for i in range(len(codes)):
        src = info_test[i][source_ind]
        ind = class_to_ind[info_test[i][label_ind]]
        h = pylab.scatter([codes[i][0]], [codes[i][1]], marker=markers[src], color=colors[ind], facecolor='none')
        handles[src + ' - ' + info_test[i][label_ind]] = h
    keys = [ k for k in handles.keys() ]
    keys.sort()
    pylab.legend([handles[k] for k in keys], keys)
    pylab.gca().set_position((.1, .7, 2.4, 1.8))
    add_figtexts_and_save(fig, name, desc)
    
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
    pylab.legend([handles[k] for k in keys], keys)
    pylab.gca().set_position((.1, .7, 2.4, 1.8))
    add_figtexts_and_save(fig, name, desc)

def plot_confusion_matrix(cm, cmap=pylab.cm.Reds):
    """
    This function plots the confusion matrix.
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    fig, ax = pylab.subplots(1, 2)
    for sub_plt, conf_mat, title, fmt in zip(ax, [cm, cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]], ['Unnormalized Confusion Matrix', 'Normalized Confusion Matrix'], ['d', '.2f']):
        #print(conf_mat)
        #print(title)

        im = sub_plt.imshow(conf_mat, interpolation='nearest', cmap=cmap)
        sub_plt.set_title(title)
        divider = make_axes_locatable(sub_plt)
        cax1 = divider.append_axes("right", size="5%", pad=0.1)
        #fig.colorbar(im, ax=sub_plt)
        fig.colorbar(im, cax=cax1)
        tick_marks = np.arange(len(cm))
        sub_plt.set_xticks(tick_marks)
        sub_plt.set_yticks(tick_marks)
        sub_plt.set_xticklabels(classes)
        sub_plt.set_yticklabels(classes)

        thresh = 0.8*conf_mat.max()
        for i, j in product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
            sub_plt.text(j, i, format(conf_mat[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if conf_mat[i, j] > thresh else "black")
        sub_plt.set_ylabel('True Label')
        sub_plt.set_xlabel('Predicted Label')
    pylab.tight_layout()
    #pylab.gca().set_position((.1, 10, 0.8, .6))
    add_figtexts_and_save(fig, 'ae_super_general_confusion_mat', "Confusion matrix for predicting sample's " + class_name + " status using 5mers", y_off=1.3)

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
# as well as unquiely naming plots for the models
def config_info():        
    config_keys = [ dataset_key, layers_key, enc_dim_key, enc_act_key, code_act_key, dec_act_key, out_act_key, loss_func_key,
                    auto_epochs_key, super_epochs_key, norm_sample_key, norm_input_key, batch_size_key, batch_norm_key,
                    # Thes two are not used yet, so skip them to save file name length
                    # early_stop_key, patience_key,
                    dropout_key, act_reg_key, backend_key, version_key, use_ae_key, use_kfold_key, kfold_key, no_random_key, iter_key ]
    config_info = ''
    for k in config_keys:
        config_info += '_' + k + ':' + str(get_config_val(k))
    return config_info

# parse config info string back into exp configs - for evaluating models after grid search
def setup_exp_configs_from_config_info(config):
    cnt1 = 0
    for part in config.split('LF:kullback_leibler_divergence'):
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
        exp_configs['LF'][0] = ['kullback_leibler_divergence']        

# Add figure texts to plots that describe configs of the experiment that produced the plot
def add_figtexts_and_save(fig, name, desc, x_off=0.02, y_off=0.56, step=0.04):
    pylab.figtext(x_off, y_off, desc)

    y_off -= step
    pylab.figtext(x_off, y_off,  get_config_desc(dataset_key) + ' --- ' + get_config_desc(norm_sample_key)  + ' --- ' + get_config_desc(no_random_key))

    y_off -= step
    pylab.figtext(x_off, y_off, get_config_desc(norm_input_key))

    y_off -= step
    pylab.figtext(x_off, y_off, get_config_desc(layers_key) + ' --- ' + get_config_desc(enc_dim_key) + ' --- ' + get_config_desc(use_ae_key))

    y_off -= step
    pylab.figtext(x_off, y_off, get_config_desc(enc_act_key) + ' --- ' + get_config_desc(code_act_key))

    y_off -= step
    pylab.figtext(x_off, y_off, get_config_desc(dec_act_key) + ' --- ' + get_config_desc(out_act_key)
                  + ' --- ' + get_config_desc(use_kfold_key) + ' --- ' + get_config_desc(kfold_key))

    y_off -= step
    pylab.figtext(x_off, y_off, get_config_desc(loss_func_key) + ' --- ' + get_config_desc(batch_size_key))

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
                  + ' --- ' + get_config_desc(version_key) + ' --- ' + get_config_desc(iter_key))


    pylab.savefig(os.path.expanduser(graph_dir + '/' + name + config_info() + '.svg')
                  , bbox_inches='tight')
    pylab.close(fig)

if __name__ == '__main__':
 
    set_config(dataset_key, ['AllContinent', 'AllCountry', 'AllHealth', 'SingleDiseaseIBD', 'SingleDiseaseT2D', 'SingleDiseaseRA'])

    # This produced 0.998 AUC and 0.988 ACC in average with 10 iterations and full randomness (shuffling and weight initialization)
    dataset_config['AllContinent'] = ['LS:1024-2_ED:2_EA:linear_CA:tanh_DA:linear_OA:linear_LF:kullback_leibler_divergence_AEP:50_SEP:200_NO:L1_NI:1_BS:16_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AE:1_UK:5__NR:1',]

    # This produced 0.990 AUC and 0.946 ACC in average with 10 iterations and full randomness (shuffling and weight initialization)
    dataset_config['AllCountry'] = ['LS:1024-2-2_ED:2_EA:tanh_CA:linear_DA:linear_OA:linear_LF:kullback_leibler_divergence_AEP:50_SEP:200_NO:L1_NI:1_BS:16_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AE:1_UK:5__NR:1',]

    # This produced 0.903 AUC and 0.726 ACC in average with 10 iterations and full randomness (shuffling and weight initialization)
    dataset_config['AllHealth'] = ['LS:1024-2_ED:2_EA:linear_CA:linear_DA:linear_OA:linear_LF:kullback_leibler_divergence_AEP:50_SEP:600_NO:L1_NI:1_BS:32_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AE:1_UK:5__NR:1',]

    # This produced 0.966 AUC and 0.920 ACC in average with 10 iterations and full randomness (shuffling and weight initialization)
    dataset_config['SingleDiseaseIBD'] = ['LS:1024-2_ED:2_EA:linear_CA:tanh_DA:linear_OA:linear_LF:kullback_leibler_divergence_AEP:50_SEP:600_NO:L1_NI:1_BS:16_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AE:1_UK:5__NR:1',]

    # This produced 0.805 AUC and 0.721 ACC in average with 10 iterations and full randomness (shuffling and weight initialization)
    dataset_config['SingleDiseaseT2D'] = ['LS:1024-2-2_ED:2_EA:linear_CA:linear_DA:linear_OA:linear_LF:kullback_leibler_divergence_AEP:50_SEP:600_NO:L1_NI:1_BS:16_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AE:1_UK:5__NR:1',]

    # This produced 0.689 AUC and 0.609 ACC in average with 10 iterations and full randomness (shuffling and weight initialization)
    # but the std is too big (around 0.1 for both), suggesting again that the RA data is not amenable to this technique.
    dataset_config['SingleDiseaseRA'] = ['LS:1024-2_ED:2_EA:linear_CA:linear_DA:linear_OA:linear_LF:kullback_leibler_divergence_AEP:50_SEP:600_NO:L1_NI:1_BS:32_BN:0_DO:0_AR:0_BE:tensorflow_V:2_AE:1_UK:5__NR:1',]

    # To do your own grid search or test any specific model config, just edit exp_configs to your needs and comment out the above
    loop_over_datasets()
