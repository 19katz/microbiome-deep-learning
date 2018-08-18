import random as rn
from keras import backend as K
import itertools


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
act_reg_key = 'AR'
norm_input_key = 'NI'
early_stop_key = 'ES'
patience_key = 'PA'
dataset_key = 'DS'
norm_sample_key = 'NO'
backend_key = 'BE'
version_key = 'V'
use_ae_key = 'AU'
use_kfold_key = 'UK'
kfold_key = 'KF'
no_random_key = 'NR'
iter_key = 'IT'
shuffle_labels_key = 'SL'
num_iters_key = 'ITS'
shuffle_abunds_key = 'SA'
kmer_size_key = 'KS'
dropout_pct_key = 'DP'
input_dropout_pct_key = 'IDP'
max_norm_key = 'MN'
pca_dim_key = 'PC'
ae_datasets_key = 'AD'
after_ae_act_key = 'AA'
after_ae_layers_key = 'AL'
nmf_dim_key = 'NM'
class_weight_key = 'CW'



config_keys = [dataset_key, layers_key, enc_act_key,
               code_act_key, dec_act_key, out_act_key, enc_dim_key, pca_dim_key, auto_epochs_key, super_epochs_key,
               batch_size_key, loss_func_key,  batch_norm_key, dropout_pct_key, input_dropout_pct_key, act_reg_key,
               norm_input_key, early_stop_key, patience_key, norm_sample_key, backend_key,
               version_key, use_ae_key, no_random_key, shuffle_labels_key, shuffle_abunds_key,
               use_kfold_key, num_iters_key, kmer_size_key, max_norm_key, ae_datasets_key, after_ae_layers_key, after_ae_act_key, nmf_dim_key, class_weight_key, iter_key, kfold_key]

config_info_filename = [ dataset_key, kmer_size_key, layers_key, enc_act_key, code_act_key, dec_act_key, out_act_key,
                #loss_func_key,
                #auto_epochs_key,
                pca_dim_key, super_epochs_key, norm_sample_key, norm_input_key, batch_size_key,
                # Thes two are not used yet, so skip them to save file name length
                # early_stop_key, patience_key,
                dropout_pct_key, input_dropout_pct_key,
                #backend_key,
                version_key,
                use_ae_key, no_random_key,
                # kfold_key should be the last and iter_key second to last
                use_kfold_key, num_iters_key, shuffle_labels_key, shuffle_abunds_key, max_norm_key,  ae_datasets_key, after_ae_layers_key, after_ae_act_key, nmf_dim_key, class_weight_key, iter_key, kfold_key]
 
SAME_AS_ENC = "asenc"

epsilon = 0.5
exp_configs = {
                # Datasets to use
                dataset_key:       [ [
                                       #'SingleDiseaseMetaHIT',
                                       #'SingleDiseaseQin',
                                       #'SingleDiseaseRA',
                                       #'SingleDiseaseFeng',
                                       #'ZellerReduced',
                                       #'KarlssonReduced',
                                       'SingleDiseaseLiverCirrhosis',
                                     ], 'Dataset: {}'],
                
                norm_sample_key:   [ [
                                       'L1',
                                       # 'L2'
                                     ], 'Normalize each sample with: {}' ],
                # 1 for supervised and 0 for autoencoder only, 2 for mean centering only without std dev scaling -- CHANGE IT BACK TO 1 FOR ANY SUPERVISED LEARNING!!!
                norm_input_key:    [ [1], 'Normalize across samples (each component with zero mean/unit std across training samples): {}' ],

                kmer_size_key:     [ [
                                        #5,
                                        #6,
                                        7,
                                        8,
                                        #10
                                      ]

                                     , 'Kmer Size used: {}'],
                # Deep net structure
                # The last entry (-1 is the placeholder) of the layer list is for code layer dimensions - this is so we don't
                # have to list too many network layer lists when we vary only the code layer dimension.
                layers_key:        [ [
                                       [1, -1],
                                       #[1, 1/2, -1],
                                       #[1, "random"],
                                       #[1, 2, -1],
                                       #[1, 4, -1],
                                       #[1, 8, -1]
                                       #[1, 2, -1],
                                       #[1, 1/2, "random"],
                                       #[1, 1/2, 1/4, "random"],
                                       #[1, 1/2, 1/4, 1/8, "random"],
                                       #[1, 1/2, 1/4, 1/8, 1/16, "random"],
                                     ], "Layers for autoencoder's first half : {}" ],
                enc_dim_key:       [ [256, 512, 1024],  'Encoding dimensions: {}' ],
                enc_act_key:       [ [
                                         'sigmoid',
                                         #'relu',
                                         #'linear',
                                         #'softmax',
                                         #'tanh',
                                     ], 'Encoding activation: {}' ],
                code_act_key:      [ [
                                         #SAME_AS_ENC,
                                         'linear',
                                         'softmax',
                                         'sigmoid',
                                         'relu',
                                         'tanh',

                                     ], 'Code (last encoding) layer activation: {}' ],
                # Decoding activations are fixed as linear as they are popped off anyway
                # after autoencoder training
                dec_act_key:       [ [
                                         #SAME_AS_ENC,
                                         #'linear',
                                         #'sigmoid',
                                         #'relu',
                                         #'softmax',
                                         'tanh',
                                     ], 'Decoding layer activation: {}' ],
                out_act_key:       [ [

                                         #SAME_AS_ENC,
                                         #'linear',
                                         #'sigmoid',
                                         #'relu',
                                         'softmax',
                                         #'tanh',
                                     ], 'Last decoding layer activation: {}' ],
                
                loss_func_key :    [ [
                                         #'mean_squared_error',
                                         'kullback_leibler_divergence'
                                     ], 'Autoencoder loss function: {}' ],
                after_ae_layers_key: [ [
                                          [],
                                          ],
                                        'Layers added after the autoencoder: {}', []],
                                          
                after_ae_act_key:         [ [
                                        'linear',
                                         #'softmax',
                                         #'sigmoid',
                                         #'relu',
                                         #'tanh',
                                        ], 'Autoencoder activation: {}', None ],
                                        
                pca_dim_key:       [ [0],
                                      
                                     'Number of principal components for PCA, if 0 no PCA should be used: {}', 0],

                nmf_dim_key:       [ [0],
                                      
                                     'Number of principal components for NMF, if 0 no NMF should be used: {}', 0],
                # boolean for whether to use autoencoder for pretraining before supervised learning
                use_ae_key:    [ [0], 'Use autoencoder pretraining for supervised learning: {}', 0 ],

                ae_datasets_key: [ [
                                        [
                                         #'H',
                                         #'Q',
                                         #'R',
                                         #'F',
                                         #'L',
                                         #'Z',
                                         #'M',
                                         #'K'
                                         ],
                                        ],
                                      'Other datasets to train the autoencoder on: {}',
                                       []
                                      ],

                class_weight_key: [ [0],
                                    'Ratio of weights of diseased over healthy: {}',
                                        0 ],
                
                # Training options
                auto_epochs_key :  [ [0
                    # 50
                    ], 'Max number of epochs for autoencoder training: {}' ],
                super_epochs_key : [ [200,
                                      #400
                                      ], 'Max number of epochs for supervised training: {}' ],
                batch_size_key:    [ [
                                      8,
                                      #16,
                                      #32
                                      ], 'Batch size used during training: {}' ],
                # two booleans
                batch_norm_key:    [ [0], 'Use batch normalization: {}' ],
                dropout_pct_key:   [ [0, 0.25, 0.5, 
                    #0.25, 0.35, 0.5, 0.75
                                      ], 'Dropout percent: {}', 0],
                input_dropout_pct_key: [ [0,0.25, 0.5, 
                    #0.25, 0.35, 0.5, 0.75
                                          ], 'Dropout percent on input layer: {}', 0],
                act_reg_key:       [ [0], 'Activation regularization (for sparsity): {}' ],
                # boolean
                early_stop_key:    [ [0],  'Use early stopping: {}' ],
                patience_key:      [ [2], 'Early stopping patience (consecutive degradations): {}' ],
                use_kfold_key:    [ [10], 'Stratified K folds (0 means one random shuffle with stratified 80/20 split): {}' ],
                kfold_key:    [ [0], 'K fold index for the current fold: {}' ],
                # boolean for whether no randomness should be used
                no_random_key:    [ [0], "Eliminate randomness in training: {}" ],
                # number of iterations
                num_iters_key:    [ [1], "Number of iterations: {}" ],
                # the current iteration index
                iter_key:    [ [0], "Iteration: {}" ],
                # boolean
                shuffle_labels_key:    [ [0],  'Shuffle labels (for supervised null): {}' ],
                # boolean
                shuffle_abunds_key:    [ [0],  'Shuffle abundances (for unsupervised null): {}' ],

                # misc
                backend_key:   [ [K.backend()], 'Backend: {}' ],
                version_key:   [ ['AD'], 'Version (catching all other unnamed configs): {}' ],
                max_norm_key:  [ [0, 1, 2, 3, 4], 'Max norm for kernel constraint: {}', 0]
            }

class ConfigIterator:
    
    def __init__(self, random=False, count=None):
        self.random = random
        self.max = count
        self.used = 0
        self.cache = {}

    def __iter__(self):
        if not self.random:
            iterator = itertools.product(*[exp_configs[key][0] for key in config_keys])
            for it in iterator:
                if self.max is not None and self.used >= self.max:
                    raise StopIteration
                next_config_dict = {}
                for i in range(len(it)):
                    next_config_dict[config_keys[i]] = it[i]
                change_layers(next_config_dict)
                config_inf = config_info(next_config_dict)
                if config_inf in self.cache:
                    continue
                else:
                    self.cache[config_inf] = 1
                self.used += 1
                yield next_config_dict
        else:
            while self.max is None or self.used < self.max:
                next_config = [exp_configs[key][0][rn.randint(0, len(exp_configs[key][0]) - 1)] for key in config_keys]
                next_config_dict = {}
                for i in range(len(next_config)):
                    next_config_dict[config_keys[i]] = next_config[i]
                change_layers(next_config_dict)
                config_inf = config_info(next_config_dict)
                if config_inf in self.cache:
                    continue
                else:
                    self.cache[config_inf] = 1
                self.used += 1
                yield next_config_dict

def name_file_from_config(config, skip_keys=[]):
    filename = ''
    # Canonicalize old config info first so we don't generate different file names for different folds
    config_info(config)
    for k in config_info_filename:
        # skip the specified keys, used for skipping the fold and iteration indices (for aggregating results across them)
        if not k in skip_keys:
           filename += '_' + k + ':' + str(get_config_val(k, config))
    return filename

def config_info(config, skip_keys=[]):
    config_info = ''
    if config[code_act_key] == SAME_AS_ENC:
        config[code_act_key] = config[enc_act_key]
    if config[out_act_key] == SAME_AS_ENC:
        config[out_act_key] = config[enc_act_key]
    if config[dec_act_key] == SAME_AS_ENC:
        config[dec_act_key] = config[enc_act_key]
    if len(config[layers_key]) <= 2:
        config[enc_act_key] = config[code_act_key]
        config[dec_act_key] = config[out_act_key]
    if not config[use_ae_key]:
        config[dec_act_key] = 'NA'
        config[out_act_key] = 'NA'
    for k in config_keys:              
        # skip the specified keys, used for skipping the fold and iteration indices (for aggregating results across them)
        if not k in skip_keys:
            config_info += '_' + k + ':' +str(get_config_val(k, config))
    return config_info

# Get the config value for the given config key - used in identifying model in grid search
# as well as unquiely naming the plots for the models
def get_config_val(config_key, config):
    val = config[config_key]
    if type(val) is list:
        val = '-'.join([ str(c) for c in val])
    return val

# Get the config description for the given config key - used in figure descriptions
def get_config_desc(config_key, config):
    return exp_configs[config_key][1].format(get_config_val(config_key, config))

def change_layers(next_config_dict):
    input_dimensions = 4 ** next_config_dict[kmer_size_key]
    if next_config_dict[kmer_size_key] % 2 == 0:
        half_kmer_size = next_config_dict[kmer_size_key] // 2
        input_dimensions = (input_dimensions - 4 ** half_kmer_size) // 2 + 4 ** half_kmer_size
    else:
        input_dimensions = input_dimensions // 2
    if next_config_dict[pca_dim_key] > 0:
        if next_config_dict[pca_dim_key] < 1:
            input_dimensions = int(next_config_dict[pca_dim_key] * input_dimensions + epsilon)
        else:
            input_dimensions = next_config_dict[pca_dim_key]
        next_config_dict[pca_dim_key] = input_dimensions
        next_config_dict[use_ae_key] = 0
        next_config_dict[nmf_dim_key] = 0
    if next_config_dict[nmf_dim_key] > 0:
        if next_config_dict[nmf_dim_key] < 1:
            input_dimensions = int(next_config_dict[nmf_dim_key] * input_dimensions + epsilon)
        else:
            input_dimensions = next_config_dict[nmf_dim_key]
        next_config_dict[nmf_dim_key] = input_dimensions
        next_config_dict[use_ae_key] = 0

    layers = list(next_config_dict[layers_key])
    for i in range(len(layers) - 1):
        if (layers[i] <= 1 and layers[i] > -1):
            layers[i] = int(input_dimensions * layers[i] + epsilon)
    if layers[-1] == -1:
        layers[-1] = next_config_dict[enc_dim_key]
    else:
        if layers[-1] == "random":
            layers[-1] = rn.randint(2, layers[-2] - 1)
        elif layers[-1] <= 1:
            layers[-1] = int(input_dimensions * layers[-1] + epsilon)
        next_config_dict[enc_dim_key] = layers[-1]
    
    next_config_dict[layers_key] = layers

    after_ae_layers = list(next_config_dict[after_ae_layers_key])
    enc_dims = layers[-1]
    for i in range(len(after_ae_layers)):
        if (after_ae_layers[i] <= 1 and after_ae_layers[i] > -1):
            after_ae_layers[i] = int(enc_dims * after_ae_layers[i] + epsilon)

    next_config_dict[after_ae_layers_key] = after_ae_layers

    if next_config_dict[dropout_pct_key] == 0:
        next_config_dict[input_dropout_pct_key] = 0
            
if __name__ == "__main__":
    for config in ConfigIterator(random=True, count=20):
        print(name_file_from_config(config))
    
        
            
            
