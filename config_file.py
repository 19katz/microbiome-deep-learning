# config file for running deep learning models
import os

config={}

config['data_set'] = 'MetaHIT'
                      #'Feng'  
                      #'HMP'  
                      #'Karlsson_2013' 
                      #'LiverCirrhosis'  
                      #'MetaHIT'  
                      #'Qin_et_al'  
                      #'RA'
                      #'Zeller_2014'


config['kmer_size'] = 5

config['norm_input'] = True #  standardize the data


### params to iterate over ###
config['encoding_dim'] = 10

config['encoded_activation'] = 'relu'

config['input_dropout_pct'] = 0.1

config['dropout_pct'] = 0.1
###                        ###

config['num_epochs'] = 400
                    
config['batch_size'] = 16

config['n_splits'] = 5

config['n_repeats'] = 10

config['compute_informative_features'] = False

config['plot_iteration'] = False

config['graph_dir'] = os.path.expanduser('~/deep_learning_microbiome/analysis')

