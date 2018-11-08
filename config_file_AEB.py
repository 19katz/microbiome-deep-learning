# config file for running deep learning models
import os

data_directory = os.path.expanduser("~/deep_learning_microbiome/data/")
analysis_directory = os.path.expanduser("~/deep_learning_microbiome/analysis/")
scripts_directory = os.path.expanduser("~/deep_learning_microbiome/scripts/")
species_directory = os.path.expanduser("~/deep_learning_microbiome/data/pasolli_speciesAb/")

config={}

config['data_set'] = ['MetaHIT']
                      #'Feng'  
                      #'HMP'  
                      #'Karlsson_2013' 
                      #'LiverCirrhosis'  
                      #'MetaHIT'  
                      #'Qin_et_al'  
                      #'RA'
                      #'Zeller_2014'


#config['kmer_size'] = [7]

config['norm_input'] = [True] #  standardize the data


### params to iterate over ###
config['encoding_dim'] = [8]

config['encoded_activation'] = ['sigmoid']

config['input_dropout_pct'] = [0]

config['dropout_pct'] = [0]


config['num_epochs'] = [400]
                    
config['batch_size'] = [16]
###                        ###

config['n_splits'] = [10]

config['n_repeats'] = [5]

config['compute_informative_features'] = [False]

config['plot_iteration'] = [False]

config['graph_dir'] = [os.path.expanduser('~/deep_learning_microbiome/analysis')]


