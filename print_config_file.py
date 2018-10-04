# config file for running deep learning models
import os

config={}

config['data_set'] = [#'Feng',  
                      #'Karlsson_2013'
                      #'Karlsson_2013_no_adapter',
                      #'LiverCirrhosis' 
                      #'MetaHIT'  
                      #'Qin_et_al'  
                      #'RA'
                      #'RA_no_adapter',
                      'Zeller_2014'
                      ]


config['kmer_size'] = [5, 6, 7 ,8, 10]

config['norm_input'] = [True] #  standardize the data


### params to iterate over ###
config['encoding_dim'] = [10, 50, 100, 200, 500]

config['encoded_activation'] = ['relu', 'sigmoid', 'linear', 'softmax', 'tanh']

config['input_dropout_pct'] = [0, 0.1, 0.25, 0.5, 0.75]

config['dropout_pct'] = [0, 0.1, 0.25, 0.5, 0.75]

config['num_epochs'] = [400]
                    
config['batch_size'] = [16]
###                        ###

config['n_splits'] = [5]

config['n_repeats'] = [1]

config['compute_informative_features'] = [False]

config['plot_iteration'] = [False]

config['graph_dir'] = ['/pollard/home/ngarud/deep_learning_microbiome/analysis']




data_sets=config['data_set']
kmer_sizes=config['kmer_size']
norm_inputs=config['norm_input']
encoding_dims=config['encoding_dim']
encoded_activations=config['encoded_activation']
input_dropout_pcts=config['input_dropout_pct']
dropout_pcts=config['dropout_pct'] 
num_epochss=config['num_epochs']
batch_sizes=config['batch_size']
n_splitss=config['n_splits']
n_repeatss=config['n_repeats']
compute_informative_features=config['compute_informative_features'][0]
plot_iteration=config['plot_iteration'][0] 
graph_dir=config['graph_dir'][0] 

outFN_list=os.path.expanduser('~/deep_learning_microbiome/tmp_intermediate_files/config_files/list_of_configs_Zeller.txt' ) 
outFile_list=open(outFN_list,'a')

for data_set in data_sets:
    for kmer_size in kmer_sizes:
        for norm_input in norm_inputs:
            for encoding_dim in encoding_dims:
                for encoded_activation in encoded_activations:
                    for input_dropout_pct in input_dropout_pcts:
                        for dropout_pct in dropout_pcts:
                            for num_epochs in num_epochss:
                                for batch_size in batch_sizes:
                                    for n_splits in n_splitss:
                                        for n_repeats in n_repeatss:
                                            summary_string='_'.join( (data_set, str(kmer_size), str(norm_input), str(encoding_dim), encoded_activation, str(input_dropout_pct), str(dropout_pct), str(num_epochs), str(batch_size), str(n_splits), str(n_repeats)) ) 

                                            outFN=os.path.expanduser('~/deep_learning_microbiome/tmp_intermediate_files/config_files/' + summary_string + '_config.py' )

                                            #print the outFN to a file so that I can submit these as jobs to the cluster
                                            outFile_list.write(outFN + '\n')

                                            outFile=open(outFN,'w')
                                            
                                            outFile.write('import os\n')
                                            outFile.write('data_directory = os.path.expanduser("~/deep_learning_microbiome/data/")')
                                            outFile.write('\n')
                                            outFile.write('analysis_directory = os.path.expanduser("~/deep_learning_microbiome/analysis/")')
                                            outFile.write('\n')
                                            outFile.write('scripts_directory = os.path.expanduser("~/deep_learning_microbiome/scripts/")')
                                            outFile.write('\n')
                                            outFile.write('config={}\n')
                                            
                                            outFile.write("config['data_set'] = ['%s']\n" %data_set)

                                            outFile.write("config['kmer_size'] = [%s]\n" %kmer_size)
                                            outFile.write("config['norm_input'] = [%s]\n" %norm_input)
                                            outFile.write("config['encoding_dim'] = [%s]\n" %encoding_dim)
                                            outFile.write("config['encoded_activation'] = ['%s']\n" %encoded_activation)
                                            outFile.write("config['input_dropout_pct'] = [%s]\n" %input_dropout_pct)
                                            outFile.write("config['dropout_pct'] = [%s]\n" %dropout_pct)
                                            outFile.write("config['num_epochs'] = [%s]\n" %num_epochs)
                                            outFile.write("config['batch_size'] = [%s]\n" %batch_size)
                                            outFile.write("config['n_splits'] = [%s]\n" %n_splits)
                                            outFile.write("config['n_repeats'] = [%s]\n" %n_repeats)




                                            outFile.write("config['compute_informative_features'] = [%s]\n" %compute_informative_features)
                                            outFile.write("config['plot_iteration'] = [%s]\n" %plot_iteration)
                                            outFile.write("config['graph_dir'] = ['%s']\n" %graph_dir)


