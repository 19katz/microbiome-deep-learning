#!/usr/bin/env python3
#~/miniconda3/bin/python3
import matplotlib
matplotlib.use('Agg') # this suppresses the console for plotting
import matplotlib.pyplot as plt
import pylab

import numpy as np
import sys
import os.path
plt.rcParams.update({'font.size': 6})

#inFN=sys.argv[1]

outFN='~/deep_learning_microbiome/analysis/best_results.txt'
outFile=open(os.path.expanduser(outFN), 'w')
outFile.write('data_set\tkmer_size\tencoding_dim\tencoded_activation\tinput_dropout_pct\tdropout_pct\t' + '\t'.join(['val_acc','val_acc_se','precision','precision_se','recall','recall_se','f1','f1_se','auc','auc_se']) + '\n')


for grid_search in ['Feng','Karlsson','LiverCirrhosis','MetaHIT','RA','LeChatelier','Qin', 'Zeller']:

    inFN='~/deep_learning_microbiome/analysis/%s_gridsearch.txt' %grid_search
    inFile=open(os.path.expanduser(inFN), 'r')

    # initialize a config dict to store all the results:
    config_dict={}
    for data_set in ['Feng','Karlsson_2013','Karlsson_2013_no_adapter','LiverCirrhosis','MetaHIT','RA','RA_no_adapter','LeChatelier','Qin_et_al', 'Zeller_2014']:
        config_dict[data_set]={}
        for kmer_size in [5,6,7,8,10]:
            config_dict[data_set][kmer_size]={}
            for encoding_dim in [8, 10, 50, 100, 200, 500]:
                config_dict[data_set][kmer_size][encoding_dim]={}
                for encoded_activation in ['linear','relu','sigmoid','softmax','tanh']:
                    config_dict[data_set][kmer_size][encoding_dim][encoded_activation]={}
                    for input_dropout_pct in [0, 0.1, 0.25, 0.5, 0.75]:
                        config_dict[data_set][kmer_size][encoding_dim][encoded_activation][input_dropout_pct]={}
                        for dropout_pct in [0, 0.1, 0.25, 0.5, 0.75]:
                            config_dict[data_set][kmer_size][encoding_dim][encoded_activation][input_dropout_pct][dropout_pct]={'val_acc':'nan','f1':'nan', 'auc':'nan'}

    # read in the data
    for line in inFile: 
        line=line.strip('\n')
        items=line.strip('\n').split('\t')
        if items[0] != 'data_set' and len(items)==28:
            data_set=items[0]
            kmer_size=int(items[1])
            norm_input=items[2]
            encoding_dim=int(items[3])
            encoded_activation=items[4]   
            input_dropout_pct=float(items[5])
            dropout_pct=float(items[6])
            num_epochs=items[7]
            batch_size=items[8]
            n_splits=items[9]
            n_repeats=items[10]
            val_acc=float(items[11])
            val_acc_se=items[12]
            acc=items[13]
            acc_se=items[14]
            val_loss=items[15]
            val_loss_se=items[16]
            loss=items[17]
            loss_se=items[18]
            f1=float(items[19])
            f1_se=items[20]
            precision=items[21]
            precision_se=items[22]
            recall=items[23]
            recall_se=items[24]
            auc=float(items[25])
            auc_se=items[26]
    
            # store in a vector if meeting criteria of interest
            config_dict[data_set][kmer_size][encoding_dim][encoded_activation][input_dropout_pct][dropout_pct]={'val_acc':val_acc,'val_acc_se':val_acc_se, 'precision': precision, 'precision_se': precision_se, 'recall':recall,'recall_se':recall_se, 'f1':f1, 'f1_se':f1_se, 'auc':auc, 'auc_se':auc_se}


    ##############################################################
    # Print results for different kmer sizes                     #
    ##############################################################


    if grid_search=='RA':
        data_sets= ['RA', 'RA_no_adapter']
    elif grid_search=='Karlsson':
        data_sets= ['Karlsson_2013', 'Karlsson_2013_no_adapter']
    else:
        data_sets=[data_set]

    for data_set in data_sets:

        best_params={}
        for kmer_size in [5, 6, 7, 8, 10]:
            best_params[kmer_size]=[0,'NA',0,0,0] # encoding_dim, encoded_activation, input_dropout_pct, dropout_pct, val_acc

        for kmer_size in [5, 6, 7, 8, 10]:
            for encoding_dim in [10, 50, 100, 200, 500]:
                for encoded_activation in ['relu', 'sigmoid', 'linear', 'softmax', 'tanh']:
                    for input_dropout_pct in [0, 0.1, 0.25, 0.5, 0.75]:
                        for dropout_pct in [0, 0.1, 0.25, 0.5, 0.75]:
                            val_acc=config_dict[data_set][kmer_size][encoding_dim][encoded_activation][input_dropout_pct][dropout_pct]['val_acc'] 
                            if float(val_acc) > best_params[kmer_size][4] and val_acc != 'nan':
                                best_params[kmer_size]=[encoding_dim, encoded_activation, input_dropout_pct, dropout_pct,val_acc]
    
            # to print:
            encoding_dim=best_params[kmer_size][0]
            encoded_activation=best_params[kmer_size][1]
            input_dropout_pct=best_params[kmer_size][2]
            dropout_pct=best_params[kmer_size][3]
            val_acc=best_params[kmer_size][4]
        
            s=''
            for param in ['val_acc','val_acc_se','precision','precision_se','recall','recall_se','f1','f1_se','auc','auc_se']:
                if val_acc != 0:
                    s+=str(config_dict[data_set][kmer_size][encoding_dim][encoded_activation][input_dropout_pct][dropout_pct][param]) + '\t' 
                else:
                    s+='nan\t'

            outFile.write('\t'.join([data_set, str(kmer_size), str(encoding_dim), str(encoded_activation), str(input_dropout_pct), str(dropout_pct)]) + '\t' + s +'\n')
    outFile.write('\n')
    
