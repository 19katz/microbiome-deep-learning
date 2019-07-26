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

inFN=sys.argv[1]

inFN='~/deep_learning_microbiome/analysis/LeChatelier_gridsearch.txt'
inFile=open(os.path.expanduser(inFN), 'r')

# initialize a config dict to store all the results:
config_dict={}
for kmer_size in [5,6,7,8,10]:
    config_dict[kmer_size]={}
    for encoding_dim in [8, 10, 50, 100, 200, 500]:
        config_dict[kmer_size][encoding_dim]={}
        for encoded_activation in ['linear','relu','sigmoid','softmax','tanh']:
            config_dict[kmer_size][encoding_dim][encoded_activation]={}
            for input_dropout_pct in [0, 0.1, 0.25, 0.5, 0.75]:
                config_dict[kmer_size][encoding_dim][encoded_activation][input_dropout_pct]={}
                for dropout_pct in [0, 0.1, 0.25, 0.5, 0.75]:
                    config_dict[kmer_size][encoding_dim][encoded_activation][input_dropout_pct][dropout_pct]={'val_acc':'nan','f1':'nan', 'auc':'nan'}


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
        config_dict[kmer_size][encoding_dim][encoded_activation][input_dropout_pct][dropout_pct]={'val_acc':val_acc,'f1':f1, 'auc':auc}


##############################################################
# Plot acc for different kmers as a function of encodng dims #
##############################################################


# create vectors to plot:
encoded_activation='sigmoid'
input_dropout_pct=0.25
dropout_pct=0

data_to_plot={}
for kmer_size in [5,6,7,8,10]:
    data_to_plot[kmer_size]=[]
    for encoding_dim in [10, 50, 100, 200, 500]:
        data_to_plot[kmer_size].append(config_dict[kmer_size][encoding_dim][encoded_activation][input_dropout_pct][dropout_pct]['val_acc'])

fig=plt.figure(figsize=(12,8))
ax=fig.add_subplot(221)   
ax.set_ylabel('val_acc')
ax.set_xlabel('encoding dimensions')
ax.set_title('%s, Encoding act: %s, dropout=%s, input dropout =%s' %(data_set, encoded_activation,dropout_pct,input_dropout_pct))

ax.plot([10, 50, 100, 200, 500],data_to_plot[5],"o-", color='#d7191c', label='5')
ax.plot([10, 50, 100, 200, 500],data_to_plot[6],"o-", color='#fdae61', label='6')
ax.plot([10, 50, 100, 200, 500],data_to_plot[7],"o-", color='gold', label='7')
ax.plot([10, 50, 100, 200, 500],data_to_plot[8],"o-", color='#abdda4', label='8')
ax.plot([10, 50, 100, 200, 500],data_to_plot[10],"o-", color='#2b83ba', label='10')

ax.set_ylim(ymin=0.5,ymax=0.9)

plt.legend(loc="upper left", bbox_to_anchor=[0, 1])
ax.get_legend()
#ax.legend([mer5, mer6, mer7, mer8, mer10], ['5mer','6mer','7mer','8mer','10mer'],loc=4)

########################
#encoded_activation='relu'
input_dropout_pct=0.25
dropout_pct=0

data_to_plot={}
for kmer_size in [5,6,7,8,10]:
    data_to_plot[kmer_size]=[]
    for encoding_dim in [10, 50, 100, 200, 500]:
        data_to_plot[kmer_size].append(config_dict[kmer_size][encoding_dim][encoded_activation][input_dropout_pct][dropout_pct]['val_acc'])


ax=fig.add_subplot(222)   
ax.set_ylabel('val_acc')
ax.set_xlabel('encoding dimensions')
ax.set_title('%s, Encoding act: %s, dropout=%s, input dropout =%s' %(data_set, encoded_activation,dropout_pct,input_dropout_pct))

ax.plot([10, 50, 100, 200, 500],data_to_plot[5],"o-", color='#d7191c', label='5')
ax.plot([10, 50, 100, 200, 500],data_to_plot[6],"o-", color='#fdae61', label='6')
ax.plot([10, 50, 100, 200, 500],data_to_plot[7],"o-", color='gold', label='7')
ax.plot([10, 50, 100, 200, 500],data_to_plot[8],"o-", color='#abdda4', label='8')
ax.plot([10, 50, 100, 200, 500],data_to_plot[10],"o-", color='#2b83ba', label='10')

ax.set_ylim(ymin=0.5,ymax=0.9)


########################
#encoded_activation='relu'
input_dropout_pct=0.5
dropout_pct=0

data_to_plot={}
for kmer_size in [5,6,7,8,10]:
    data_to_plot[kmer_size]=[]
    for encoding_dim in [10, 50, 100, 200, 500]:
        data_to_plot[kmer_size].append(config_dict[kmer_size][encoding_dim][encoded_activation][input_dropout_pct][dropout_pct]['val_acc'])


ax=fig.add_subplot(223)   
ax.set_ylabel('val_acc')
ax.set_xlabel('encoding dimensions')
ax.set_title('%s, Encoding act: %s, dropout=%s, input dropout =%s' %(data_set, encoded_activation,dropout_pct,input_dropout_pct))


ax.plot([10, 50, 100, 200, 500],data_to_plot[5],"o-", color='#d7191c', label='5')
ax.plot([10, 50, 100, 200, 500],data_to_plot[6],"o-", color='#fdae61', label='6')
ax.plot([10, 50, 100, 200, 500],data_to_plot[7],"o-", color='gold', label='7')
ax.plot([10, 50, 100, 200, 500],data_to_plot[8],"o-", color='#abdda4', label='8')
ax.plot([10, 50, 100, 200, 500],data_to_plot[10],"o-", color='#2b83ba', label='10')

ax.set_ylim(ymin=0.5,ymax=0.9)




########################
#encoded_activation='relu'
input_dropout_pct=0.75
dropout_pct=0

data_to_plot={}
for kmer_size in [5,6,7,8,10]:
    data_to_plot[kmer_size]=[]
    for encoding_dim in [10, 50, 100, 200, 500]:
        data_to_plot[kmer_size].append(config_dict[kmer_size][encoding_dim][encoded_activation][input_dropout_pct][dropout_pct]['val_acc'])


ax=fig.add_subplot(224)   
ax.set_ylabel('val_acc')
ax.set_xlabel('encoding dimensions')
ax.set_title('%s, Encoding act: %s, dropout=%s, input dropout =%s' %(data_set, encoded_activation,dropout_pct,input_dropout_pct))

ax.plot([10, 50, 100, 200, 500],data_to_plot[5],"o-", color='#d7191c', label='5')
ax.plot([10, 50, 100, 200, 500],data_to_plot[6],"o-", color='#fdae61', label='6')
ax.plot([10, 50, 100, 200, 500],data_to_plot[7],"o-", color='gold', label='7')
ax.plot([10, 50, 100, 200, 500],data_to_plot[8],"o-", color='#abdda4', label='8')
ax.plot([10, 50, 100, 200, 500],data_to_plot[10],"o-", color='#2b83ba', label='10')

ax.set_ylim(ymin=0.5,ymax=0.9)


pylab.savefig(os.path.expanduser('~/deep_learning_microbiome/analysis/gridsearch_results_1.pdf'), bbox_inches='tight')







####################################################################
# Plot acc for different activations as a function of encodng dims #
####################################################################
# create vectors to plot:
kmer_size=5
input_dropout_pct=0
dropout_pct=0

data_to_plot={}
for encoded_activation in ['linear','relu','sigmoid','softmax','tanh']:
    data_to_plot[encoded_activation]=[]
    for encoding_dim in [10, 50, 100, 200, 500]:
        data_to_plot[encoded_activation].append(config_dict[kmer_size][encoding_dim][encoded_activation][input_dropout_pct][dropout_pct]['val_acc'])

fig=plt.figure(figsize=(12,8))
ax=fig.add_subplot(221)   
ax.set_ylabel('val_acc')
ax.set_xlabel('encoding dimensions')
ax.set_title('%s, Kmer size: %s, dropout=%s, input dropout =%s' %(data_set, kmer_size,dropout_pct,input_dropout_pct))

ax.plot([10, 50, 100, 200, 500],data_to_plot['linear'],"o-", color='#d7191c', label='linear')
ax.plot([10, 50, 100, 200, 500],data_to_plot['relu'],"o-", color='#fdae61', label='relu')
ax.plot([10, 50, 100, 200, 500],data_to_plot['sigmoid'],"o-", color='gold', label='sigmoid')
ax.plot([10, 50, 100, 200, 500],data_to_plot['softmax'],"o-", color='#abdda4', label='softmax')
ax.plot([10, 50, 100, 200, 500],data_to_plot['tanh'],"o-", color='#2b83ba', label='tanh')

ax.set_ylim(ymin=0.5,ymax=0.9)

plt.legend(loc="upper left", bbox_to_anchor=[0, 1])
ax.get_legend()

##################################

kmer_size=7

data_to_plot={}
for encoded_activation in ['linear','relu','sigmoid','softmax','tanh']:
    data_to_plot[encoded_activation]=[]
    for encoding_dim in [10, 50, 100, 200, 500]:
        data_to_plot[encoded_activation].append(config_dict[kmer_size][encoding_dim][encoded_activation][input_dropout_pct][dropout_pct]['val_acc'])

ax=fig.add_subplot(222)   
ax.set_ylabel('val_acc')
ax.set_xlabel('encoding dimensions')
ax.set_title('%s, Kmer size: %s, dropout=%s, input dropout =%s' %(data_set, kmer_size,dropout_pct,input_dropout_pct))

ax.plot([10, 50, 100, 200, 500],data_to_plot['linear'],"o-", color='#d7191c', label='linear')
ax.plot([10, 50, 100, 200, 500],data_to_plot['relu'],"o-", color='#fdae61', label='relu')
ax.plot([10, 50, 100, 200, 500],data_to_plot['sigmoid'],"o-", color='gold', label='sigmoid')
ax.plot([10, 50, 100, 200, 500],data_to_plot['softmax'],"o-", color='#abdda4', label='softmax')
ax.plot([10, 50, 100, 200, 500],data_to_plot['tanh'],"o-", color='#2b83ba', label='tanh')

ax.set_ylim(ymin=0.5,ymax=0.9)



#######################

kmer_size=8

data_to_plot={}
for encoded_activation in ['linear','relu','sigmoid','softmax','tanh']:
    data_to_plot[encoded_activation]=[]
    for encoding_dim in [10, 50, 100, 200, 500]:
        data_to_plot[encoded_activation].append(config_dict[kmer_size][encoding_dim][encoded_activation][input_dropout_pct][dropout_pct]['val_acc'])

ax=fig.add_subplot(223)   
ax.set_ylabel('val_acc')
ax.set_xlabel('encoding dimensions')
ax.set_title('%s, Kmer size: %s, dropout=%s, input dropout =%s' %(data_set, kmer_size,dropout_pct,input_dropout_pct))

ax.plot([10, 50, 100, 200, 500],data_to_plot['linear'],"o-", color='#d7191c', label='linear')
ax.plot([10, 50, 100, 200, 500],data_to_plot['relu'],"o-", color='#fdae61', label='relu')
ax.plot([10, 50, 100, 200, 500],data_to_plot['sigmoid'],"o-", color='gold', label='sigmoid')
ax.plot([10, 50, 100, 200, 500],data_to_plot['softmax'],"o-", color='#abdda4', label='softmax')
ax.plot([10, 50, 100, 200, 500],data_to_plot['tanh'],"o-", color='#2b83ba', label='tanh')

ax.set_ylim(ymin=0.5,ymax=0.9)



########################
kmer_size=10

data_to_plot={}
for encoded_activation in ['linear','relu','sigmoid','softmax','tanh']:
    data_to_plot[encoded_activation]=[]
    for encoding_dim in [10, 50, 100, 200, 500]:
        data_to_plot[encoded_activation].append(config_dict[kmer_size][encoding_dim][encoded_activation][input_dropout_pct][dropout_pct]['val_acc'])

ax=fig.add_subplot(224)   
ax.set_ylabel('val_acc')
ax.set_xlabel('encoding dimensions')
ax.set_title('%s, Kmer size: %s, dropout=%s, input dropout =%s' %(data_set, kmer_size,dropout_pct,input_dropout_pct))

ax.plot([10, 50, 100, 200, 500],data_to_plot['linear'],"o-", color='#d7191c', label='linear')
ax.plot([10, 50, 100, 200, 500],data_to_plot['relu'],"o-", color='#fdae61', label='relu')
ax.plot([10, 50, 100, 200, 500],data_to_plot['sigmoid'],"o-", color='gold', label='sigmoid')
ax.plot([10, 50, 100, 200, 500],data_to_plot['softmax'],"o-", color='#abdda4', label='softmax')
ax.plot([10, 50, 100, 200, 500],data_to_plot['tanh'],"o-", color='#2b83ba', label='tanh')

ax.set_ylim(ymin=0.5,ymax=0.9)

pylab.savefig(os.path.expanduser('~/deep_learning_microbiome/analysis/gridsearch_results_2.pdf'), bbox_inches='tight')


#############################################################
# Plot acc for different kmers as a function of dropout     #
#############################################################

# create vectors to plot:
encoded_activation='relu'
dropout_pct=0
encoding_dim=10

data_to_plot={}
for input_dropout_pct in [0, 0.1, 0.25, 0.5, 0.75]:
    data_to_plot[input_dropout_pct]=[]
    for kmer_size in [5, 6, 7, 8, 10]:
        data_to_plot[input_dropout_pct].append(config_dict[kmer_size][encoding_dim][encoded_activation][input_dropout_pct][dropout_pct]['val_acc'])

fig=plt.figure(figsize=(12,8))
ax=fig.add_subplot(221)   
ax.set_ylabel('val_acc')
ax.set_xlabel('kmer size')
ax.set_title('%s, Encoding act: %s, dropout=%s, encoding_dim =%s' %(data_set, encoded_activation,dropout_pct,encoding_dim))

ax.plot([5, 6, 7, 8, 10],data_to_plot[0],"o-", color='#d7191c', label='0')
ax.plot([5, 6, 7, 8, 10],data_to_plot[0.1],"o-", color='#fdae61', label='0.1')
ax.plot([5, 6, 7, 8, 10],data_to_plot[0.25],"o-", color='gold', label='0.25')
ax.plot([5, 6, 7, 8, 10],data_to_plot[0.5],"o-", color='#abdda4', label='0.5')
ax.plot([5, 6, 7, 8, 10],data_to_plot[0.75],"o-", color='#2b83ba', label='0.75')

ax.set_ylim(ymin=0.5,ymax=0.9)

plt.legend(loc="upper left", bbox_to_anchor=[0, 1])
ax.get_legend()


##########
encoding_dim=50

data_to_plot={}
for input_dropout_pct in [0, 0.1, 0.25, 0.5, 0.75]:
    data_to_plot[input_dropout_pct]=[]
    for kmer_size in [5, 6, 7, 8, 10]:
        data_to_plot[input_dropout_pct].append(config_dict[kmer_size][encoding_dim][encoded_activation][input_dropout_pct][dropout_pct]['val_acc'])

ax=fig.add_subplot(222)   
ax.set_ylabel('val_acc')
ax.set_xlabel('kmer size')
ax.set_title('%s, Encoding act: %s, dropout=%s, encoding_dim =%s' %(data_set, encoded_activation,dropout_pct,encoding_dim))

ax.plot([5, 6, 7, 8, 10],data_to_plot[0],"o-", color='#d7191c', label='0')
ax.plot([5, 6, 7, 8, 10],data_to_plot[0.1],"o-", color='#fdae61', label='0.1')
ax.plot([5, 6, 7, 8, 10],data_to_plot[0.25],"o-", color='gold', label='0.25')
ax.plot([5, 6, 7, 8, 10],data_to_plot[0.5],"o-", color='#abdda4', label='0.5')
ax.plot([5, 6, 7, 8, 10],data_to_plot[0.75],"o-", color='#2b83ba', label='0.75')

ax.set_ylim(ymin=0.5,ymax=0.9)



##########
encoding_dim=100

data_to_plot={}
for input_dropout_pct in [0, 0.1, 0.25, 0.5, 0.75]:
    data_to_plot[input_dropout_pct]=[]
    for kmer_size in [5, 6, 7, 8, 10]:
        data_to_plot[input_dropout_pct].append(config_dict[kmer_size][encoding_dim][encoded_activation][input_dropout_pct][dropout_pct]['val_acc'])

ax=fig.add_subplot(223)   
ax.set_ylabel('val_acc')
ax.set_xlabel('kmer size')
ax.set_title('%s, Encoding act: %s, dropout=%s, encoding_dim =%s' %(data_set, encoded_activation,dropout_pct,encoding_dim))

ax.plot([5, 6, 7, 8, 10],data_to_plot[0],"o-", color='#d7191c', label='0')
ax.plot([5, 6, 7, 8, 10],data_to_plot[0.1],"o-", color='#fdae61', label='0.1')
ax.plot([5, 6, 7, 8, 10],data_to_plot[0.25],"o-", color='gold', label='0.25')
ax.plot([5, 6, 7, 8, 10],data_to_plot[0.5],"o-", color='#abdda4', label='0.5')
ax.plot([5, 6, 7, 8, 10],data_to_plot[0.75],"o-", color='#2b83ba', label='0.75')

ax.set_ylim(ymin=0.5,ymax=0.9)


############
encoding_dim=200

data_to_plot={}
for input_dropout_pct in [0, 0.1, 0.25, 0.5, 0.75]:
    data_to_plot[input_dropout_pct]=[]
    for kmer_size in [5, 6, 7, 8, 10]:
        data_to_plot[input_dropout_pct].append(config_dict[kmer_size][encoding_dim][encoded_activation][input_dropout_pct][dropout_pct]['val_acc'])

ax=fig.add_subplot(224)   
ax.set_ylabel('val_acc')
ax.set_xlabel('kmer size')
ax.set_title('%s, Encoding act: %s, dropout=%s, encoding_dim =%s' %(data_set, encoded_activation,dropout_pct,encoding_dim))

ax.plot([5, 6, 7, 8, 10],data_to_plot[0],"o-", color='#d7191c', label='0')
ax.plot([5, 6, 7, 8, 10],data_to_plot[0.1],"o-", color='#fdae61', label='0.1')
ax.plot([5, 6, 7, 8, 10],data_to_plot[0.25],"o-", color='gold', label='0.25')
ax.plot([5, 6, 7, 8, 10],data_to_plot[0.5],"o-", color='#abdda4', label='0.5')
ax.plot([5, 6, 7, 8, 10],data_to_plot[0.75],"o-", color='#2b83ba', label='0.75')

ax.set_ylim(ymin=0.5,ymax=0.9)


pylab.savefig(os.path.expanduser('~/deep_learning_microbiome/analysis/gridsearch_results_3.pdf'), bbox_inches='tight')


###################################
# Plot dropout vs input dropout
###################################

# create vectors to plot:
encoded_activation='sigmoid'
kmer_size=8
encoding_dim=10

data_to_plot={}
for dropout_pct in [0, 0.1, 0.25, 0.5, 0.75]:
    data_to_plot[dropout_pct]=[]
    for input_dropout_pct in [0, 0.1, 0.25, 0.5, 0.75]:
        data_to_plot[dropout_pct].append(config_dict[kmer_size][encoding_dim][encoded_activation][input_dropout_pct][dropout_pct]['val_acc'])

fig=plt.figure(figsize=(12,8))
ax=fig.add_subplot(221)   
ax.set_ylabel('val_acc')
ax.set_xlabel('input dropout pct')
ax.set_title('%s, Encoding act: %s, kmer_size=%s, encoding_dim =%s' %(data_set, encoded_activation,kmer_size,encoding_dim))

ax.plot([0, 0.1, 0.25, 0.5, 0.75],data_to_plot[0],"o-", color='#d7191c', label='0')
ax.plot([0, 0.1, 0.25, 0.5, 0.75],data_to_plot[0.1],"o-", color='#fdae61', label='0.1')
ax.plot([0, 0.1, 0.25, 0.5, 0.75],data_to_plot[0.25],"o-", color='gold', label='0.25')
ax.plot([0, 0.1, 0.25, 0.5, 0.75],data_to_plot[0.5],"o-", color='#abdda4', label='0.5')
ax.plot([0, 0.1, 0.25, 0.5, 0.75],data_to_plot[0.75],"o-", color='#2b83ba', label='0.75')

ax.set_ylim(ymin=0.5,ymax=0.9)

plt.legend(loc="upper left", bbox_to_anchor=[0, 1])
ax.get_legend()


##########
encoding_dim=50

data_to_plot={}
for dropout_pct in [0, 0.1, 0.25, 0.5, 0.75]:
    data_to_plot[dropout_pct]=[]
    for input_dropout_pct in [0, 0.1, 0.25, 0.5, 0.75]:
        data_to_plot[dropout_pct].append(config_dict[kmer_size][encoding_dim][encoded_activation][input_dropout_pct][dropout_pct]['val_acc'])

ax=fig.add_subplot(222)   
ax.set_ylabel('val_acc')
ax.set_xlabel('input dropout pct')
ax.set_title('%s, Encoding act: %s, kmer_size=%s, encoding_dim =%s' %(data_set, encoded_activation,kmer_size,encoding_dim))

ax.plot([0, 0.1, 0.25, 0.5, 0.75],data_to_plot[0],"o-", color='#d7191c', label='0')
ax.plot([0, 0.1, 0.25, 0.5, 0.75],data_to_plot[0.1],"o-", color='#fdae61', label='0.1')
ax.plot([0, 0.1, 0.25, 0.5, 0.75],data_to_plot[0.25],"o-", color='gold', label='0.25')
ax.plot([0, 0.1, 0.25, 0.5, 0.75],data_to_plot[0.5],"o-", color='#abdda4', label='0.5')
ax.plot([0, 0.1, 0.25, 0.5, 0.75],data_to_plot[0.75],"o-", color='#2b83ba', label='0.75')

ax.set_ylim(ymin=0.5,ymax=0.9)

##########
encoding_dim=100

data_to_plot={}
for dropout_pct in [0, 0.1, 0.25, 0.5, 0.75]:
    data_to_plot[dropout_pct]=[]
    for input_dropout_pct in [0, 0.1, 0.25, 0.5, 0.75]:
        data_to_plot[dropout_pct].append(config_dict[kmer_size][encoding_dim][encoded_activation][input_dropout_pct][dropout_pct]['val_acc'])

ax=fig.add_subplot(223)   
ax.set_ylabel('val_acc')
ax.set_xlabel('input dropout pct')
ax.set_title('%s, Encoding act: %s, kmer_size=%s, encoding_dim =%s' %(data_set, encoded_activation,kmer_size,encoding_dim))

ax.plot([0, 0.1, 0.25, 0.5, 0.75],data_to_plot[0],"o-", color='#d7191c', label='0')
ax.plot([0, 0.1, 0.25, 0.5, 0.75],data_to_plot[0.1],"o-", color='#fdae61', label='0.1')
ax.plot([0, 0.1, 0.25, 0.5, 0.75],data_to_plot[0.25],"o-", color='gold', label='0.25')
ax.plot([0, 0.1, 0.25, 0.5, 0.75],data_to_plot[0.5],"o-", color='#abdda4', label='0.5')
ax.plot([0, 0.1, 0.25, 0.5, 0.75],data_to_plot[0.75],"o-", color='#2b83ba', label='0.75')

ax.set_ylim(ymin=0.5,ymax=0.9)

############
encoding_dim=200

data_to_plot={}
for dropout_pct in [0, 0.1, 0.25, 0.5, 0.75]:
    data_to_plot[dropout_pct]=[]
    for input_dropout_pct in [0, 0.1, 0.25, 0.5, 0.75]:
        data_to_plot[dropout_pct].append(config_dict[kmer_size][encoding_dim][encoded_activation][input_dropout_pct][dropout_pct]['val_acc'])

ax=fig.add_subplot(224)   
ax.set_ylabel('val_acc')
ax.set_xlabel('input dropout pct')
ax.set_title('%s, Encoding act: %s, kmer_size=%s, encoding_dim =%s' %(data_set, encoded_activation,kmer_size,encoding_dim))

ax.plot([0, 0.1, 0.25, 0.5, 0.75],data_to_plot[0],"o-", color='#d7191c', label='0')
ax.plot([0, 0.1, 0.25, 0.5, 0.75],data_to_plot[0.1],"o-", color='#fdae61', label='0.1')
ax.plot([0, 0.1, 0.25, 0.5, 0.75],data_to_plot[0.25],"o-", color='gold', label='0.25')
ax.plot([0, 0.1, 0.25, 0.5, 0.75],data_to_plot[0.5],"o-", color='#abdda4', label='0.5')
ax.plot([0, 0.1, 0.25, 0.5, 0.75],data_to_plot[0.75],"o-", color='#2b83ba', label='0.75')

ax.set_ylim(ymin=0.5,ymax=0.9)

pylab.savefig(os.path.expanduser('~/deep_learning_microbiome/analysis/gridsearch_results_4.pdf'), bbox_inches='tight')


