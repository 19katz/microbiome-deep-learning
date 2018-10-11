import matplotlib
matplotlib.use('Agg') # this suppresses the console for plotting
import matplotlib.pyplot as plt
import bz2
import numpy as np
from numpy import random
import pandas as pd
import os
import pylab
from importlib import reload
from sklearn.preprocessing import normalize
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import History, TensorBoard
from keras import backend as K
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve
import itertools
from itertools import cycle, product
from sklearn.model_selection import StratifiedKFold

import config_file_local as config_file

data_directory = config_file.data_directory
analysis_directory = config_file.analysis_directory  
scripts_directory = config_file.scripts_directory 


def format_plotting_string(data_set, kmer_size, norm_input, encoding_dim, encoded_activation, input_dropout_pct, dropout_pct, num_epochs, batch_size, n_splits, n_repeats):


    TF_dictionary={True:'T',False:'F'}

    plotting_string=''
    plotting_string+=data_set + '_'
    plotting_string+=str(kmer_size) +'_'
    plotting_string+='N:'+TF_dictionary[norm_input] +'_'
    plotting_string+='ED:'+ str(encoding_dim) +'_'
    plotting_string+='EA:'+ encoded_activation +'_'
    plotting_string+='IDP:' +str(input_dropout_pct) +'_'
    plotting_string+='DP:' + str(dropout_pct)

    return plotting_string


def format_plotting_string_2layers(data_set, kmer_size, norm_input, encoding_dim_1, encoding_dim_2, encoded_activation, input_dropout_pct, dropout_pct, num_epochs, batch_size, n_splits, n_repeats):


    TF_dictionary={True:'T',False:'F'}

    plotting_string=''
    plotting_string+=data_set + '_'
    plotting_string+=str(kmer_size) +'_'
    plotting_string+='N:'+TF_dictionary[norm_input] +'_'
    plotting_string+='ED1:'+ str(encoding_dim_1) +'_'
    plotting_string+='ED2:'+ str(encoding_dim_2) +'_'
    plotting_string+='EA:'+ encoded_activation +'_'
    plotting_string+='IDP:' +str(input_dropout_pct) +'_'
    plotting_string+='DP:' + str(dropout_pct)

    return plotting_string



def format_plotting_string_3layers(data_set, kmer_size, norm_input, encoding_dim_1, encoding_dim_2, encoding_dim_3, encoded_activation, input_dropout_pct, dropout_pct, num_epochs, batch_size, n_splits, n_repeats):


    TF_dictionary={True:'T',False:'F'}

    plotting_string=''
    plotting_string+=data_set + '_'
    plotting_string+=str(kmer_size) +'_'
    plotting_string+='N:'+TF_dictionary[norm_input] +'_'
    plotting_string+='ED1:'+ str(encoding_dim_1) +'_'
    plotting_string+='ED2:'+ str(encoding_dim_2) +'_'
    plotting_string+='ED3:'+ str(encoding_dim_3) +'_'
    plotting_string+='EA:'+ encoded_activation +'_'
    plotting_string+='IDP:' +str(input_dropout_pct) +'_'
    plotting_string+='DP:' + str(dropout_pct)

    return plotting_string



def plot_confusion_matrix(cm, classes, file_name):
    
    cmap=pylab.cm.Reds
    """
    This function plots the confusion matrix.
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    pylab.figure()
    im = pylab.imshow(cm, interpolation='nearest', cmap=cmap)
    pylab.title('confusion_matrix')
    pylab.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        pylab.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    pylab.xlabel('Predicted label')
    pylab.ylabel('True label')
    pylab.gca().set_position((.1, 10, 0.8, .8))

    pylab.savefig(file_name , bbox_inches='tight')






def plot_roc_aucs(fpr, tpr, auc, acc, graph_dir):
    title='ROC Curves, auc=%s, acc=%s' %(auc,acc)
    pylab.figure()

    pylab.plot([0, 1], [0, 1], 'k--')
    pylab.plot(fpr, tpr)
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.05])
    pylab.xlabel('False Positive Rate')
    pylab.ylabel('True Positive Rate')
    pylab.title(title)
    #pylab.gca().set_position((.1, .7, .8, .8))


    pylab.savefig(os.path.expanduser(graph_dir + '/roc.pdf'))



def plot_loss_vs_epoch(history, graph_dir):

    pylab.figure()
    pylab.plot(history.history['loss'])
    pylab.plot(history.history['val_loss'])
    pylab.legend(['training','test'], loc='upper right')
    
    pylab.title('Model loss by epochs')
    pylab.ylabel('Loss')
    pylab.xlabel('Epoch')

    pylab.savefig(os.path.expanduser(graph_dir + '/Loss.pdf') , bbox_inches='tight')



def plot_accuracy_vs_epoch(history, graph_dir):
    pylab.figure()
    pylab.plot(history.history['acc'])
    pylab.plot(history.history['val_acc'])
    pylab.legend(['training','test'], loc='upper right')
    
    pylab.title('Model accuracy by epochs')
    pylab.ylabel('Accuracy')
    pylab.xlabel('Epoch')
    
    pylab.savefig(os.path.expanduser(graph_dir + '/accuracy.pdf') , bbox_inches='tight')



def plot_precision_recall(precision, recall, f1_score, plotting_string):
    pylab.figure()

    pylab.step(recall, precision, color='b', alpha=0.2, where='post')
    pylab.fill_between(recall, precision, step='post', alpha=0.2, color='b')

    pylab.xlabel('Recall')
    pylab.ylabel('Precision')
    pylab.ylim([0.0, 1.05])

    pylab.xlim([0.0, 1.0])

    print('Saving figure %sprecision_recall_%s.pdf' %(analysis_directory,plotting_string) )
    
    pylab.savefig(os.path.expanduser('%sprecision_recall_%s.pdf' %(analysis_directory, plotting_string)) , bbox_inches='tight')



def plot_all_for_iteration(aggregated_statistics, n_repeat, history, y_test, y_pred):

    graph_dir=os.path.expanduser('~/deep_learning_microbiome/analysis')

    fpr=aggregated_statistics[n_repeat]['fpr']
    tpr=aggregated_statistics[n_repeat]['tpr']
    auc=aggregated_statistics[n_repeat]['auc']
    accuracy=aggregated_statistics[n_repeat]['accuracy']
    f1=aggregated_statistics[n_repeat]['f1']
    conf_mat=aggregated_statistics[n_repeat]['conf_mat']

    # plot roc: 
    plot_roc_aucs(fpr, tpr, auc, accuracy, graph_dir)

    # Plot accuracy vs epoch 
    plot_accuracy_vs_epoch(history, graph_dir)   

    # plot loss vs epoch     
    plot_loss_vs_epoch(history, graph_dir)

    # Plot a confusion matrix 
    file_name=os.path.expanduser(graph_dir + 'confusion_matrix.pdf')
    classes=['0','1']
    plot_confusion_matrix(conf_mat, classes,file_name)

    # Plot precision_recall  
    precision_graph, recall_graph, _ = precision_recall_curve(y_test, y_pred)
    plot_precision_recall(precision_graph, recall_graph, f1, graph_dir)

