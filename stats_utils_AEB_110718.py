import matplotlib
matplotlib.use('Agg') # this suppresses the console for plotting
import matplotlib.pyplot as plt
import os
import pylab
import numpy as np 
import scipy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from scipy import interp
import shap

import plotting_utils_AEB
import config_file_AEB

data_directory = config_file_AEB.data_directory
analysis_directory = config_file_AEB.analysis_directory  
scripts_directory = config_file_AEB.scripts_directory 


def standardize_data(x_train, x_test):
    sample_mean = x_train.mean(axis=0)
    sample_std = x_train.std(axis=0)
    
    # Standardize both training and test samples with the training mean and std
    x_train = (x_train - sample_mean) / sample_std
    # test samples are standardized using only the mean and std of the training samples
    x_test = (x_test - sample_mean) / sample_std

    return x_train, x_test

def standardize_data_bootstrap(x_train, x_test, x_train_bootstrap):
    sample_mean = x_train.mean(axis=0)
    sample_std = x_train.std(axis=0)
    
    # Standardize both training and test samples with the training mean and std
    x_train_bootstrap = (x_train_bootstrap - sample_mean) / sample_std
    # test samples are standardized using only the mean and std of the training samples
    x_test = (x_test - sample_mean) / sample_std

    return x_train_bootstrap, x_test


def compute_summary_statistics_autoencoder(history, aggregated_statistics, n_repeat):
    # accuracy and loss of the last epoch
    val_acc = history.history['val_acc'][-1]
    acc = history.history['acc'][-1]
    val_loss = history.history['val_loss'][-1]
    loss = history.history['loss'][-1]

    #store all of this in the dictionary:
    aggregated_statistics[n_repeat]['val_acc']=val_acc
    aggregated_statistics[n_repeat]['acc']=acc
    aggregated_statistics[n_repeat]['val_loss']=val_loss
    aggregated_statistics[n_repeat]['loss']=loss

    return aggregated_statistics    


def aggregate_statistics_across_folds_autoencoder(aggregated_statistics, rskf, n_splits, outFile, summary_string, plotting_string):
    # This definition aggregates all the information for all folds

    val_acc=np.array([])
    acc=np.array([])
    val_loss=np.array([])
    loss=np.array([])

    for n_repeat in range(0,len(rskf[0])): 
        val_acc = np.append(val_acc, aggregated_statistics[n_repeat]['val_acc'])
        acc = np.append(acc, aggregated_statistics[n_repeat]['acc'])
        val_loss = np.append(val_loss, aggregated_statistics[n_repeat]['val_loss'])
        loss = np.append(loss, aggregated_statistics[n_repeat]['loss'])

    #######################################    
    # print all statistics to an outfile  #
    #######################################

    print('Saving summary statistics to file %s%s' %(analysis_directory,outFile))

    outFN=open(os.path.expanduser('%s%s' %(analysis_directory,outFile)), 'a')       
    outFN.write('data_set\tkmer_size\tnorm_input\tencoding_dim\tencoded_activation\tinput_dropout_pct\tdropout_pct\tnum_epochs\tbatch_size\tn_splits\tn_repeats\t')
    outFN.write('val_acc\tval_acc_se\tacc\tacc_se\tval_loss\tval_loss_se\tloss\tloss_se\n')

    s=''
    for value in [val_acc, acc, val_loss, loss]:
        m, se = mean_confidence_interval(value, n_splits, confidence=0.95)
        s+= str(m) + '\t' + str(se) + '\t' 

    outFN.write(summary_string + '\t' + s +'\n')


def compute_summary_statistics(y_test, y_pred, history, aggregated_statistics, n_repeat):

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    y_pred_rounded = (y_pred > 0.5)
    conf_mat=confusion_matrix(y_test, y_pred_rounded)
    auc_val= auc(fpr,tpr) # uncomment this
    #auc =0
    accuracy=accuracy_score(y_test, y_pred_rounded)

    # compute for each class, then weight these by the size of each class (as did Pasolli)
    f1 = f1_score(y_test, y_pred_rounded, pos_label=None, average='weighted')
    precision = precision_score(y_test, y_pred_rounded, pos_label=None, average='weighted')
    recall = recall_score(y_test, y_pred_rounded, pos_label=None, average='weighted')

    # accuracy and loss of the last epoch
    val_acc = history.history['val_acc'][-1]
    acc = history.history['acc'][-1]
    val_loss = history.history['val_loss'][-1]
    loss = history.history['loss'][-1]


    #store all of this in the dictionary:
    aggregated_statistics[n_repeat]['fpr']=fpr
    aggregated_statistics[n_repeat]['tpr']=tpr
    aggregated_statistics[n_repeat]['conf_mat']=conf_mat
    aggregated_statistics[n_repeat]['auc']=auc_val
    aggregated_statistics[n_repeat]['accuracy']=accuracy
    aggregated_statistics[n_repeat]['f1']=f1
    aggregated_statistics[n_repeat]['precision']=precision
    aggregated_statistics[n_repeat]['recall']=recall
    aggregated_statistics[n_repeat]['val_acc']=val_acc
    aggregated_statistics[n_repeat]['acc']=acc
    aggregated_statistics[n_repeat]['val_loss']=val_loss
    aggregated_statistics[n_repeat]['loss']=loss
    aggregated_statistics[n_repeat]['y_test']=y_test
    aggregated_statistics[n_repeat]['y_pred']=y_pred

    return aggregated_statistics



def aggregate_statistics_across_folds(aggregated_statistics, rskf, n_splits, outFile, summary_string, plotting_string, outFile_header):
    # This definition aggregates all the information for all folds

    conf_mat=np.zeros_like(aggregated_statistics[0]['conf_mat'])
    val_acc=np.array([])
    acc=np.array([])
    val_loss=np.array([])
    loss=np.array([])
    f1=np.array([])
    precision=np.array([])
    recall=np.array([])
    auc=np.array([])
    all_y_test=aggregated_statistics[0]['y_test']
    all_y_pred=aggregated_statistics[0]['y_pred']
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for n_repeat in range(0,len(rskf[0])): 
        conf_mat+=aggregated_statistics[n_repeat]['conf_mat']
        val_acc = np.append(val_acc, aggregated_statistics[n_repeat]['val_acc'])
        acc = np.append(acc, aggregated_statistics[n_repeat]['acc'])
        val_loss = np.append(val_loss, aggregated_statistics[n_repeat]['val_loss'])
        loss = np.append(loss, aggregated_statistics[n_repeat]['loss'])
        f1 = np.append(f1, aggregated_statistics[n_repeat]['f1'])
        precision = np.append(precision,aggregated_statistics[n_repeat]['precision'])
        recall = np.append(recall, aggregated_statistics[n_repeat]['recall'])
        auc = np.append(auc, aggregated_statistics[n_repeat]['auc'])
        fpr=aggregated_statistics[n_repeat]['fpr']
        tpr=aggregated_statistics[n_repeat]['tpr']
        tprs.append(interp(mean_fpr, fpr, tpr))    
        #
        if n_repeat==0:
            all_y_test = aggregated_statistics[n_repeat]['y_test']
            all_y_pred = aggregated_statistics[n_repeat]['y_pred']
        else:
            all_y_test = np.append(all_y_test,aggregated_statistics[n_repeat]['y_test'], axis=0)
            all_y_pred = np.append(all_y_pred,aggregated_statistics[n_repeat]['y_pred'], axis=0)

    #######################################    
    # print all statistics to an outfile  #
    #######################################

    print('Saving summary statistics to file %s%s' %(analysis_directory,outFile))

    outFN=open(os.path.expanduser('%s%s' %(analysis_directory,outFile)), 'a')       
    outFN.write(outFile_header)
    outFN.write('val_acc\tval_acc_se\tacc\tacc_se\tval_loss\tval_loss_se\tloss\tloss_se\tf1\tf1_se\tprecision\tprecision_se\trecall\trecall_se\tauc\tauc_se\n')

    s=''
    for value in [val_acc, acc, val_loss, loss, f1, precision, recall, auc]:
        m, se = mean_confidence_interval(value, n_splits, confidence=0.95)
        s+= str(m) + '\t' + str(se) + '\t' 

    outFN.write(summary_string + '\t' + s +'\n')

    ###########################
    # precision, recall, curve #
    ###########################

    precision_graph, recall_graph, _ = precision_recall_curve(all_y_test, all_y_pred)
    # suppressing plotting for now
    #plotting_utils.plot_precision_recall(precision_graph, recall_graph, f1, plotting_string)

    # optional: could also add colored lines for the folds so that we can see the variance. 

    #######
    # ROC #
    #######

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
     
    # suppressing plotting for now
    '''
    pylab.figure()
    pylab.plot(mean_fpr, mean_tpr, color='b',lw=2, alpha=.8)

    # get the 95% CI
    std_tpr = np.std(tprs, axis=0)
    sem_tpr = scipy.stats.sem(tprs)
    z=calculate_z_value(0.95, n_splits)
    tprs_upper=np.minimum(mean_tpr + sem_tpr*z, 1) 
    tprs_lower=np.maximum(mean_tpr - sem_tpr*z, 0) 

    pylab.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,)

    pylab.xlim([-0.05, 1.05])
    pylab.ylim([-0.05, 1.05])
    pylab.xlabel('False Positive Rate')
    pylab.ylabel('True Positive Rate')
    pylab.title('Receiver operating characteristic curve')
    print('Saving figure %sroc_%s.pdf' %(analysis_directory,plotting_string) )
    pylab.savefig(os.path.expanduser('%sroc_%s.pdf' %(analysis_directory,plotting_string)))

    '''
def aggregate_statistics_across_folds_supervised_and_auto(aggregated_statistics, rskf, n_splits, outFile, summary_string, plotting_string, outFile_header):
    # This definition aggregates all the information for all folds

    conf_mat=np.zeros_like(aggregated_statistics[0]['conf_mat'])
    val_acc=np.array([])
    acc=np.array([])
    val_loss=np.array([])
    loss=np.array([])
    val_loss_auto=np.array([])
    f1=np.array([])
    precision=np.array([])
    recall=np.array([])
    auc=np.array([])
    all_y_test=aggregated_statistics[0]['y_test']
    all_y_pred=aggregated_statistics[0]['y_pred']
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for n_repeat in range(0,len(rskf[0])): 
        conf_mat+=aggregated_statistics[n_repeat]['conf_mat']
        val_acc = np.append(val_acc, aggregated_statistics[n_repeat]['val_acc'])
        acc = np.append(acc, aggregated_statistics[n_repeat]['acc'])
        val_loss = np.append(val_loss, aggregated_statistics[n_repeat]['val_loss'])
        loss = np.append(loss, aggregated_statistics[n_repeat]['loss'])
        val_loss_auto=np.append(val_loss, aggregated_statistics[n_repeat]['val_loss_auto'])
        f1 = np.append(f1, aggregated_statistics[n_repeat]['f1'])
        precision = np.append(precision,aggregated_statistics[n_repeat]['precision'])
        recall = np.append(recall, aggregated_statistics[n_repeat]['recall'])
        auc = np.append(auc, aggregated_statistics[n_repeat]['auc'])
        fpr=aggregated_statistics[n_repeat]['fpr']
        tpr=aggregated_statistics[n_repeat]['tpr']
        tprs.append(interp(mean_fpr, fpr, tpr))    
        #
        if n_repeat==0:
            all_y_test = aggregated_statistics[n_repeat]['y_test']
            all_y_pred = aggregated_statistics[n_repeat]['y_pred']
        else:
            all_y_test = np.append(all_y_test,aggregated_statistics[n_repeat]['y_test'], axis=0)
            all_y_pred = np.append(all_y_pred,aggregated_statistics[n_repeat]['y_pred'], axis=0)

    #######################################    
    # print all statistics to an outfile  #
    #######################################

    print('Saving summary statistics to file %s%s' %(analysis_directory,outFile))

    outFN=open(os.path.expanduser('%s%s' %(analysis_directory,outFile)), 'a')       
    outFN.write(outFile_header)
    outFN.write('val_acc\tval_acc_se\tacc\tacc_se\tval_loss\tval_loss_se\tloss\tloss_se\tf1\tf1_se\tprecision\tprecision_se\trecall\trecall_se\tauc\tauc_se\tval_loss_auto\tval_loss_auto_se\n')

    s=''
    for value in [val_acc, acc, val_loss, loss,f1, precision, recall, auc,val_loss_auto]:
        m, se = mean_confidence_interval(value, n_splits, confidence=0.95)
        s+= str(m) + '\t' + str(se) + '\t' 

    outFN.write(summary_string + '\t' + s +'\n')


def mean_confidence_interval(data, n, confidence=0.95):
    # note that n typically = n_splits as this is the correct num of df
    a = 1.0 * np.array(data)
    m, se = np.mean(a), scipy.stats.sem(a)
    z = calculate_z_value(confidence, n)
    h = se * z
    # to get the CI: m-h, m+h
    return m, h


def calculate_z_value(confidence, n):
    z = scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return z


def compute_shap_values_deeplearning(input_dim, model, x_test):
    background = np.zeros((1, input_dim))
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(x_test)
    # note that there is likely a lot of meaning in the negative vs positive values of shap that we need to understand better. 
    shap_values_summed = np.sum(np.absolute(shap_values[0]), axis=0)

    return shap_values, shap_values_summed


def aggregate_shap(aggregated_statistics, rskf):
    shap_values_summed=np.zeros_like(aggregated_statistics[0]['shap_values_summed'])
    for n_repeat in range(0,len(rskf[0])):
        shap_values_summed += aggregated_statistics[n_repeat]['shap_values_summed']
    shap_values_summed /= len(rskf[0])

    # TODO: 
    # print shap values to an outfile
    # make a barplot with top informative features
    # plot the shap values for top features, colored by feature importance. 


def format_input_parameters_printing(data_set, norm_input, encoding_dim, encoded_activation,input_dropout_pct,dropout_pct,num_epochs,batch_size,n_splits,n_repeats,compute_informative_features,plot_iteration):

    # for saving results later: summarize the input options into a single str:
    summary_string='\t'.join( (data_set, str(norm_input), str(encoding_dim), encoded_activation, str(input_dropout_pct), str(dropout_pct), str(num_epochs), str(batch_size), str(n_splits), str(n_repeats)) ) 

    # for labeling plots:
    plotting_string=plotting_utils_AEB.format_plotting_string(data_set, norm_input, encoding_dim, encoded_activation, input_dropout_pct, dropout_pct, num_epochs, batch_size, n_splits, n_repeats)

    # print the parameters being tested to stdout just for record keeping 
    print('Parameters being tested:')
    print(data_set)
    print('Normalize input? ' + str(norm_input))
    print('Encoding dim: ' + str(encoding_dim))
    print('Encoded activation: ' + encoded_activation)
    print('Input dropout percent: ' + str(input_dropout_pct))
    print('Dropout percent: ' + str(dropout_pct))
    print('Num epochs: ' + str(num_epochs))
    print('Batch size: ' + str(batch_size))
    print('n_splits (k-folds): ' + str(n_splits))
    print('n_repeats (iterations): ' + str(n_repeats))
    print('Compute infromative features with Shap? ' + str(compute_informative_features))
    print('Plots for each iteration? ' + str(plot_iteration) + '\n')


    return summary_string, plotting_string


def format_input_parameters_printing_2layers(data_set, kmer_size, norm_input, encoding_dim_1, encoding_dim_2, encoded_activation,input_dropout_pct,dropout_pct,num_epochs,batch_size,n_splits,n_repeats,compute_informative_features,plot_iteration):

    # for saving results later: summarize the input options into a single str:
    summary_string='\t'.join( (data_set, str(kmer_size), str(norm_input), str(encoding_dim_1),str(encoding_dim_2), encoded_activation, str(input_dropout_pct), str(dropout_pct), str(num_epochs), str(batch_size), str(n_splits), str(n_repeats)) ) 

    # for labeling plots:
    plotting_string=plotting_utils.format_plotting_string_2layers(data_set, kmer_size, norm_input, encoding_dim_1, encoding_dim_2, encoded_activation, input_dropout_pct, dropout_pct, num_epochs, batch_size, n_splits, n_repeats)

    # print the parameters being tested to stdout just for record keeping 
    print('Parameters being tested:')
    print(data_set)
    print(str(kmer_size))
    print('Normalize input? ' + str(norm_input))
    print('Encoding dim_1: ' + str(encoding_dim_1))
    print('Encoding dim_2: ' + str(encoding_dim_2))
    print('Encoded activation: ' + encoded_activation)
    print('Input dropout percent: ' + str(input_dropout_pct))
    print('Dropout percent: ' + str(dropout_pct))
    print('Num epochs: ' + str(num_epochs))
    print('Batch size: ' + str(batch_size))
    print('n_splits (k-folds): ' + str(n_splits))
    print('n_repeats (iterations): ' + str(n_repeats))
    print('Compute infromative features with Shap? ' + str(compute_informative_features))
    print('Plots for each iteration? ' + str(plot_iteration) + '\n')


    return summary_string, plotting_string



def format_input_parameters_printing_3layers(data_set, kmer_size, norm_input, encoding_dim_1, encoding_dim_2, encoding_dim_3, encoded_activation,input_dropout_pct,dropout_pct,num_epochs,batch_size,n_splits,n_repeats,compute_informative_features,plot_iteration):

    # for saving results later: summarize the input options into a single str:
    summary_string='\t'.join( (data_set, str(kmer_size), str(norm_input), str(encoding_dim_1),str(encoding_dim_2), str(encoding_dim_3), encoded_activation, str(input_dropout_pct), str(dropout_pct), str(num_epochs), str(batch_size), str(n_splits), str(n_repeats)) ) 

    # for labeling plots:
    plotting_string=plotting_utils.format_plotting_string_3layers(data_set, kmer_size, norm_input, encoding_dim_1, encoding_dim_2, encoding_dim_3, encoded_activation, input_dropout_pct, dropout_pct, num_epochs, batch_size, n_splits, n_repeats)

    # print the parameters being tested to stdout just for record keeping 
    print('Parameters being tested:')
    print(data_set)
    print(str(kmer_size))
    print('Normalize input? ' + str(norm_input))
    print('Encoding dim_1: ' + str(encoding_dim_1))
    print('Encoding dim_2: ' + str(encoding_dim_2))
    print('Encoding dim_3: ' + str(encoding_dim_3))
    print('Encoded activation: ' + encoded_activation)
    print('Input dropout percent: ' + str(input_dropout_pct))
    print('Dropout percent: ' + str(dropout_pct))
    print('Num epochs: ' + str(num_epochs))
    print('Batch size: ' + str(batch_size))
    print('n_splits (k-folds): ' + str(n_splits))
    print('n_repeats (iterations): ' + str(n_repeats))
    print('Compute infromative features with Shap? ' + str(compute_informative_features))
    print('Plots for each iteration? ' + str(plot_iteration) + '\n')


    return summary_string, plotting_string


def bootstrap_data(data_normalized, kmer_cnts, num_replicates, num_kmers):
    
    #store bootstrapped data in a dictionary:
    bootstrapped_data={}

    for i in range(0,len(kmer_cnts)):
        print(i)
        bootstrapped_data[i]=[]
        for replicate in range(0,num_replicates):
            sample = np.random.choice(len(data_normalized[0]), num_kmers, p=data_normalized[i])
            unique, counts =np.unique(sample, return_counts=True)
            bootstrapped_array=np.zeros(len(data_normalized[0]))
            bootstrapped_array[unique] = counts
            bootstrapped_data[i].append(bootstrapped_array)

    return bootstrapped_data