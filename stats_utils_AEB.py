import matplotlib
matplotlib.use('Agg') # this suppresses the console for plotting
import matplotlib.pyplot as plt
import os
import pylab
import pandas as pd
import pickle
import numpy as np 
import scipy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.decomposition import NMF
from scipy import interp
#import shap

#import plotting_utils

def standardize_data(x_train, x_test):
    sample_mean = x_train.mean(axis=0)
    sample_std = x_train.std(axis=0)
    
    # Standardize both training and test samples with the training mean and std
    x_train = (x_train - sample_mean) / sample_std
    # test samples are standardized using only the mean and std of the training samples
    x_test = (x_test - sample_mean) / sample_std

    return x_train, x_test



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



def aggregate_statistics_across_folds(aggregated_statistics, rskf, n_splits):
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
        
        outFile=open(os.path.expanduser('~/deep_learning_microbiome/analysis/summary_statistics.txt'), 'w')
        
        outFile.write('val_acc\tval_acc_se\tacc\tacc_se\tval_loss\tval_loss_se\tloss\tloss_se\tf1\tf1_se\tprecision\tprecision_se\trecall\trecall_se\tauc\tauc_se\n')

        s=''
        for value in [val_acc, acc, val_loss, loss, f1, precision, recall, auc]:
            m, se = mean_confidence_interval(value, n_splits, confidence=0.95)
            s+= str(m) + '\t' + str(se) + '\t' 

        outFile.write(s +'\n')

        ###########################
        # precision, recall, curve #
        ###########################

        precision_graph, recall_graph, _ = precision_recall_curve(all_y_test, all_y_pred)
        plotting_utils.plot_precision_recall(precision_graph, recall_graph, f1, graph_dir=os.path.expanduser('~/deep_learning_microbiome/analysis'))

        # optional: could also add colored lines for the folds so that we can see the variance. 

        #######
        # ROC #
        #######

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        
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
        graph_dir=os.path.expanduser('~/deep_learning_microbiome/analysis')
        pylab.savefig(os.path.expanduser(graph_dir + '/roc.pdf'))



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
    
def NMF_factor(data, kmer_size, n_components=5, init = 'random', solver='mu', beta_loss='frobenius', max_iter=1000, random_state=0, title="dataset"):
    model = NMF(
        n_components = n_components,
        init = init,
        solver = solver, 
        beta_loss = beta_loss,
        max_iter = max_iter, 
        random_state = random_state
    )
    
    #NMF matrixes
    V = data.T
    W = model.fit_transform(V)
    H = model.components_
    
    #Saving to pickle
    kmer_dir = os.environ['HOME'] + '/deep_learning_microbiome/data/' + str(kmer_size) + 'mers_jf/'
    W_all = pd.DataFrame(W)
    W_all['Features'] = pd.read_csv(kmer_dir + str(kmer_size) + "mer_dictionary.gz", compression='gzip', header=None)
    meltedW = pd.melt(W_all, id_vars = "Features", var_name='Signature (i.e. Factor)', value_name='Weight')
    meltedW.to_pickle("/pollard/home/abustion/deep_learning_microbiome/data_AEB/pickled_dfs/" + title + '.pickle')
    
    #Output new data for models
    return H.T


# Uses the model to predict labels given the test features
# and compares them to the labels by calculating accuracy and error
# This is used by Lasso and Elastic Net
def evaluate(model, test_features, test_labels):
    predictions = np.array(model.predict(test_features))
    # Convert the predicted values to 0 or 1
    for r in range(len(predictions)):
        if (predictions[r] > 0.5):
            predictions[r] = 1
        else:
            predictions[r] = 0
            
    # Calculates error and accuracy
    test_labels = np.array(test_labels)
    errors = abs(predictions - test_labels)
    total_error = np.sum(errors)
    
    mape = total_error / len(test_labels)
    accuracy = 1 - mape
    return accuracy

