import matplotlib
matplotlib.use('Agg') # this suppresses the console for plotting
import pylab
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import numpy as np
import os, sys
import re
from itertools import product, cycle
from sklearn.metrics import auc

graph_dir = os.environ['HOME'] + '/deep_learning_microbiome/analysis/kmers/'

# controls transparency of the std band around the ROCs as in Pasolli
plot_alpha = 0.2

plot_text_size = 10
plot_title_size = plot_text_size + 2

source_ind, class_name, label_ind, classes, n_classes, class_to_ind, colors, markers = None, None, None, None, None, None, None, None

iter_fold_results = {}

config_aggr_results = {}
config_ae_results = {}

def process_pickles(filenames):
    roc_aucs = []
    ae_results = None
    for filename in filenames:
        with open(filename, "rb") as f:
            print("Loading pickle from " + filename)
            config = re.search(r'^aggr_results(.+)\.pickle', filename).groups()[0]
            dump_dict = pickle.load(f)
            dataset_info = dump_dict['dataset_info']
#            has_ae_results = 'ae_results' in dump_dict
#            has_ae_results = False
#            ae_results = None
#            if has_ae_results:
#                ae_results = dump_dict['ae_results']
#                config_ae_results[config] = ae_results

            global source_ind, class_name, label_ind, classes, n_classes, class_to_ind, colors, markers
            source_ind, class_name, label_ind, classes, n_classes, class_to_ind, colors, markers = dataset_info
            aggr_results, iter_results = dump_dict['results']
            val_acc, acc, val_loss, loss, conf_mat, fpr, tpr, roc_auc, accs, std_down, std_up, perf_means, perf_stds = aggr_results
            config_aggr_results[config] = aggr_results
            is_shuffled = re.search(r'_SL:1', config)
            roc_aucs.append([class_name, is_shuffled, config, fpr, tpr, roc_auc, accs, std_down, std_up])
            
            # plot_confusion_matrix(conf_mat, config)
            # plot_loss_vs_epochs(loss, val_loss, config)
            # plot_acc_vs_epochs(acc, val_acc, config)
            # plot_roc_aucs(fpr, tpr, roc_auc, accs, std_down, std_up, config)
            
            print("Accuracy: {}({}), F1: {}({}), precision: {}({}), recall: {}({}), avg fold AUC:{}({}), avg MSE:{}({}), for config:{}".format(perf_means[0], perf_stds[0], perf_means[1], perf_stds[1],
                                                                                                                perf_means[2], perf_stds[2], perf_means[3], perf_stds[3],
                                                                                                                perf_means[4] if n_classes == 2 else perf_means[5],
                                                                                                                               perf_stds[4] if n_classes == 2 else perf_stds[5],
                                                                                                                                               ae_results[2][1] if ae_results else 'none',
                                                                                                                                               ae_results[3][1] if ae_results else 'none',
                                                                                                                                               config))

            # if the supervised training epochs are too small, we don't plot 2D codes because it's most likely
            # only for autoencoder experiments.
            if len(val_acc) > 50:
                is_binary = (n_classes == 2)
                plot_2d_codes_for_avg_fold(perf_means[0], perf_means[4] if is_binary else perf_means[5], is_binary, iter_results, config, title='')

            # changed at 1354 on 081418
            plot_2d_codes_for_avg_fold(perf_means[0], perf_means[4] if is_binary else perf_means[5], is_binary, iter_results, config, title='')

            iter_fold_results[config] = iter_results

            for i in [0]: #range(len(iter_results)):
                for j in range(len(iter_results[0])):

                    codes, info = iter_results[i][j][1]
                    #print(filename + " with config: " + config + ", content: " + str(iter_results))
                    #print(" Codes: " + str(codes) + ", info: " + str(info))

                    config_iter_fold = config + '_IT:' + str(i) + '_KF:' + str(j)
                    #plot_2d_codes(codes, config_iter_fold, info)

                    # plot_confusion_matrix(iter_results[i][j][2], config_iter_fold)
                    # plot_loss_vs_epochs(iter_results[i][j][0]['loss'], iter_results[i][j][0]['val_loss'], 
                    #                     config_iter_fold)

                    # plot_acc_vs_epochs(iter_results[i][j][0]['acc'], iter_results[i][j][0]['val_acc'], 
                    #                    config_iter_fold)


                    # plot_loss_vs_epochs(iter_results[i][j][7][0]['loss'], iter_results[i][j][7][0]['val_loss'], 
                    #                     config_iter_fold, name='picked_ae_loss_vs_epochs')

    plot_all_roc_aucs(roc_aucs)
    for config in config_aggr_results:
        s_config = re.sub(r'SL:0', 'SL:1', config)
        if s_config == config:
            continue
        if not s_config in config_aggr_results:
            continue
        n_config = re.sub(r'SL:0', 'SL:All', config)
        res1 = config_aggr_results[config]
        # if the supervised training epochs are too small, we don't plot because it's most likely
        # only for autoencoder experiments.
        if len(res1[3]) < 50:
            continue
        res2 = config_aggr_results[s_config]
        plot_loss_vs_epoch_pair([res1[3], res1[2]], [res2[3], res2[2]], n_config, title='')#, xlabel='', ylabel='')
        plot_acc_vs_epoch_pair([res1[1], res1[0]], [res2[1], res2[0]], n_config, title='')#, xlabel='', ylabel='')

    for config in config_ae_results:
        s_config = re.sub(r'SA:0', 'SA:1', config)
        if s_config == config:
            continue
        if not s_config in config_ae_results:
            continue
        # if the supervised training epochs are big, we don't plot because this is likely not
        # for autoencoder experiment
        if len(config_aggr_results[config][3]) > 50:
            continue

        n_config = re.sub(r'SA:0', 'SA:All', config)
        res1 = config_ae_results[config]
        res2 = config_ae_results[s_config]
        plot_loss_vs_epoch_pair([res1[0], res1[1]], [res2[0], res2[1]], n_config, title='')#xlabel='', ylabel='')

    compute_t_stats()
    #if has_ae_results:
    #    compute_ae_t_stats()

def shorten_label(label):
    d = { 'North America': 'NA', 'United States': 'US', 'MetaHIT': 'MH' }
    if label in d:
        return d[label]
    else:
        return label

# Plot the 2D codes in the layer right before the final classification layer
def plot_2d_codes(codes, config, info, y_test_pred=None, name='pickled_two_codes', title='2D Codes Before Classfication Layer', 
                  xlabel='Code 1', ylabel='Code 2'):
    fig = pylab.figure()
    pylab.title(title, size=plot_title_size)
    pylab.xlabel(xlabel, size=plot_text_size)
    pylab.ylabel(ylabel, size=plot_text_size)
    pylab.tick_params(axis='both', labelsize=plot_text_size)

    handles = {}
    for i in range(len(codes)):
        src = info[i][source_ind]
        ind = class_to_ind[info[i][label_ind]]
        label = None
        if y_test_pred is not None:
            pred_ind = np.argmax(y_test_pred[i])
            if pred_ind != ind:
                label = classes[pred_ind][0]

        h = pylab.scatter([codes[i][0]], [codes[i][1]], marker=markers[src], color=colors[ind], facecolor='none')
        if label:
            point = (codes[i][0], codes[i][1])
            pylab.annotate(label, xy=point, xytext=point, color='black', size=9)

        handles[('' if n_classes == 2 else shorten_label(src)) + ' - ' + shorten_label(info[i][label_ind])] = h
    keys = [ k for k in handles.keys() ]
    keys.sort()
    pylab.legend([handles[k] for k in keys], keys, prop={'size': plot_text_size})
    #pylab.gca().set_position((.1, .7, 0.8, .8))
    fig.set_size_inches(4, 4)
    save_close_fig(fig, name, config)

# Plot the 2D codes for the fold that has accuracy and AUC closest to their respective averages over all runs of the model
def plot_2d_codes_for_avg_fold(acc, auc, is_binary, iter_results, config, name='pickled_two_codes_avg_fold', title='2D Codes of an Average Fold', 
                               xlabel='Code 1', ylabel='Code 2'):
    codes, info, preds = None, None, None
    min_dist = 999
    if np.isnan(auc):
        # supervised null may have folds that contain no positive labels, which yield NaN AUCs - 
        # we calculate the avg of those that are not NaNs here
        aucs = []
        auc1s = []
        for i in range(len(iter_results)):
            for j in range(len(iter_results[i])):
                perf_stats = iter_results[i][j][6]
                auc2 = (perf_stats[4] if is_binary else perf_stats[5])
                if not np.isnan(auc2):
                    aucs.append(auc2)
                if not np.isnan(perf_stats[4]):
                    auc1s.append(perf_stats[4])
        auc = np.mean(aucs)
        if np.isnan(auc):
            auc = np.mean(auc1s)
        
    for i in range(len(iter_results)):
        for j in range(len(iter_results[i])):
            perf_stats = iter_results[i][j][6]
            auc2 = (perf_stats[4] if is_binary else perf_stats[5])
            #print("acc:{}, acc2:{}, auc:{}, auc2:{}".format(acc, perf_stats[0], auc, auc2))
            d = np.max([np.abs(acc - perf_stats[0]), np.abs(auc - auc2)])
            if d < min_dist:
                codes, info = iter_results[i][j][1]
                min_dist = d
                preds = iter_results[i][j][5][1]

    # remove the iteration and fold index fromm the config because this is the model wide average plot
    config = re.sub(r'_IT:\d+', '', config)
    config = re.sub(r'_KF:\d+', '', config)
    plot_2d_codes(codes, config, info, y_test_pred=preds, name=name, title=title)

# plot loss vs epochs
def plot_loss_vs_epochs(loss, val_loss, config, name='pickled_loss_vs_epochs', title='Loss vs Epochs', 
                        xlabel='Epoch', ylabel='Loss'):
    fig = pylab.figure()
    pylab.plot(loss)
    pylab.plot(val_loss)
    pylab.legend(['train ' + str(loss[-1]), 'test ' + str(val_loss[-1])], prop={'size': plot_text_size})
    pylab.title(title, size=plot_title_size)
    pylab.xlabel(xlabel, size=plot_text_size)
    pylab.ylabel(ylabel, size=plot_text_size)
    pylab.gca().set_position((.1, .7, .8, .6))
    save_close_fig(fig, name, config)

# plot loss vs epochs
def plot_loss_vs_epoch_pair(model_losses, null_losses, config, name='pickled_loss_vs_epoch_pair', title='Loss vs Epochs', 
                            xlabel='Epoch', ylabel='Loss', h_size=2, v_size=2, lw=1):
    fig = pylab.figure()
    legs = []
    colors = cycle(['darkblue', 'darkorange', 'r', 'g'])
    for (loss, val_loss), label in [ (model_losses, ''), (null_losses, ' - shuffled') ]:
        l = pylab.plot(loss, ls='--', lw=lw, color=next(colors), label='Training' + label)
        legs.append(l)
        l = pylab.plot(val_loss, ls='-', lw=lw, color=next(colors), label='Test' + label)
        legs.append(l)

    #pylab.legend(prop={'size': plot_text_size}, loc='upper right')
    pylab.title(title, size=plot_title_size)
    pylab.xlabel(xlabel, size=plot_text_size)
    pylab.ylabel(ylabel, size=plot_text_size)
    #pylab.gca().set_position((.1, .7, .8, .8))
    fig.set_size_inches(h_size, v_size)
    save_close_fig(fig, name, config)

# plot accuracy vs epochs
def plot_acc_vs_epochs(acc, val_acc, config, name='pickled_accu_vs_epochs', title='Accuracy vs Epochs',
                       xlabel='Epoch', ylabel='Accuracy'):
    fig = pylab.figure()
    pylab.plot(acc)
    pylab.plot(val_acc)
    pylab.legend(['train ' + str(acc[-1]), 'test ' + str(val_acc[-1])], loc='lower right', prop={'size': plot_text_size})
    pylab.title(title, size=plot_title_size)
    pylab.xlabel(xlabel, size=plot_text_size)
    pylab.ylabel(ylabel, size=plot_text_size)
    pylab.tick_params(axis='both', labelsize=plot_text_size)
    pylab.gca().set_position((.1, .7, .8, .6))
    save_close_fig(fig, name, config)

def plot_acc_vs_epoch_pair(model_accs, null_accs, config, name='pickled_accu_vs_epoch_pair', title='Accuracy vs Epochs', 
                            xlabel='Epoch', ylabel='Accuracy', h_size=2, v_size=2, lw=1):
    fig = pylab.figure()
    legs = []
    colors = cycle(['darkblue', 'darkorange', 'r', 'g'])
    for (acc, val_acc), ls, label in [ (model_accs, '-', ''), (null_accs, '--', ' - shuffled') ]:
        l = pylab.plot(acc, ls='--', lw=lw, color=next(colors), label='Training' + label)
        legs.append(l)
        l = pylab.plot(val_acc, ls='-', lw=lw, color=next(colors), label='Test' + label)
        legs.append(l)

    #pylab.legend(prop={'size': plot_text_size}, loc='upper right')
    pylab.title(title, size=plot_title_size)
    pylab.xlabel(xlabel, size=plot_text_size)
    pylab.ylabel(ylabel, size=plot_text_size)
    #pylab.gca().set_position((.1, .7, .8, .8))
    fig.set_size_inches(h_size, v_size)
    save_close_fig(fig, name, config)

def plot_confusion_matrix(cm, config, cmap=pylab.cm.Reds):
    """
    This function plots the confusion matrix.
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    fig, ax = pylab.subplots(1, 2)
    for sub_plt, conf_mat, title, fmt in zip(ax, [cm, cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]], ['Unnormalized Confusion Matrix', 'Normalized Confusion Matrix'], ['d', '.2f']):
        #print(conf_mat)
        #print(title)

        im = sub_plt.imshow(conf_mat, interpolation='nearest', cmap=cmap)
        sub_plt.set_title(title, size=plot_title_size)
        divider = make_axes_locatable(sub_plt)
        cax1 = divider.append_axes("right", size="5%", pad=0.1)
        #fig.colorbar(im, ax=sub_plt)
        fig.colorbar(im, cax=cax1)
        tick_marks = np.arange(len(cm))
        sub_plt.set_xticks(tick_marks)
        sub_plt.set_yticks(tick_marks)
        sub_plt.set_xticklabels(classes)
        sub_plt.set_yticklabels(classes)
        sub_plt.tick_params(labelsize=plot_text_size, axis='both')

        thresh = 0.8*conf_mat.max()
        for i, j in product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
            sub_plt.text(j, i, format(conf_mat[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if conf_mat[i, j] > thresh else "black", size=plot_title_size)
        sub_plt.set_ylabel('True Label', size=plot_text_size)
        sub_plt.set_xlabel('Predicted Label', size=plot_text_size)
    pylab.tight_layout()
    pylab.gca().set_position((.1, 10, 0.8, .8))
    save_close_fig(fig, 'pickled_confusion_mat', config)

# Plot the ROCs
def plot_roc_aucs(fpr, tpr, roc_auc, accs, std_down, std_up, config, name='pickled_roc_auc', title='ROC Curves with AUCs/ACCs', 
                  xlabel='False Positive Rate', ylabel='True Positive Rate'):
    fig = pylab.figure()
    lw = 2

    if n_classes > 2:
        pylab.plot(fpr["micro"], tpr["micro"],
                   label='micro-average ROC (AUC = {0:0.4f})'
                   ''.format(roc_auc["micro"]),
                   color='deeppink', linestyle=':', linewidth=4)
        pylab.fill_between(fpr['micro'], std_down['micro'], std_up['micro'], color='deeppink', lw=0, alpha=plot_alpha)

        pylab.plot(fpr["macro"], tpr["macro"],
                   label='macro-average ROC (AUC = {0:0.4f})'
                   ''.format(roc_auc["macro"]),
                   color='navy', linestyle=':', linewidth=4)
        pylab.fill_between(fpr['macro'], std_down['macro'], std_up['macro'], color='navy', lw=0, alpha=plot_alpha)

        roc_colors = cycle(['green', 'red', 'purple', 'darkorange'])
        for i, color in zip(range(n_classes), roc_colors):
            pylab.plot(fpr[i], tpr[i], color=color, lw=lw,
                       label='ROC of {0} (AUC = {1:0.4f}, acc={2:0.4f})'
                       ''.format(classes[i], roc_auc[i], accs[i]))
            pylab.fill_between(fpr[i], std_down[i], std_up[i], color=color, lw=0, alpha=plot_alpha)
    else:
        i = 1
        pylab.plot(fpr[i], tpr[i], color='darkorange', lw=lw,
                   label='ROC of {0} (AUC = {1:0.4f}, acc={2:0.4f})'
                   ''.format(classes[i], roc_auc[i], accs[i]))
        pylab.fill_between(fpr[i], std_down[i], std_up[i], color='darkorange', lw=0, alpha=plot_alpha)

    pylab.plot([0, 1], [0, 1], 'k--', lw=lw)
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.05])
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)
    pylab.title(title, size=plot_title_size)
    pylab.legend(loc="lower right", prop={'size': plot_text_size})
    pylab.gca().set_position((.1, .7, .8, .8))
    save_close_fig(fig, name, config)

# Plot all the ROCs from all of the input files
def plot_all_roc_aucs(roc_auc_info_list, config='', name='pickled_all_roc_auc', title='ROC Curves', 
                  xlabel='False Positive Rate', ylabel='True Positive Rate'):
    fig = pylab.figure()
    lw = 2

    #roc_colors = cycle(['green', 'red', 'blue', 'purple', 'darkorange', 'darkgreen'])
    roc_colors = cycle(['b','g','r','c','m','orange'])

    color = None
    config_color = {}
    for cls_name, shuffled, config, fpr, tpr, roc_auc, accs, std_down, std_up in roc_auc_info_list:
        config_key = re.sub(r'_SL:1', '_SL:0', config)
        if not config_key in config_color:
            color = next(roc_colors)
            config_color[config_key] = color
            #print("setting color for " + config_key + " to " + color)
        else:
            color = config_color[config_key]
            #print("using color for " + config + ": " + color)

        i = None
        if 2 in fpr:
            # multiclass classification - use the macro average
            i = 'macro'
            pylab.plot(fpr[i], tpr[i],
                       label='ROC {0} (AUC = {1:0.4f})'
                       ''.format(cls_name if not shuffled else cls_name + ' shuffled', roc_auc[i]),
                       color=color, linestyle='--' if shuffled else '-',  linewidth=lw)
            pylab.fill_between(fpr[i], std_down[i], std_up[i], color=color, lw=0, alpha=plot_alpha)
        else:
            i = 1
            pylab.plot(fpr[i], tpr[i], color=color, lw=lw, linestyle='--' if shuffled else '-', 
                       label='ROC of {0} (AUC = {1:0.4f}, acc={2:0.4f})'
                       ''.format(cls_name if not shuffled else cls_name + ' shuffled', roc_auc[i], accs[i]))
            pylab.fill_between(fpr[i], std_down[i], std_up[i], color=color, lw=0, alpha=plot_alpha)

        overall_auc = auc(fpr[i], tpr[i])
        lower_auc = auc(fpr[i], std_down[i])
        upper_auc = auc(fpr[i], std_up[i])
        print("Overall AUC is: " + str(overall_auc) + ", 95% CI: [" + str(lower_auc) + ", " + str(upper_auc) + "] for " +  config)

    #pylab.plot([0, 1], [0, 1], 'k--', lw=lw)
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.0])
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)
    pylab.title(title, size=plot_title_size)
    #pylab.legend(loc="lower right", prop={'size': plot_text_size})

    real_models = list(filter(lambda r: not r[1], roc_auc_info_list))
    roc_colors = ['b','g','r','c','m','orange']
    #leg_col = [pylab.Rectangle((0, 0), 1, 1, fc=s, linewidth=0) for s in roc_colors[:len(real_models)]] + [pylab.Line2D([0,1], [0,1], c='k', ls='-', lw=2)] + [pylab.Line2D([0,1], [0,1], c='k', ls='--', lw=2)]
    leg_col = [pylab.Rectangle((0, 0), 0.25, 0.25, fc=s, linewidth=0) for s in roc_colors[:len(real_models)]] + [pylab.Line2D([0,0.25], [0,0.25], c='k', ls='-', lw=2)] + [pylab.Line2D([0,0.25], [0,0.25], c='k', ls='--', lw=2)]
    leg_l = [r[0] for r in real_models] + ['True labels'] + ['Shuffled labels']
    #leg = pylab.legend(leg_col, leg_l, prop={'size':plot_title_size}, loc='center left', bbox_to_anchor=(1.02,0.5), numpoints=1)

    pylab.gca().set_position((0.0, .7, .8, .8))

    leg = pylab.legend(leg_col, leg_l, prop={'size':9}, loc='lower center', bbox_to_anchor=(0.5, -.4), ncol=3)
    leg.get_frame().set_alpha(0)

    fig.set_size_inches(4, 4)
    save_close_fig(fig, name, config='')

def compute_t_stats():
    for config in iter_fold_results:
        if re.search(r'_SL:1', config):
            continue
        iter_results = iter_fold_results[config]
        shuffled_key = re.sub(r'_SL:0', '_SL:1', config)
        if not shuffled_key in iter_fold_results:
            continue
        shuffled_iter_results = iter_fold_results[shuffled_key]

        total_diff = 0.0
        iter_stds = []
        n_iters = len(iter_results)
        for i in range(n_iters):
            fold_diffs = []
            for j in range(len(iter_results[i])):
                fpr, tpr, roc_auc = iter_results[i][j][3]
                _, _, s_roc_auc = shuffled_iter_results[i][j][3]
                k = 'macro' if 2 in roc_auc else 1
                auc1 = roc_auc[k]
                auc2 = s_roc_auc[k]
                if np.isnan(auc1):
                    if 2 in roc_auc:
                        aucs = []
                        for l in range(len(roc_auc.keys())-2):
                            if not np.isnan(roc_auc[l]):
                                aucs.append(roc_auc[l])
                        if len(aucs) > 0:
                            auc1 = np.mean(aucs)
                    else:
                        auc1 = roc_auc[0]
                if np.isnan(auc2):
                    if 2 in s_roc_auc:
                        aucs = []
                        for l in range(len(s_roc_auc.keys())-2):
                            if not np.isnan(s_roc_auc[l]):
                                aucs.append(s_roc_auc[l])
                        if len(aucs) > 0:
                            auc2 = np.mean(aucs)
                    else:
                        auc2 = s_roc_auc[0]
                    
                #print("auc1: " + str(auc1), ", auc2: " + str(auc2))
                d = auc1 - auc2
                if not np.isnan(d):
                    fold_diffs.append(d)
                    total_diff += d
            iter_stds.append(np.std(fold_diffs))
        
        #t_stats = total_diff/(n_iters * len(iter_results[0]) *np.mean(iter_stds))
        t_stats = total_diff/(n_iters * np.sqrt(len(iter_results[0])) * np.mean(iter_stds))

        print('t statistic for ' + config + ' is: ' + str(t_stats))

def compute_ae_t_stats():
    for config in iter_fold_results:
        if re.search(r'_SA:1', config):
            continue
        iter_results = iter_fold_results[config]
        shuffled_key = re.sub(r'_SA:0', '_SA:1', config)
        if not shuffled_key in iter_fold_results:
            continue
        shuffled_iter_results = iter_fold_results[shuffled_key]

        total_diff = 0.0
        iter_stds = []
        n_iters = len(iter_results)
        for i in range(n_iters):
            fold_diffs = []
            for j in range(len(iter_results[i])):
                _, _, mse = iter_results[i][j][7]
                _, _, s_mse = shuffled_iter_results[i][j][7]
                d = mse - s_mse
                fold_diffs.append(d)
                total_diff += d
            iter_stds.append(np.std(fold_diffs))
        
        #t_stats = total_diff/(n_iters * len(iter_results[0]) *np.mean(iter_stds))
        t_stats = total_diff/(n_iters * np.sqrt(len(iter_results[0])) * np.mean(iter_stds))

        print('t statistic for ' + config + ' is: ' + str(t_stats))
        
        

# Add figure texts to plots that describe configs of the experiment that produced the plot
def save_close_fig(fig, name, config):
    filename = graph_dir + '/' + name + config + '.png'
    pylab.savefig(filename , bbox_inches='tight', dpi=550)
    pylab.close(fig)

#process_pickles('aggr_results_DS\:SingleDiseaseMetaHIT_KS\:7_LS\:8192\:8_EA\:sigmoid_CA\:sigmoid_DA\:NA_OA\:NA_LF\:mean_squared_error_PC\:0_SEP\:400_NO\:L1_NI\:1_BS\:16_DP\:0_IDP\:0_AU\:0_NR\:0_UK\:10_ITS\:20_SL\:0_SA\:0_MN\:0_AD\:_AL\:_AA\:None.pickle')

if __name__ == '__main__':
    process_pickles(sys.argv[1:])
