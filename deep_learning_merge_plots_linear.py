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
from scipy import stats

graph_dir = os.environ['HOME'] + '/deep_learning_microbiome/analysis/kmers'

# controls transparency of the std band around the ROCs as in Pasolli
plot_alpha = 0.1

plot_text_size = 10
plot_title_size = plot_text_size + 2

n_classes = 2

iter_fold_results = {}

config_aggr_results = {}

model_type = ''

def process_pickles(filenames):
    roc_aucs = []
    for filename in filenames:
        with open(filename, "rb") as f:
            global model_type
            if re.search(r':svm_', filename):
                model_type = 'SVM'
            elif re.search(r':rf_', filename):
                model_type = 'RF'
            print("Loading pickle from " + filename)
            config = re.search(r'aggr_results(.+)\.pickle', filename).groups()[0]
            dump_dict = pickle.load(f)
            dataset_info = dump_dict['dataset_info']
            

            source_name = dataset_info
            aggr_results, iter_results = dump_dict['results']
            conf_mat, fpr, tpr, roc_auc, accs, std_down, std_up, perf_means, perf_stds = aggr_results
            config_aggr_results[config] = aggr_results
            is_shuffled = re.search(r'_SL:1', config)
            roc_aucs.append([source_name, is_shuffled, config, fpr, tpr, roc_auc, accs, std_down, std_up])

            
            print("Accuracy: {}({}), F1: {}({}), precision: {}({}), recall: {}({}), avg fold AUC:{}({}), for config:{}".format(perf_means[0], perf_stds[0], perf_means[1], perf_stds[1],
                                                                                                                perf_means[2], perf_stds[2], perf_means[3], perf_stds[3],
                                                                                                                perf_means[4] if n_classes == 2 else perf_means[5],
                                                                                                                               perf_stds[4] if n_classes == 2 else perf_stds[5],
                                                                                                                                               config))
            iter_fold_results[config] = iter_results

    plot_all_roc_aucs(roc_aucs)

    print("COMPUTING T STATS")
    compute_t_stats()



# Plot all the ROCs from all of the input files
def plot_all_roc_aucs(roc_auc_info_list, config='', name='pickled_all_roc_auc', title='ROC Curves', 
                  xlabel='False Positive Rate', ylabel='True Positive Rate'):
    fig = pylab.figure()
    lw = 2

    color_list = ['b','g','r','c','m','orange', 'darkblue']
    #roc_colors = cycle(['green', 'red', 'blue', 'purple', 'darkorange', 'darkgreen'])
    roc_colors = cycle(color_list)

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
    pylab.title(title + " for " + model_type, size=plot_title_size)
    #pylab.legend(loc="lower right", prop={'size': plot_text_size})

    real_models = list(filter(lambda r: not r[1], roc_auc_info_list))
    # roc_colors = ['b','g','r','c','m','orange', 'teal']
    #leg_col = [pylab.Rectangle((0, 0), 1, 1, fc=s, linewidth=0) for s in roc_colors[:len(real_models)]] + [pylab.Line2D([0,1], [0,1], c='k', ls='-', lw=2)] + [pylab.Line2D([0,1], [0,1], c='k', ls='--', lw=2)]
    leg_col = [pylab.Rectangle((0, 0), 0.25, 0.25, fc=s, linewidth=0) for s in color_list[:len(real_models)]] + [pylab.Line2D([0,0.25], [0,0.25], c='k', ls='-', lw=2)] + [pylab.Line2D([0,0.25], [0,0.25], c='k', ls='--', lw=2)]
    leg_l = [r[0] for r in real_models] + ['Real labels'] + ['Shuffled labels']
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
                fpr, tpr, roc_auc = iter_results[i][j][1]
                _, _, s_roc_auc = shuffled_iter_results[i][j][1]
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
        t_stats = abs(total_diff/(n_iters * np.sqrt(len(iter_results[0])) * np.mean(iter_stds)))

        print('t statistic for ' + config + ' is: ' + str(t_stats) +" with p-value: " + str(stats.t.sf(t_stats, 9)*2))
        

# Add figure texts to plots that describe configs of the experiment that produced the plot
def save_close_fig(fig, name, config):
    filename = graph_dir + '/' + name + "_" + model_type + config + '.png'
    pylab.savefig(filename , bbox_inches='tight', dpi=550)
    pylab.close(fig)
    

if __name__ == '__main__':
    process_pickles(sys.argv[1:])
