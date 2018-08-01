import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pylab

graph_dir = os.environ['HOME'] + '/deep_learning_microbiome/analysis/kmers'

class PlotPar:
    bar_color = 'royalblue'
    bar_edgecolor = 'none'
    bar_stacked = 'no'
    cmap_color = 'jet'
    feature_number = 25
    bar_width = [0.48 for i in range(feature_number)]
    n_kmers_togroup = 5
    text_size = 10
    text_size_y = 10
    text_style = 'italic'
    x_label_s = 'Relative Importances (As Percentages)'
    y_label = 'Kmers'
    xticks_n = 5
    config = None
    

    def __init__(self, dataset_name, config_name):
        self.title = 'Feature Importances for ' + dataset_name
        self.config_name = config_name

    def get_graph_info(self):
        return self.title, self.config_name

def load_importances(filename):
    kmers = []
    kmer_imps = []
    count = 0
    plot_par = None
    with open(os.path.expanduser("~/deep_learning_microbiome/scripts/" + filename)) as text:
        for line in text:
            if count == 0:
                fields = line.split("\t")
                plot_par = PlotPar(fields[2], fields[3])
            else:
                if count >= plot_par.feature_number + 1:
                    break
                line = line.rstrip("\n")
                fields = line.split("\t")
                kmers.append(fields[0])
                kmer_imps.append(float(fields[1]))
            count += 1
    return kmer_imps, kmers, plot_par
    

def plot_importances(kmer_imps, kmers, plot_par):
    fig = pylab.figure()
    ax = plt.gca()
    indices = range(plot_par.feature_number)
    features = [100*kmer_imps[i] for i in range(len(kmer_imps))]

    title, config = plot_par.get_graph_info()
    
    ax.barh(indices, features, height=plot_par.bar_width)
    #ax.set_style(plot_par.text_style)
    ax.tick_params(labelsize=plot_par.text_size_y, axis='y')
    ax.set_xlabel(plot_par.x_label_s, size=plot_par.text_size)
    ax.tick_params(labelsize=plot_par.text_size)
    plt.yticks(np.arange(0, plot_par.feature_number, 1.0))
    ax.set_yticklabels(kmers)
    ax.invert_yaxis()
    plt.title(title)
    pylab.gca().set_position((.2, .8, 0.8, .8))
    add_figtexts_and_save(fig, "feature_imp_" + config, 'Feature Importances for top {} features from {}'.format(
              plot_par.feature_number, config))

def add_figtexts_and_save(fig, name, desc, x_off=0.02, y_off=0.56, step=0.04, config=None):
    filename = graph_dir + '/' + name + '.png'
    pylab.figtext(x_off, y_off, desc)
    pylab.savefig(filename , bbox_inches='tight')
    pylab.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "Plots bar chart of feature importances")
    parser.add_argument('-file', type = str, default = None, help = "File storing the feature importances")

    arg_vals = parser.parse_args()
    filename = arg_vals.file
    kmer_imps, kmers, plot_par = load_importances(filename)
    plot_importances(kmer_imps, kmers, plot_par)
