import argparse
import re
import matplotlib.pyplot as plt
import pylab
import os

graph_dir = os.environ['HOME'] + '/deep_learning_microbiome/analysis/kmers/iter_feat_imps'

def add_figtexts_and_save(fig, name,  x_off=0.02, y_off=0.56, step=0.04, config=None):
    filename = graph_dir + '/' + name + '_' + config + '.png'
    pylab.savefig(filename , bbox_inches='tight')
    pylab.close(fig)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description= "Program for analyzing distributions of top kmers over iterations")
    parser.add_argument('config', type = str, default = None, help = "Config string")
    parser.add_argument('-maxfeats', type = int, default = 100, help = 'Number of features for histogram')
    
    arg_vals = parser.parse_args()
    config = arg_vals.config
    max_feats = arg_vals.maxfeats

    kmer_freqs = []
    model_type = ''
    file_exists = True
    
    if re.search(r'M:svm', config):
        model_type = 'SVM'
    elif re.search(r'M:rf', config):
        model_type = 'RF'
    else:
        model_type = 'Deep Learning'

    if model_type == 'Deep Learning':
        n_iter = int(re.search(r'_ITS:(.+)_SL:', config).groups()[0])
    else:
        n_iter = int(re.search(r'_N:(.+)_M:', config).groups()[0])

    for i in range(n_iter):
        filename = graph_dir + "/" + "feat_imps_" +  config + "_IT:" + str(i) + ".txt"
        kmer_freqs.append({})
        try:
            with open (filename, "r") as f:
                for line in f:
                    line = line.strip("\n")
                    fields = line.split('\t')
                    if len(fields) == 2:
                        kmer_freqs[i][fields[0]] = float(fields[1])
        except Exception as e:
            continue
            
    
    if model_type == 'RF':
        for i in range(n_iter - 1, 0, -1):
            current_dict = kmer_freqs[i]
            prev_dict = kmer_freqs[i - 1]
            for kmer in current_dict:
                if kmer in prev_dict:
                    current_dict[kmer] = current_dict[kmer] - prev_dict[kmer]
                    if current_dict[kmer] < 1e-9:
                        current_dict[kmer] = 0

    important_kmers = []
    all_kmers = {}
    for i in range(n_iter):
        values = []
        if len(kmer_freqs) > 0:
            s = [(k,kmer_freqs[i][k]) for k in sorted(kmer_freqs[i], key=kmer_freqs[i].get, reverse=True)]
        count = 0
        for key, value in s:
            if not value > 0:
                continue
            if value < 1e-9:
                print(model_type + " " + key + " has low importance: " + str(value))
            if count < max_feats:
                if key not in all_kmers:
                    all_kmers[key] = 1
                else:
                    all_kmers[key] += 1
                values.append(value)
                count += 1
            else:
                break
        
    config += "_MFI:" + str(max_feats)
    fig = pylab.figure()
    plt.hist(list(all_kmers.values()), bins=range(1, n_iter + 1))
    plt.xlabel('Iterations')
    plt.ylabel('Number of Kmers')
    plt.xticks(range(1, n_iter + 1))
    plt.title('Histogram of top ' + str(max_feats) + ' features over iterations')
    add_figtexts_and_save(fig,name='feat_hist', config=config)
    
    
    
    
    
    
            

                
            
            
                    
                
            
    

    
