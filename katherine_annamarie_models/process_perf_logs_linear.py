
import sys
import re
import numpy as np

config_score = {}

if __name__ == "__main__":
    config_score = {}

    for line in sys.stdin:
        s = re.search(r'fold-perf-metrics for (.+)', line)
        if s:
            key = s.groups()[0]
            fields = line.split('\t')
            acc = float(fields[0])
            f1 = float(fields[1])
            precision = float(fields[2])
            recall = float(fields[3])
            auc1 = float(fields[4])
            auc = float(fields[5])

            stats = [acc, f1, precision, recall, auc1, auc]
            has_nan = False
            for n in stats:
                if np.isnan(n):
                    has_nan = True
                    break
            if has_nan:
                continue
                    
            try:
                config_score[key].append(stats)
            except:
                config_score[key] = [stats]
                
    for k, vs in config_score.items():
        vs = np.array(vs)
        means = np.mean(vs, axis=0)
        stds = np.std(vs, axis=0)
        print("{}(auc-{})\t{}(acc-{})\t{}(auc1-{})\t{}(f1-{})\t{}(precision-{})\t{}(recall-{})\t{}\t(after {} test folds)".format(means[5], stds[5], means[0], stds[0], means[4], stds[4], means[1], stds[1],  means[2], stds[2],   means[3], stds[3],k, str(len(vs))))



    '''
    for line in sys.stdin:
        s = re.search(r'fold-overall-perf-metrics for (.+)', line)
        if s:
            key = s.groups()[0]
            fields = line.split(")")
            acc, acc_std = fields[0].strip().split("(")
            f1, f1_std = fields[1].strip().split("(")
            precision, precision_std = fields[2].strip().split("(")
            recall, recall_std = fields[3].strip().split("(")
            last_fields = fields[4].strip().split("\t")
            auc1, auc_macro = last_fields[0], last_fields[1]

            stats = [float(acc), float(acc_std), float(f1), float(f1_std), float(precision), float(precision_std),
                     float(recall), float(recall_std), float(auc1), float(auc_macro), last_fields[2]]
            
            print(("{}({})(acc)\t{}({})(f1)\t{}({})(precision)\t{}({})(recall)\t{}(auc1)\t{}(auc-macro)\t {}").format(stats[0], stats[1], stats[2], stats[3], stats[4], stats[5], stats[6], stats[7], stats[8], stats[9], stats[10]))
    '''
