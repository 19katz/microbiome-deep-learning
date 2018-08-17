
import sys
import re
import numpy as np

config_score = {}



for line in sys.stdin:
    s = re.search(r'autoencoder-super-min-loss-max-acc for (.+)IT:\d+_KF:\d+', line)
    p = re.search(r'fold-perf-metrics for (.+)IT:\d+_KF:\d+', line)
    if s:
        key = s.groups()[0]
        fields = line.split('\t')
        acc = float(fields[5])
        loss = float(fields[7])
        ae_loss = float(fields[9])
        ae_mse = float(fields[11])
        auc1 = float(fields[12])
        auc = float(fields[13])
        vas = fields[1].split(',')
        vls = fields[3].split(',')
        stats = [auc + acc - loss, auc, acc, loss, float(vas[0]), float(vas[1]), float(vas[3]), float(vls[0]), float(vls[1]), float(vls[2]), ae_loss, ae_mse, auc1, 0, 0, 0]
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

    elif p:
        key = p.groups()[0]
        fields = line.split('\t')

        config_score[key][-1][13] = float(fields[1])
        config_score[key][-1][14] = float(fields[2])
        config_score[key][-1][15] = float(fields[3])



for k, vs in config_score.items():
    vs = np.array(vs)
    means = np.mean(vs, axis=0)
    stds = np.std(vs, axis=0)
    print("{}(auc+acc-loss)\t{}(auc-{})\t{}(acc-{})\t{}\t{}\t{}\t{}\t{}({})\t{}({})\t{}({})\t{}({})\t{}({})\t{}({})\t{}(ae-val-loss-std:{})\t{}(ae-mse-std:{})\t{}(auc1-{})\t{}(f1-{})\t{}(precision-{})\t{}(recall-{})\t{}\t(after {} test folds)".format(means[0], means[1], stds[1], means[2], stds[2], means[3], stds[1], stds[2], stds[3], means[4], stds[4], means[5], stds[5], means[6], stds[6], means[7], stds[7], means[8], stds[8], means[9], stds[9], means[10], stds[10], means[11], stds[11], means[12], stds[12], means[13], stds[13], means[14], stds[14], means[15], stds[15], k, str(len(vs))))
