
import sys
import re
import numpy as np

config_score = {}

if __name__ == "__main__":

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
            
            print(("{}({})(acc)\t{}({})(f1)\t{}({})(precision)\t{}({})(recall)\t{}(auc1)\t{}(auc macro)\t {}").format(stats[0], stats[1], stats[2], stats[3], stats[4], stats[5], stats[6], stats[7], stats[8], stats[9], stats[10]))


