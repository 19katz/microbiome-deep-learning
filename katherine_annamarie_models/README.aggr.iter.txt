This file provides instructions for analyzing the per-iteration feature importances for deep learning, RF, and SVM. aggr_iter_feats.py gets the top n features per iteration and plots the histogram of the frequencies of these features across all iterations. 

##################################
PLOTTING THE FEATURE IMP HISTOGRAM
##################################

Before running aggr_iter_feats.py, linear_stats_plots.py or deep_learning_supervised_Katherine_jf.py should be run with logiterfeats as True. 

aggr_iter_feats.py creates and plots the histogram given a config string. It uses this config string to retrieve the per-iteration feature importance text files, so the config should include all parameters except for iteration.

The parameter maxfeats should be given to indicate the number of important features to be extracted from each iteration

[python] aggr_iter_feats.py -maxfeats [number of features] [config] >> [output file]
