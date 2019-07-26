The grid search for the best linear models is done by running all_linear_models_katherine.py, and the stats and plots for the best models after grid search can be obtained through running linear_stats_plots.py.


###########
GRID SEARCH
###########

In all_linear_models_katherine.py: 

In data_sets_to_use (line 45), comment or uncomment the datasets that you want to run

In param_dict (line 61), add or remove search parameters for the grid search


Running all_linear_models_katherine.py:

nohup [python] all_linear_models_katherine.py -m [model type] -k [kmer size] >> [output file] &

This will run the given model type on the given kmer size for the indicated dataset(s) and output the results into the output file. These results can then be used to find the best models:

cat [output files] | grep params | sort -n -r | less

This orders the models by their search accuracy and allows you to retrieve the parameters for the best models. To find the best models across kmer sizes per dataset, put the output files for all the kmer sizes after cat and search for the dataset name in less by typing /[dataset name]

#################################
PLOTS AND STATISTICAL AGGREGATION
#################################

In linear_stats_plots.py: 

Once the parameters for the best models are retrieved, they should be entered into model_param_grid (line 114) in the following format: 

For SVMs: [name]: {'DS': [datasets],  'CVT': [# k folds],'N': [# iterations],'M': "svm",'CL': [0, 1],
             'C': [value for C], 'KN': [value for kernel], 'GM': [value for gamma], 'KS': [kmer size], 'SL': [shuffle labels]}

For RFs: [name]: {'DS': [datasets],  'CVT': [# k folds],'N': [# iterations],'M': "svm",'CL': [0, 1],
		'CR': [value for criterion], 'MD': [value for max_depth], 'MF': [value for max_features], 'MS': [value for min_samples_split], 
		'NE': [value for number of estimators], 'NJ': [number of jobs], 'KS': [kmer size], 'NR': [whether to use normalization to zero mean 		and unit stdev],  'SL': [shuffle labels]}

(In many cases, the best params for your model are already in the model_param_grid, so you can make a copy of them)

Then, in dataset_model_grid (line 80), you can indicate which models you want to run for which datasets. 

Running linear_stats_plots.py: 

nohup [python] linear_stats_plots.py >> [output file] &

This will run the models in dataset_model_grid, print statistics into the output file, and generate graphs stored in deep_learning_microbiome/analysis/kmers

In order to view detailed statistics: 

cat [output files] | [python] process_perf_logs_linear.py | less

#####################################################
RETRIEVING AND PLOTTING TOP N MOST IMPORTANT FEATURES
#####################################################

In order to get the feature importance dump, run the linear_stats_plots.py with a -features option as follows: 

[python] linear_stats_plots.py -features [preferred number of features to be dumped] >> [output file]

The default number of features to be dumped is set to be -1, denoting that all non-zero features should be dumped. 0 means no feature importance dump. 

The feature importance dump is stored in a file named as follows: 

feat_imps_[config].txt

In the case of shap on SVM, a Singular Matrix Exception may cause the feature importance dump to fail. If you suspect that this has happened, search for the printout "Got exception:" in the output file, as I have caught and printed all shap-related exceptions. 

With the feature importances file, you can plot by running plot_feature_importances.py with the following line: 

[python] plot_feature_importances.py -file [feature importances file] -numfeats [number of features to be recorded in the graph] -name [name of file to store bar graph]

######################################
LOGGING THE PER-ITERATION FEATURE IMPS
######################################

In linear_stats_plots.py, I added the parameter logiterfeats, which indicates whether the features should be logged per iteration. 

Run linear_stats_plots.py with the parameter logiterfeats set to True. 

nohup [python] [file to run] -logiterfeats True >> [output file] &

#######################
RUNNING SHUFFLED MODELS
#######################

In linear_stats_plots.py, the model_param_grid now includes a key 'SL' which indicates shuffled (1) or not (0). In order to run a shuffled model, set 'SL' to 1 and put the model in dataset_model_grid. 

########
VERSIONS
########

The new key -version can be used to manually set a version for the config (for different runs of the same model, for example). 

###################################################
TO COMBINE ROC PLOTS AND CALCULATE T-STATS/P-VALUES
###################################################

Run deep_learning_merge_plots_linear.py and give the names of the pickle file names of all the models whose ROC's are to be merged should be in the list of filenames. In order to get the t-statistics and p-values for a certain model, both the real model's pickle file and the corresponding shuffled model's pickle file should be in the list. 

[python] deep_learning_merge_plots_linear.py [file names] >> [output file]

To view the t-statistics and p-values: 

less [output file]

The plots will be generated in analysis/kmers. 
