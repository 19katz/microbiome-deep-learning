The grid search for the best linear models is done by running all_linear_models_katherine.py, and the stats and plots for the best models after grid searh can be obtained through running linear_stats_plots.py.

In all_linear_models_katherine.py: 

In data_sets_to_use (line 44), comment or uncomment the datasets that you want to run

In param_dict (line 60), add or remove values for the grid search


Running all_linear_models_katherine.py:

nohup [python] all_linear_models_katherine.py -m [model type] -k [kmer size] >> [output file] &

This will run the given model type on the given kmer size for the indicated dataset(s) and output the results into the output file. These results can then be used to find the best models:

cat [output files] | grep Aggregated | sort -n -r | less

This orders the models by their cross-validated accuracy and allows you to retrieve the parameters for the best models. To find the best models across kmer sizes per dataset, put the output files for all the kmer sizes after cat and search for the dataset name in less by typing /[dataset name]



In linear_stats_plots.py: 

Once the parameters for the best models are retrieved, they should be entered into model_param_grid (line 89) in the following format: 

For SVMs: [name]: {'DS': [datasets],  'CVT': [# k folds],'N': [# iterations],'M': "svm",'CL': [0, 1],
             'C': [value for C], 'KN': [value for kernel], 'GM': [value for gamma], 'KS': [kmer size]}

For RFs: [name]: {'DS': [datasets],  'CVT': [# k folds],'N': [# iterations],'M': "svm",'CL': [0, 1],
		'CR': [value for criterion], 'MD': [value for max_depth], 'MF': [value for max_features], 'MS': [value for min_samples_split], 
		'NE': [value for number of estimators], 'NJ': [number of jobs], 'KS': [kmer size], 'NR': [whether to use normalization to zero mean 		and unit stdev]}

(In many cases, the best params for your model are already in the model_param_grid, so you can make a copy of them)

Then, in dataset_model_grid (line 76), you can indicate which models you want to run for which datasets. 

Running linear_stats_plots.py: 

nohup [python] linear_stats_plots.py >> [output file] &

This will run the models in dataset_model_grid, print statistics into the output file, and generate graphs stored in deep_learning_microbiome/analysis/kmers

In order to view detailed statistics: 

cat [output files] | [python] process_perf_logs_linear.py | less


