##Getting the CSV into a pd df                                                                                                                                                                              
import pandas as pd
import os
import numpy as np
import os.path
from os.path import basename
import pickle

#Create empty data frame                                                                                                                                                                                    
newDF = pd.DataFrame()

#and then loop add                                                                                                                                                                                          
csv_files=os.listdir('/pollard/home/abustion/deep_learning_microbiome/data/jf_files/3mers_tech_reps/')
for csv in csv_files:
    if csv.endswith(".csv") and not ('700117172_1.fastq.gz.jf.csv'):
        csv_df = pd.read_csv(csv, sep = ' ', header = None)
        csv_df.columns = csv_df.iloc[0]
        csv_df = csv_df[1:]
        newDF = newDF.append(csv_df)

#Add in file names in a header column                                                                                                                                                                       
no_path_files = [os.path.basename(csv) for csv in csv_files if csv.endswith(".csv")]
newDF.index = no_path_files

#print(newDF)                                                                                                                                                                                               

newDF.to_pickle('/pollard/home/abustion/deep_learning_microbiome/data/pickled_dfs/3mer_tech_reps.pickle')
print(pd.read_pickle('/pollard/home/abustion/deep_learning_microbiome/data/pickled_dfs/3mer_tech_reps.pickle'))
