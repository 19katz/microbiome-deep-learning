# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import pickle

#Read in kmer counts
df = pd.read_pickle('../Qin_5mers.pickle')
df.head()

#Set random seed
np.random.seed(0)

#Add in T2D column
t2d = pd.read_csv('../Qin_label.csv', header=None)
t2d_list = t2d[0].tolist()
df['T2D'] = t2d_list
df.head()

#Create a new column that for each row, generates a random number between
#0 and 1, and if that value is less than or equal to .9, then sets the 
#value of that cell as True and false otherwise. This is to randomly 
#assigning some rows to be used as the training data and 
#some as the test data.
df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.9

#View the top 5 rows
df.head()

#Create two new dataframes, one with the training rows, 
#one with the test rows
train, test = df[df['is_train']==True], df[df['is_train']==False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

# Create a list of the feature column's names
features = df.columns[:1024]
print(features)

#Needed to get T2D values into usable format for classifier
#NOTE THIS CHANGED 0'S TO 1'S!
y = pd.factorize(train['T2D'])[0]
print(y)

# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0)

#Train the Classifier to take the training features and learn how they relate
#to the training y (the species)
clf.fit(train[features], y)

# Apply the Classifier trained to the test data (which, remember, it has never seen before)
clf.predict(test[features])

# View the predicted probabilities of the first 10 observations
clf.predict_proba(test[features])[0:10]

preds = clf.predict(test[features])

# View the PREDICTED disease for the first five observations
preds[0:5]

# View the ACTUAL disease for the first five observations
test['T2D'].head()

# Create confusion matrix
pd.crosstab(test['T2D'], preds, rownames=['Actual Disease'], colnames=['Predicted Disease'])

#Maybe plot in a bar plot?
#View a list of the features and their importance scores
list(zip(train[features], clf.feature_importances_))
