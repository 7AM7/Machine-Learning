##
# i'm use https://archive.ics.uci.edu/ml/datasets.html for get data set
# use Breast Cancer Wisconsin (Original) Data Set 
# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
##

import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, neighbors, svm

df = pd.read_csv('../data/breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train,X_text,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_text,y_test) 
print("Accuracy: ",accuracy)

## unique data for test it 
example_measures = np.array([[4,2,1,1,1,2,3,2,1], [9,5,8,1,2,3,2,1,5]])
example_measures = example_measures.reshape(len(example_measures),-1)

prediction = clf.predict(example_measures)
print("Class: (2 for benign, 4 for malignant)")
print(prediction)
