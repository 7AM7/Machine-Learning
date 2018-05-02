##
# i'm use https://archive.ics.uci.edu/ml/datasets.html for get data set
# use Breast Cancer Wisconsin (Original) Data Set 
# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
#
# using custom k nearest neighbors
##

from math import sqrt
import numpy as np
from collections import Counter
import warnings
import pandas as pd
import random

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >=k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            enclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([enclidean_distance, group])
            
    votes = [i[1] for i in sorted(distances)[:k]] 
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    
    return vote_result, confidence

accuracies = []
for i in range(25):
    df = pd.read_csv('../data/breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)

    # convert all data to float data type and convert it to list of list
    full_data = df.astype(float).values.tolist()
    random.shuffle(full_data)

    test_size = 0.4
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    train_data = full_data[:-int(test_size*len(full_data))] ## 80% of data
    # test the last 20% of the data
    test_data = full_data[-int(test_size*len(full_data)):] ## 20% of data

    ## get last column in the data
    ## print(i[-1])
    ## appending all data without last coulmn to the this key 
    ## in train_set dictionary
    ## so here we have two part :
    ## fist part:train_set[i[-1]] get the dictionary key
    ## second part:append(i[:-1]) appending all data without last coulmn
    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0
    for group in test_set:## key
        for data in test_set[group]:## data
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            if group == vote:
               correct += 1
            #else:  # print confidence for correct vote
               # print(confidence)
            total += 1

    #print('Accuracy:',correct/total)
    accuracies.append(correct/total)


accuracies_average = sum(accuracies) / len(accuracies)
print('Average of Accuracies: ',accuracies_average)
