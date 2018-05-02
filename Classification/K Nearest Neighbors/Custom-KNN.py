from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings

style.use('fivethirtyeight')
#euclidean_distance = sqrt((plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2)

dataset = {'k': [[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >=k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            # culc the euclidean_distance between all features and then new features we have
            ##euclidean_distance = np.sqrt(np.sum((np.array(features) - np.array(predict))**2))
            enclidean_distance = np.linalg.norm(np.array(features)-np.array(predict)) ## fast way
            ## append the group name and enclidean_distance we culc it in distances
            distances.append([enclidean_distance, group])
    #print(distances)
    votes = [i[1] for i in sorted(distances)[:k]] #append group name or key (r or k)
    print(Counter(votes).most_common(1))
    ## get first vote from all votes we have beacuse this is the best vote
    vote_result = Counter(votes).most_common(1)[0][0] #get the group name or key and return it
    
    return vote_result


result = k_nearest_neighbors(dataset, new_features, k=3)
print(result)

# firsy way
#for i in dataset:
   # for ii in dataset[i]:
        #plt.scatter(ii[0],ii[1],s=100,color=i)
       # print(i) #print key r and k 
       # print(ii[0]) #print all x value in k and r
       # print(ii[1]) #print all y value in k and r

# another way fast way
# draw the graph after culuc the group for the new_features and give it the same grop color(color=result)
[[plt.scatter(ii[0],ii[1],s=100,color=result)for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0],new_features[1])
plt.show()


## So, we have a simple dataset here with label and features we have 2 label and 3 features in one label
## we culc the enclidean distance for new feature and give the new feature same color as the group joining it
## finally draw the graph
