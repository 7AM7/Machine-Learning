import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):

        self.centroids = {} # self.centroids this mean center get best center point
        
        # add point from 0 to k in the dictionary ... just start  k point 
        for i in range(self.k):
            self.centroids[i] = data[i]


        for i in range(self.max_iter):
            self.classifications = {}
            ## create  classification dictionary and set it empty list form 0 to k
            for i in range(self.k):
                self.classifications[i] = []
            
            for featureset in data:
                ## get distances between featureset and center point 
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                ## get  index of min value in distances is go to group 0 or 1 or ...  to k
                classification = distances.index(min(distances))
                ## add featureset to the the gourp of k
                # exmp: if you have k = 2 you have 2 group
                # so add the distances min value to her group 1 or 0 
                self.classifications[classification].append(featureset)
                
            ## last centroids point before get average
            prev_centroids = dict(self.centroids)
            for classification in self.classifications:
                ## get average to the point of the gourp 0 or 1 or ... in range k and add her to centroids dictionary
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)


            ## compare between original_centroid and current_centroid
            optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                ## get all current_centroid and original_centroid if they are within our required tolerance, this is good
                ## else the optimized = False and stop fist loop for i in range(self.max_iter):
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    #print(c, np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False
            
            if optimized:
                break
    def predict(self,data):
        ## get distances between featureset and center point (centroid)
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        ## get  index of min value in distances is go to group 0 or 1 or ...  to k
        classification = distances.index(min(distances))
        return classification


X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11]])
colors = 10*["g","r","c","b","k"]
clf = K_Means(k=2)
clf.fit(X)

##  scatter center point of ths groups
for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker="o", color="k", s=150, linewidths=5)
    
##  scatter classification point of ths group
for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)


## predict new features
new_features = np.array([[1, 3],
              [8, 9],
              [0, 3 ],
              [5, 4],
              [6, 4],])
for feature in new_features:
    classification = clf.predict(feature)
    plt.scatter(feature[0], feature[1], marker="*", color=colors[classification], s=150, linewidths=5)
plt.show()
