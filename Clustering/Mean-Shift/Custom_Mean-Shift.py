import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

class Mean_Shift:
    def __init__(self, radius=4):
        self.radius = radius

    def fit(self, data):
        centroids = {}
        ## make id for values 
        for i in range(len(data)):
            centroids[i] = data[i]
        #print(centroids)

            
        #Make all datapoints centroids
        #Take mean of all featuresets within centroid's radius, setting this mean as new centroid.
        #Repeat step #2 until convergence.
        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                for featureset in data:
                    ## if distance between featureset and centroid less then radius
                    ## add featureset in bandwidth list
                    if np.linalg.norm(featureset-centroid) < self.radius:
                        in_bandwidth.append(featureset)
        
                # get the average between values in bandwidth list 
                new_centroid = np.average(in_bandwidth,axis=0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids))) # sorted and remove duplicate value
            print(new_centroids)
            prev_centroids = dict(centroids)

            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i]) ## add uniques in centroids

            optimized = True
            '''
            Here we note the previous centroids, before we begin to reset "current" or "new" centroids
            by setting them as the uniques. Finally, we compare the previous centroids to the new ones, and measure movement.
            If any of the centroids have moved, then we're not content that we've got full convergence
            and optimization, and we want to go ahead and run another cycle.
            If we are optimized, great, we break, and then finally set the centroids attribute to the final centroids we came up with.
            '''

            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break
                
            if optimized:
                break

        self.centroids = centroids

    def predict(self,data):
        pass


X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11],
              [8,2],
              [10,2],
              [9,3],])

colors = 10*["g","r","c","b","k"]

clf = Mean_Shift()
clf.fit(X)

centroids = clf.centroids

plt.scatter(X[:,0], X[:,1], s=150)

for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)

plt.show()
