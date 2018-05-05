import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
style.use('ggplot')

#ORIGINAL:

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

## X[:, 0] X values
## X[:, 1]) Y values
#plt.scatter(X[:, 0],X[:, 1], s=150, linewidths = 5, zorder = 10)
#plt.show()

clf = KMeans(n_clusters=2) ## create 2 group 
clf.fit(X)
centroids = clf.cluster_centers_ ## the center point of the group 
labels = clf.labels_ ## the point in group 0 or 1 or number of group form 0 to n_clusters
#print(labels)

colors = ["g","r","c","y","k","b"]
for i in range(len(X)):
    print(colors[labels[i]], X[i]) # (x,y) with them color
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)
plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)

## predict new features
new_features = np.array([[1, 3],
              [8, 9],
              [0, 3 ],
              [5, 4],
              [6, 4]])

new_features = new_features.reshape((new_features.shape[0],-1))
classification = clf.predict(new_features)
i = 0 
for feature in new_features:
    c = colors[classification[i]]
    plt.scatter(feature[0], feature[1], marker="*", color=c, s=150, linewidths=5)
    i += 1
plt.show()

