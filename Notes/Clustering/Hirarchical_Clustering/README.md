# INTRODUCTION

- There are two types of hierarchical clustering
  1. Agglomerative (Bottom up)
     i. Steps given in the photos of this folder
  2. Divisive (Top Down)

## DENDOGRAMS

- Basically a diagram type structure helps in the deciding clusters (in this sense).
- We can define a limiting constraint for DISSIMILARITY which can help us on the define clusters.
- The number of clusters defined by , number of intersection of horizontal differentiating line, with the vertical line.
- To find the optimal # of clusters: Find the longest vertical distance line. Which do not crosses any horizontal line (Even if the horizontal lines in the dendogram are extended).

# CODE

## PREPROCESSING

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('/content/drive/MyDrive/Dataset/k_means/Mall_Customers.csv')

# We can specify the column index by specifying the column index such as 3, 4
X = dataset.iloc[:, [3, 4]].values
# print(X)
```

### DENDROGRAM USAGE

```python
# library used here is scipy
import scipy.cluster.hierarchy as sch
# second parameter of linkage is ward.
# it consists of minimising the variance inside the clusters
dendrogram = sch.dendrogram(sch.linkage(X,method= 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()
```

### TRAINING

```python
# Now through the we recognize the 5 clusters
from sklearn.cluster import AgglomerativeClustering
# affinity : distance type
hc = AgglomerativeClustering(n_clusters = 5,affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)
```

### VISUALISATION

```python
plt.scatter(X[y_hc == 0,0],X[y_hc == 0,1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1,0],X[y_hc == 1,1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2,0],X[y_hc == 2,1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3,0],X[y_hc == 3,1], s = 100, c = 'yellow', label = 'Cluster 4')
plt.scatter(X[y_hc == 4,0],X[y_hc == 4,1], s = 100, c = 'black', label = 'Cluster 5')

# plt.scatter(hc.cluster*centers*[:,0],hc.cluster*centers*[:,1],s = 300, c = 'magenta',label = 'centroid')

plt.title("Clusters of customers ")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100")
plt.legend()
plt.show()

```
