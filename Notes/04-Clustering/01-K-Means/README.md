# INTRODUCTION

- Dataset may exist in groups and we may not realise.
- K-Means clustering helps us in finding these unknown groups (CLUSTERS) exists in our dataset.

## RANDOM INTIALIZATION MAP

- There is an assumption taken in the algorithm regarding the existence of the centroid.
- We want our model to as deterministic as possible.
- But different placement of centroid can bring out different results.
- That is called random intialization trap.

## SOLUTION

- KMeans++ Algorithm.
- Read about it in the , wikepedia.

## CHOOSING NUMBER OF CLUSTERS

- To find out if the number of clusters influence the clustering result. There came a requirement of quantifiable metric
  needed.
- Which tells us the number of clusters are optimal .
- We use WCSS score , and refer graph of the K means.

## EXECUTION UNDERSTANDING

- We would be creating a dependent variable
- The values of various attributes, will define the class of this attribute.
- So in here we do not have any dependent variable intially from the dataset, usually.
- And while choosing the attributes to be included in indeependent variable , some variable such as S.NO. , IDs, etc.
  needs to be neglected as they are different for each the row of data.
- Reason behind it is, we are trying to find the common ground here. Any thing which is not helping us to keep the data in clusters, is a detterent to the process, IDs and Sno are like primary keys to the dataset. Which act as detterent for clustering.
- In the dataset provided they are intial

# CODE

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

```python
dataset = pd.read_csv('/content/drive/MyDrive/Dataset/k_means/Mall_Customers.csv')
# We can specify the column index by specifying the column index such as 3, 4
X = dataset.iloc[:, [3, 4]].values
# print(X)
```

```python
from sklearn.cluster import KMeans
# We need to keep changing the number of clusters , so we would be using an iterative approach
# The loop would be based on the WCSS score, as it gets lower, we would be getting our clusters.
wcss = []
for i in range(1,11):
# initialization with kmeans++ avoids us to fall in the random intialization trap
  kmeans = KMeans(n*clusters = i,init = 'k-means++', random_state = 42)
  kmeans.fit(X)
  wcss.append(kmeans.inertia*)
plt.plot(range(1,11),wcss)
plt.title("The Elbow Method ")
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.show()
```

```
# From above we identified that the elbow ( that is point where the graph becomes straight)
# To zoom the graph , we will try to lower the limit.
wcss_less = []
for i in range(1,7):
  # initialization with kmeans++ avoids us to fall in the random intialization trap
  kmeans = KMeans(n*clusters = i,init = 'k-means++', random_state = 42)
  kmeans.fit(X)
  wcss_less.append(kmeans.inertia*)
plt.plot(range(1,7),wcss_less)
plt.title("The Elbow Method ")
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.show()
```

```python
# Now as we know our model is having optimal clusters as 5
# We will train our model

kmeans = KMeans(n_clusters = 5,init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
print(y_kmeans)

# to select all the customers where y_means ==0 ,
# We are specifying the index of where y means == 0 or 1 or 2 or 3 or 4

plt.scatter(X[y_kmeans == 0,0],X[y_kmeans == 0,1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1,0],X[y_kmeans == 1,1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2,0],X[y_kmeans == 2,1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3,0],X[y_kmeans == 3,1], s = 100, c = 'yellow', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4,0],X[y_kmeans == 4,1], s = 100, c = 'black', label = 'Cluster 5')
plt.scatter(kmeans.cluster*centers*[:,0],kmeans.cluster*centers*[:,1],s = 300, c = 'magenta',label = 'centroid')
plt.title("Clusters of customers ")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100")
plt.legend()
plt.show()
```
