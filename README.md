# Applications-of-KMeans
KMeans algorithm implementation from scratch  and some of its applications

## 1. Introduction to clustering
</p>As the name suggests clustering is a way to cluster similar data points together. It is part of unsupervised machine learning problems where the data is unlabelled and uncategorized.</p> 
</p>By formal definition, clustering is a class of unsupervised algorithms used in machine learning to find patterns or natural groupings in data. It has a wide range of applications including but not limited to certain types of recommendation engines, image compression, market segmentation, document clustering etc. 
In this project, I will explore kmeans as a way to generate features for supervised machine learning algorithms as well as its application in image compression.</p>

## 2. K-Means Algorithm 
</p>The algorithm works by first selecting k unique points as the name suggests. </p>

1. Pick k unique points. This is the number of clusters we want to categorize or group our data into. There are several ways such as the Elbow method, Silhouette method to determine this hyperparameter.
2. Initialize the k clusters based on random data points from the sample. 
3. Compute distances of each point from the cluster centroid initialized in the first step.
4. Assign labels to each point based on minimum distance from the clusters
5. Compute mean of each cluster.
6. Reassign cluster centroid to the mean and recompute steps 3,4 and 5 till the clusters converge based on some predecided tolerance value.

### 2.1 Limitations of K-Means based on random initialization

1. Highly unstable performance for some use-cases due to the effect of random initialization.
2. Clusters converge to different values based on the initial cluster centroid location chosen.

### 2.2. K-Means++ Algorithm
</p> The main motivation here is to improve the stability of the kmeans centroid initialization and address the limitations of random initialization.</p>

</p> The main steps involved are as follows:</p>

1. Pick the initial point randomly
2. Compute distance from all existing clusters
3. Pick the next point based on the maximum of minimum distance from each cluster.
4. Repeat the process until all the k clusters is initialized.

</p> The k means++ algorithm improves the prediction performance and stability since the randomness of the initial points is addressed by the way of computing the next k-1 points which maximize the distance from points previously picked by the algorithm </p>

### 2.3. Elbow method to choose k
</p>The parameter k is quite important to determine since the incorrect value will result in groupings that might not be useful. Hence we will try to determine its value using some synthetic data that has about 4 clusters. The data distribution is as follows</p> 

<p float="left">
      <img src="images/blob.png" width="500" />
</p>

</p>The elbow method plots the number of clusters with the within-cluster sum of square(WCSS) distance. As seen from the figure below on the left we can see the elbow where the curve changes the slope and becomes constant or flat. The value at which this happens is our optimal k value.</p> 

