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

<p float="left">
      <img src="images/elbow.png" width="500" />
</p>
</p> Clearly, there is an elbow observed at k = 4 for the curve on the left since the curve almost becomes flat and changes the behavior at this point. This is one of the simplest methods to determine the optimal number of clusters. However it is not always clear which point is the elbow as seen above from the plot on the right. This happens because the separation in the clusters is not clear.
</p>
## 3. Application of K-Means for classification
</p>Despite kmeans being an unsupervised problem it can still be used for classification problems. Here I will be using scikit-learns inbuilt breast cancer dataset to classify healthy patients from patients having cancer. Despite being an unsupervised algorithm it performs decently giving an accuracy score of 91.2%.</p>

<p float="left">
      <img src="images/confusion_matrix.png" width="500" />
</p>
## 4. Application of K-Means for grouping and finding patterns/associations
</p>One of the use-cases of kmeans revolves around finding associations and patterns in the data by grouping certain features.
I will be using the California housing prices data here which contains information regarding latitude, longitude, household income etc. I will try to find patterns between location data and average household price based on clusters created using location and median household income. Based on the plots below it is clear that a relationship exists between location(latitude and longitude) and median household income/median house price </p>
<p float="left">
      <img src="images/scatter.png" width="500" />
</p>
<p float="left">
      <img src="images/violin_plot.png" width="500" />
</p>
</p>Here from the above plot, it is clear that the median house value can be clustered based on income and location as we see pockets of very high median house value in clusters 1, 3 and 4 whereas clusters 0, 2 and 5 clusters have very low median house value. This gives us the underlying relation between location and median house value.</p>
## 5. Application of K-Means to Image compression
</p>One of the applications of kmeans is in image compression. Each pixel in an image takes a value from 0 to 255 and is of size 3 bytes (RGB). So we have about 16 million colors possible (256*256*256). However human eye cannot perceive these many colors. Hence reducing the number of colors in an image to only a handful will suffice in most cases. I will show examples below where I will reduce the colors to only 32 and produces an almost identical image.</p>
</p>Below are my calico cats taking an afternoon nap retaining all their glorious colors even when we reduce the number of colors to 32: </p>
<p float="left">
      <img src="images/cats.png" width="500" />
</p>
</p>Applying image compression to black and white image of famous San-Francisco bridge (Credits: https://www.pinterest.com/pin/67694800618575464/) we get following result:</p>
<p float="left">
      <img src="images/bay_bridge.png" width="500" />
</p>
</p>
Compression using kmeans is not perfect as we can see that the below image of San Francisco's ocean beach as it suffers due to the lack of colors in the high contrast regions of the image on the right. This can be fixed by better initialization techniques and choosing(tuning) the appropriate number of colors for the image. </p>
<p float="left">
      <img src="images/ocean_beach.png" width="500" />
</p>

## 6. Spectral clustering
</p>Another limitation of kmeans algorithm is that it cannot cluster highly non-convex data like nested circles on the 2D plane. Hence we have methods like Spectral clustering which apply complex linear algebra to get clusters. </p>
Spectral clustering makes use of eigenvalues (spectrum) of the similarity matrix of the data to perform dimensionality reduction of complex multidimensional data to cluster in fewer dimensions. It first groups data based on graphs (connectivity approach) to identify communities of data points that are in the vicinity of each other.</p>
<p float="left">
      <img src="images/kmeans_fail.png" width="500" />
</p>
We need to generate a similarity matrix. Here we will use Breiman's Random forest using random forest classifier from sklearn.
</p> To generate a similarity matrix we need to follow these steps:</p>

1. Shuffle the samples and treat them as negative samples
2. Combine the original sample(taken as positive) and shuffled sample to train a random forest classifier. Be sure to keep high number of estimators and minimum leaf samples to get a good result.
3. Create a matrix by counting all the pairs that appear in the same leaf from the trained model.
4. Normalize the matrix by dividing the number of trees to get the similarity matrix.

Here I will use sklearn's built-in spectral clustering algorithm implementation by passing in the computed similarity matrix.
<p float="left">
      <img src="images/nested-spectral.png" width="500" />
</p>
</p>As seen above the spectral clustering does a wonderful job in separating the clusters formed by the concentric circles which our kmeans algorithm failed to perform earlier.</p>

## 7. Summary
</p>To summarize we have discussed how kmeans algorithm works along with some limitations of kmeans and why we need kmeans++ initialization to improve the stability and performance.</p>
</p>Further, we discussed some of the applications of kmeans in classification, grouping and finding association in data, its application in image compression. We also found a limitation in kmeans as it cannot cluster data that is highly non-convex like concentric circles.</p>
</p>Finally we found a way to cluster non-convex data using spectral clustering by building a similarity matrix using Breiman's trick.</p>
