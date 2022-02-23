import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.image as mpimg
import seaborn as sns
from PIL import Image
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_blobs
import scipy.spatial.distance
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
from sklearn.preprocessing import normalize
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, accuracy_score
import cv2


def kmeans(X: np.ndarray, k: int, centroids=None, max_iter=30, tolerance=1e-2):
    """
    :param X: Numpy array with shape (Number of records, number of features)
    :param k: Number of clusters
    :param centroids: Centroid initialization method. Default None for random initialization.
    :param max_iter: Default 30
    :param tolerance: Default 10^-2
    :return: Centroids with shape (k, Number of features) and labels for each input record
    """
    n, p = X.shape
    if centroids == 'kmeans++':
        centroids = select_centroids(X, k)
    else:
        centroids = np.zeros((k, p))
        idx = np.random.choice(n, k, replace=False)
        centroids = np.copy(X[idx, :])
    distances_from_centroids = np.zeros((n, k))
    for i in range(max_iter):
        old_distance = np.copy(distances_from_centroids)
        distances_from_centroids = scipy.spatial.distance.cdist(X, centroids)
        # distances_from_centroids = np.array([np.linalg.norm(x.astype(None) - centroids.astype(None), axis=1) for x in X])
        labels = distances_from_centroids.argmin(axis=1)
        for j in range(k):
            if len(X[labels == j, :]) < 1:
                idx = np.random.choice(n, 1, replace=False)
                centroids[j, :] = X[idx, :]
            else:
                centroids[j, :] = np.mean(X[labels == j, :], axis=0)
        if np.linalg.norm(old_distance - distances_from_centroids) < tolerance:
            break
    return centroids, labels


def select_centroids(X, k):
    """
    :param X: Numpy array with shape (Number of records, number of features)
    :param k: Number of clusters to be initialized
    :return
    kmeans++ algorithm to select initial points:
    1. Pick first point randomly
    2. Pick next k-1 points by selecting points that maximize the minimum
       distance to all existing clusters. So for each point, compute distance
       to each cluster and find that minimum.  Among the min distances to a cluster
       for each point, find the max distance. The associated point is the new centroid.
    Return centroids as k x p array of points from X.
    """
    n, p = X.shape
    c1 = np.random.choice(n, 1, replace=False)
    centroids = X[c1, :].reshape(1, -1)
    for i in range(1, k):
        distances_from_centroids = scipy.spatial.distance.cdist(X, centroids)
        # distances_from_centroids = np.array([np.linalg.norm(x.astype(None) - centroids.astype(None), axis=1) for x in X])
        centroids = np.append(centroids, X[np.argmax(distances_from_centroids.min(axis=1))].reshape(1, -1), axis=0)
    return centroids


def SimilarityMatrix(rf, X):
    """
    :param rf: pretrained random forest model
    :param X:  Numpy array with shape (Number of records, number of features)
    :return: similarity matrix based on leaf indices of x obtained from every tree in the forest
    """
    leaf_ids = rf.apply(X)
    nTrees = leaf_ids.shape[1]

    a = leaf_ids[:, 0]
    SimilarityMat = 1 * np.equal.outer(a, a)  # Inititalization of matrix

    for t in range(1, nTrees):  #
        a = leaf_ids[:, t]
        SimilarityMat += 1 * np.equal.outer(a, a)
    SimilarityMat = SimilarityMat / nTrees
    return SimilarityMat


def wcss(X, centroids, labels):
    """
    :param X: Numpy array with shape (Number of records, number of features)
    :param centroids: Numpy array with shape (k, number of features)
    :param labels: Numpy array with shape (Number of records) indicating which cluster each data point belongs to
    :return: Within cluster sum of squares
    """
    s = 0
    for j in range(centroids.shape[0]):
        s += np.sum((scipy.spatial.distance.cdist(X[labels == j], centroids[j].reshape(1, -1))) ** 2)
    return s


def ElbowPlot(X, K=10, title='Elbow plot'):
    """
    :param X: Numpy array with shape (Number of records, number of features)
    :param K: Number of k values to plot in X-axis
    :param title:
    :return: Matplotlib object
    """
    S = []
    for k in range(1, K + 1):
        centroids, labels = kmeans(X, k=k, centroids='kmeans++', tolerance=0.01)
        s = wcss(X, centroids, labels)
        S.append(s)
    plt.plot(range(1, K + 1), S)
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.title(title)
    plt.grid(b=None)
    return plt
