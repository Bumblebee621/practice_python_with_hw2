import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(2)
from data import load_data

def add_noise(data):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :return: data + noise, where noise~N(0,0.00001^2)
    """
    noise = np.random.normal(loc=0, scale=1e-5, size=data.shape)
    return data + noise


def choose_initial_centroids(data, k):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :param k: number of clusters
    :return: numpy array of k random items from dataset
    """
    n = data.shape[0]
    indices = np.random.choice(range(n), k, replace=False)
    return data[indices]


# ====================
def transform_data(df, features):
    """
    Performs the following transformations on df:
        - selecting relevant features
        - scaling
        - adding noise
    :param df: dataframe as was read from the original csv.
    :param features: list of 2 features from the dataframe
    :return: transformed data as numpy array of shape (n, 2)
    """
    data = df[features].to_numpy
    minimum = np.min(data, axis=0)
    maximum = np.max(data, axis=0)
    data = (data - minimum)/(maximum - minimum)
    data = add_noise(data)
    return data


def kmeans(data, k):
    """
    Running kmeans clustering algorithm.
    :param data: numpy array of shape (n, 2)
    :param k: desired number of cluster
    :return:
    * labels - numpy array of size n, where each entry is the predicted label (cluster number)
    * centroids - numpy array of shape (k, 2), centroid for each cluster.
    """
    centroids = choose_initial_centroids(data, k)
    prev_cent = []
    labels = []
    while not np.array_equal(prev_cent, centroids):
        prev_cent = centroids
        labels = assign_to_clusters(data, centroids)
        centroids = recompute_centroids(data, labels, k)
    return labels, centroids


def visualize_results(data, labels, centroids, path):
    """
    Visualizing results of the kmeans model, and saving the figure.
    :param data: data as numpy array of shape (n, 2)
    :param labels: the final labels of kmeans, as numpy array of size n
    :param centroids: the final centroids of kmeans, as numpy array of shape (k, 2)
    :param path: path to save the figure to.
    """
    pass
    # plt.savefig(path)


def dist(x, y, axis =0):
    """
    Euclidean distance between matrices x, y
    :param axis:
    :param x: numpy array of size n
    :param y: numpy array of size n
    :return: the Euclidean distance
    """
    return np.linalg.norm(x-y, axis= axis)

def assign_to_clusters(data, centroids):
    """
    Assign each data point to a cluster based on current centroids
    :param data: data as numpy array of shape (n, 2)
    :param centroids: current centroids as numpy array of shape (k, 2)
    :return: numpy array of size n
    """




    diffs = dist(data[:,np.newaxis,:], centroids[np.newaxis,:,:],2)
    labels = np.argmin(diffs,axis=1)
    return labels


def recompute_centroids(data, labels, k):
    """
    Recomputes new centroids based on the current assignment
    :param data: data as numpy array of shape (n, 2)
    :param labels: current assignments to clusters for each data point, as numpy array of size n
    :param k: number of clusters
    :return: numpy array of shape (k, 2)
    """
    centroids = np.array([data[labels == i].mean() for i in range(k)])

    for i in range(k):
        cluster = data.where(labels == i)
        centroids[i] = np.mean(cluster)
    return centroids

