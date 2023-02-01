import numpy as np
import pandas as pd
import sklearn as sk
import scipy.cluster.hierarchy as sch
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.cluster import BisectingKMeans

from sklearn.cluster import MeanShift
from sklearn.preprocessing import normalize
import sklearn.preprocessing
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
import sklearn.decomposition
import matplotlib.pyplot as plt
import json
import sklearn.cluster
import random


def ok_means(variance_threshold,
             iteration_start,
             max_iteration_number,
            
             data,
             number_of_clusters,
             
             use_ok_means):


    training_data = data.copy()
    outliers_index = []
    stop_criteria_met = False
    n_iteration = 0
    # set initial matrixes to be used
    cluster_indexes = np.arange(0, number_of_clusters)
    data_cluster = np.zeros(len(data))
    # minimum and maximum values from the dataset
    min_value, max_value = np.min(data, axis=0), np.max(data, axis=0)
    random_factor = 0
    set_initial_centroids = False
    while not set_initial_centroids:
        centroids = np.random.default_rng(random_factor).uniform(low=min_value, high=max_value,
                                                                 size=(number_of_clusters, *data[0].shape))

        cluster_indexes = np.arange(0, number_of_clusters)
        data_cluster = np.zeros(len(data))

        # sets data points to closest centroid
        for it, data_point in enumerate(data):
            distance_array = []

            for centroid in centroids:
                dist = np.linalg.norm(data_point - centroid)
                distance_array.append(dist)

            index = np.argmin(distance_array)
            data_cluster[it] = index

        if len(np.unique(data_cluster)) == number_of_clusters:
            set_initial_centroids = True
        else:
            random_factor = random_factor + 1

    # # sets data points to closest centroid
    # for it, data_point in enumerate(data):
    #     distance_array = []
    #
    #     for centroid in centroids:
    #         dist = np.linalg.norm(data_point - centroid)
    #         distance_array.append(dist)
    #
    #     index = np.argmin(distance_array)
    #     data_cluster[it] = index

    # algorithm start
    while not stop_criteria_met:

        if n_iteration >= iteration_start and use_ok_means == True:
            for cluster in cluster_indexes:
                filtered_data_cluster = []
                filtered_index_cluster = []
                for it, entry in enumerate(data_cluster):
                    if it not in outliers_index and cluster == entry:
                        filtered_data_cluster.append(training_data[it])
                        filtered_index_cluster.append(it)

                normalized_cluster = normalize(filtered_data_cluster, axis=0,
                                               norm='l2')

                indexes = np.argwhere(abs(normalized_cluster) > variance_threshold)

                if len(filtered_data_cluster) == len(np.unique(indexes[:, 0])):
                    continue

                # if len(np.unique(indexes[:, 0])) != 0:
                #     print(filtered_index_cluster)
                #     print(np.unique(indexes[:, 0]))
                for it in np.unique(indexes[:, 0]):
                    if filtered_index_cluster[it] not in outliers_index:
                        outliers_index.append(filtered_index_cluster[it])

        # updates datapoints
        for it, data_point in enumerate(training_data):
            distance_array = []

            if use_ok_means == True and it in outliers_index:
                continue
            for centroid in centroids:
                dist = np.linalg.norm(data_point - centroid)
                distance_array.append(dist)

            index = np.argmin(distance_array)

            data_cluster[it] = index

        # calculate new cluster centroids
        for it in cluster_indexes:
            occurrences = np.where(data_cluster == it)[0]

            if use_ok_means == True:
                filtered_occurrences = []
                for occurrence in occurrences:
                    if occurrence not in outliers_index:
                        filtered_occurrences.append(occurrence)
                cluster_values = training_data[filtered_occurrences]
            else:
                cluster_values = training_data[occurrences]

            centroids[it] = np.mean(cluster_values, axis=0)

        if n_iteration == max_iteration_number:
            stop_criteria_met = True

        n_iteration = n_iteration + 1


    data_cluster = np.delete(data_cluster, outliers_index).astype(int)
    return [np.delete(data[:], outliers_index, axis=0), data_cluster]
