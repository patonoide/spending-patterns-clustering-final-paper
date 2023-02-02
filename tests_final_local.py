import numpy as np
import pandas as pd
import sklearn as sk
import scipy.cluster.hierarchy as sch
import sklearn.preprocessing
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
import sklearn.decomposition
import matplotlib.pyplot as plt
import json
import sklearn.cluster

file = open('dataset_mb.json')
data = json.load(file)
file.close()


data_frame = pd.DataFrame(data)
numpy_data = data_frame.to_numpy()
normalized_data = sk.preprocessing.normalize(numpy_data, axis=0, norm='l2')
formatted_data_local_kmeans = pd.DataFrame(
    columns=['N Clusters', 'Silhoutte Score', 'CH Score', 'Davis Score', 'Local Outlier Factor Neighbors'])

formatted_data_local_bkmeans = pd.DataFrame(
    columns=['N Clusters', 'Silhoutte Score', 'CH Score', 'Davis Score', 'Local Outlier Factor Neighbors'])

formatted_data_local_mean_shift = pd.DataFrame(
    columns=['Bandwidth', 'Silhoutte Score', 'CH Score', 'Davis Score', 'N clusters', 'Local Outlier Factor Neighbors'])

for neighbors in range(10, 51):

    clf = LocalOutlierFactor(n_neighbors=neighbors)

    clf.fit_predict(numpy_data)

    dataset_no_outliers = []

    for it in range(len(clf.negative_outlier_factor_)):
        if clf.negative_outlier_factor_[it] > -40:
            dataset_no_outliers.append(normalized_data[it])



    sel_variance = VarianceThreshold()
    normalized_dataset_no_outliers = sel_variance.fit_transform(dataset_no_outliers)

        
    for i in range(2, 11):
        kmeans = sk.cluster.KMeans(n_clusters=i, random_state=0)
        kmeans_predicted_labels = kmeans.fit_predict(dataset_no_outliers)

        k_score = sk.metrics.silhouette_score(dataset_no_outliers, kmeans_predicted_labels)
        calinski = sk.metrics.calinski_harabasz_score(dataset_no_outliers, kmeans_predicted_labels)
        davis = sk.metrics.davies_bouldin_score(
        dataset_no_outliers, kmeans_predicted_labels)
        print('Kmeans for n clusters ', i,
          ' has silhouette score of ', k_score, ', CH score of ', calinski, ' and Davis Boundin score of ', davis)

        formatted_data_local_kmeans = pd.concat([formatted_data_local_kmeans, pd.DataFrame(
        [[i, k_score, calinski, davis, neighbors]], columns=['N Clusters', 'Silhoutte Score', 'CH Score', 'Davis Score', 'Local Outlier Factor Neighbors'])])

        bkmeans = sk.cluster.BisectingKMeans(n_clusters=i, random_state=0, bisecting_strategy="biggest_inertia")

        bkmeans_predicted_labels = bkmeans.fit_predict(normalized_dataset_no_outliers)

        bk_score = sk.metrics.silhouette_score(normalized_dataset_no_outliers, bkmeans_predicted_labels)
        bcalinski = sk.metrics.calinski_harabasz_score(normalized_dataset_no_outliers, bkmeans_predicted_labels)
        bdavis = sk.metrics.davies_bouldin_score(
        normalized_dataset_no_outliers, bkmeans_predicted_labels)
        print('BKmeans for n clusters ', i,
          ' has silhouette score of ', bk_score, ', CH score of ', bcalinski, ' and Davis Boundin score of ', bdavis)

        formatted_data_local_bkmeans = pd.concat([formatted_data_local_bkmeans, pd.DataFrame(
        [[i, bk_score, bcalinski, bdavis, neighbors]], columns=['N Clusters', 'Silhoutte Score', 'CH Score', 'Davis Score', 'Local Outlier Factor Neighbors'])])


    for i in range(80, 225, 5):

      i = i/100
      meanshift = sk.cluster.MeanShift(bandwidth=i)
      meanshift_predicted_labels = meanshift.fit_predict(dataset_no_outliers)

      n_clusters = len(meanshift.cluster_centers_)
      if(n_clusters < 2):
          continue 

      k_score = sk.metrics.silhouette_score(dataset_no_outliers, meanshift_predicted_labels)
      calinski = sk.metrics.calinski_harabasz_score(
        dataset_no_outliers, meanshift_predicted_labels)
      davis = sk.metrics.davies_bouldin_score(
        dataset_no_outliers, meanshift_predicted_labels)
      print('Meanshift with bandwidth ', i,
          ' has silhouette score of ', k_score, ', CH score of ', calinski, ' Davis Boundin score of ', davis, ' and N clusters of ',n_clusters )

      formatted_data_local_mean_shift = pd.concat([formatted_data_local_mean_shift, pd.DataFrame(
        [[i, k_score, calinski, davis, n_clusters, neighbors]], columns=['Bandwidth', 'Silhoutte Score', 'CH Score', 'Davis Score', 'N clusters','Local Outlier Factor Neighbors'])])


formatted_data_local_bkmeans.to_pickle('./pickle_mb/bkmeans_local')
formatted_data_local_kmeans.to_pickle('./pickle_mb/kmeans_local')
formatted_data_local_mean_shift.to_pickle('./pickle_mb/mean_shift_local')

    
        

