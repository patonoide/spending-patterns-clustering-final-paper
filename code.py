import numpy as np
import pandas as pd
import sklearn as sk
import scipy.cluster.hierarchy as sch
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.cluster import BisectingKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
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


dataset_url = 'dataset_pc.json'

file = open(dataset_url)
data = json.load(file)
file.close()

data_frame = pd.DataFrame(data)


n_clusters = 4

normalized_data = normalize(data_frame, axis=0, norm='l2')

tree_outliers = IsolationForest(n_estimators=90, random_state=0).fit_predict(data_frame)


dataset_no_outliers = []
raw_dataset_no_outliers = []

for it in range(len(tree_outliers)):
    if(tree_outliers[it] == 1):
        dataset_no_outliers.append(normalized_data[it])
        # raw_dataset_no_outliers.append(data_frame[it])

dataset_no_outliers = normalized_data

# clf = LocalOutlierFactor(n_neighbors=5)

# clf.fit_predict(data_frame)



# for it in range(len(clf.negative_outlier_factor_)):
#     if clf.negative_outlier_factor_[it] > -40:
#         dataset_no_outliers.append(normalized_data[it])


# dataset_no_outliers = normalized_data

data_frame = pd.DataFrame(dataset_no_outliers, columns= data_frame.columns)

# kmeans = MeanShift(bandwidth=0.5)
# kmeans = AgglomerativeClustering(n_clusters=n_clusters, linkage='single')
# kmeans = BisectingKMeans(n_clusters=n_clusters, random_state=0, bisecting_strategy="biggest_inertia")
# kmeans = sk.cluster.MeanShift(bandwidth=1.3)
kmeans = KMeans(n_clusters=n_clusters, random_state=0)

pca = PCA(10)

pca_data = pca.fit_transform(dataset_no_outliers)


kmeans_predicted_labels = kmeans.fit_predict(dataset_no_outliers)
# n_clusters = len(kmeans.cluster_centers_)

silhouette = silhouette_score(dataset_no_outliers, kmeans_predicted_labels)
calinski = calinski_harabasz_score(dataset_no_outliers, kmeans_predicted_labels)
davis = davies_bouldin_score(dataset_no_outliers, kmeans_predicted_labels)





print('CH Score ', calinski)
print('DB Score', davis)
print('Silhouette Score', silhouette)



data_frame.insert(0, "cluster", kmeans_predicted_labels, True)





features = data_frame.columns
ncols = 5
nrows = len(features) // ncols + (len(features) % ncols > 0)
# nrows = 4
fig = plt.figure(figsize=(15,15))
cluster_colors = color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(n_clusters)]



for n, feature in enumerate(features.drop('cluster')):    


    # if(n >= 20):
    
    ax = plt.subplot(nrows, ncols, n + 1)
    box = data_frame[[feature, 'cluster']].boxplot(by='cluster',ax=ax,return_type='both',patch_artist = True)
    # print(len(data_frame[[feature, 'cluster']]))

    for row_key, (ax,row) in box.items():
        ax.set_xlabel('cluster')
        ax.set_title(feature,fontweight="bold")
        for i,box in enumerate(row['boxes']):
            box.set_facecolor(cluster_colors[i])

fig.suptitle('Feature distribution per cluster', fontsize=18, y=1)
fig = plt.gcf()
fig.set_size_inches(30, 40)
fig.savefig("test.png", dpi=50)
# plt.tight_layout()
# plt.show()

u_labels = np.unique(kmeans_predicted_labels)

pca = PCA(2)

pca_data = pca.fit_transform(dataset_no_outliers)
plt.clf()
#plotting the results:
for i in u_labels:
    plt.scatter(pca_data[kmeans_predicted_labels == i , 0] , pca_data[kmeans_predicted_labels == i , 1])
fig = plt.gcf()
fig.set_size_inches(30, 30)
fig.savefig("test_2.png", dpi=50)


plt.clf()
colors = ['#9EBD6E','#81a094','#775b59','#32161f', '#946846', '#E3C16F', '#fe938c', '#E6B89C','#EAD2AC',
          '#DE9E36', '#4281A4','#37323E','#95818D'
         ]

X_mean = pd.concat([pd.DataFrame(data_frame.mean().drop('cluster'), columns=['mean']), 
                   data_frame.groupby('cluster').mean().T], axis=1)

X_dev_rel = X_mean.apply(lambda x: round((x-x['mean'])/x['mean'],2)*100, axis = 1)

fig = plt.figure(figsize=(80,40), dpi=200)
X_dev_rel.T.plot(kind='bar', 
                       ax=fig.add_subplot(), 
                       title="Cluster characteristics", 
                       color=colors,
                       xlabel="Cluster",
                       ylabel="Deviation from overall mean in %"
                      )
plt.axhline(y=0, linewidth=1, ls='--', color='black')
plt.legend(bbox_to_anchor=(1.04,1))
fig.autofmt_xdate(rotation=0)
# plt.tight_layout()
fig.savefig("test_3.png", dpi=50)

