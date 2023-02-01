import sklearn as sk
from ok_means_implementation import ok_means
import pandas as pd
import json
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt

dataset_path = 'dataset_pc.json'

file = open(dataset_path)
data = json.load(file)
file.close()

data_frame = pd.DataFrame(data)

normalized_data = normalize(data_frame, axis=0, norm='l2')

threshold_values = []
anomalies_values = []

for x in range(2, 9, 1):

    # threshold = x/100
    # threshold_values.append(threshold)

    clustered_data, data_cluster = ok_means(variance_threshold=0.9,
                                        iteration_start=200,
                                        max_iteration_number=500,
                                        data=normalized_data,
                                        number_of_clusters=x,
                                        use_ok_means=True,
                                        )



    unique_okmeans, counts_okmeans = np.unique(data_cluster, return_counts=True)
    print(unique_okmeans)
    print(counts_okmeans)
    anomalies_values.append(109 - np.sum(counts_okmeans))
    print(dict(zip(unique_okmeans, counts_okmeans)))
    ok_means_s_score = sk.metrics.silhouette_score(clustered_data, data_cluster)
    # print(data_cluster)
    print(ok_means_s_score)


print(threshold_values)
print(anomalies_values)


# plt.plot(threshold_values, anomalies_values)
#
# plt.xlabel('Threshold Value')
# plt.ylabel('Number of Outliers')
# plt.title('Number of Outliers vs Threshold Value')
#
#
# plt.show()


    # ok_means_s_score = sk.metrics.silhouette_score(clustered_data, data_cluster)
    # print(data_cluster)
    # print(ok_means_s_score)
