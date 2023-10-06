# %%
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import random_center_initializer
from pyclustering.utils.metric import distance_metric
from pyclustering.utils.metric import type_metric
from pyclustering.cluster.encoder import cluster_encoder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from scipy.spatial.distance import cdist

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
# note: SCIPY 'jaccard' is the Jaccard-Tanimoto distance
#
#SCIPY_DISTANCES = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 
#                   'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulczynski1', 
#                   'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 
#                   'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
MY_DISTANCES = ['chebyshev', 'cityblock', 'euclidean', 'hamming', 'jaccard', 
                'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 
                'seuclidean', 'sokalmichener', 'sqeuclidean', 'yule']
MY_BIN_DISTANCES = ['hamming', 'jaccard', 'matching', 'rogerstanimoto', 'russellrao', 
                    'sokalmichener', 'yule']
MY_CLUSTER_ENGS = ['sklearn', 'pyclustering', 'scratch']

def my_euclidean(point1, point2):
  dimension = len(point1)
  result = 0.0
  for i in range(dimension):
    result += math.pow(point1[i] - point2[i], 2)
  return math.sqrt(result)

def my_jaccard_tanimoto(point1, point2):
  dimension = len(point1)
  a = 0
  bc = 0
  for i in range(dimension):
    if point1[i] == 1 and point2[i] == 1:
      a = a + 1
    elif point1[i] == 1 or point2[i] == 1:
      bc = bc + 1

  if (a + bc) == 0:
    return 0 # they are both all zero/one

  jt_sim = a / (a + bc) # similarity
  return 1 - jt_sim # distance

def my_hamming(point1, point2):
  dimension = len(point1)
  a = 0
  bc = 0
  for i in range(dimension):
    if point1[i] == 1 and point2[i] == 1:
      a = a + 1
    elif point1[i] == 1 or point2[i] == 1:
      bc = bc + 1
  return bc

def my_kmeans(df, n_clusters, n_iters, clstr_distance):
  X = df.to_numpy()
  
  # random init
  prng = np.random.RandomState(1)
  idxs = prng.choice(len(X), n_clusters, replace=False)
  centroids = X[idxs, :]
  
  labels = []
  for iter_idx in range(n_iters):
    distances = cdist(X, centroids, clstr_distance)
    labels = np.array([np.argmin(i) for i in distances])
    
    # update centroids (but the last time)
    if iter_idx < n_iters - 1:
      centroids = []
      for idx in range(n_clusters):
        cent_new = X[labels == idx].mean(axis=0) 
        centroids.append(cent_new)
      centroids = np.vstack(centroids)
    
      # purely binary  
      if (clstr_distance in MY_BIN_DISTANCES):
        centroids = np.where(centroids > 0.5, 1, 0)
                
  return (labels, centroids)

def clustering(clstr_eng, clstr_distance, df, n_clusters, save_imgs):
  clstr_name = ''
  centroids = None
  labels = None
  df_centroids = df[0:0].copy()
  if clstr_eng == 'sklearn':
    clstr_name = 'sklearn KMeans (euclidean)'
    sklearn_kmeans = KMeans(n_clusters=n_clusters, random_state=1, n_init='auto')
    sklearn_kmeans.fit(df)
    labels = sklearn_kmeans.labels_
    centroids = sklearn_kmeans.cluster_centers_
    for cent in centroids:
      df_centroids.loc[len(df_centroids.index)] = cent
  elif clstr_eng == 'pyclustering':
    metric = None
    if clstr_distance == 'euclidean':
      metric = distance_metric(type_metric.EUCLIDEAN)
      #metric = distance_metric(type_metric.USER_DEFINED, func=my_euclidean)
    elif clstr_distance == 'manhattan':
      metric = distance_metric(type_metric.MANHATTAN)
    elif clstr_distance == 'jaccard':
      metric = distance_metric(type_metric.USER_DEFINED, func=my_jaccard_tanimoto)
    elif clstr_distance == 'chebyshev':
      metric = distance_metric(type_metric.CHEBYSHEV)
    elif clstr_distance == 'minkowski':
      metric = distance_metric(type_metric.MINKOWSKI)
    elif clstr_distance == 'canberra':
      metric = distance_metric(type_metric.CANBERRA)
    elif clstr_distance == 'hamming':
      metric = distance_metric(type_metric.USER_DEFINED, func=my_hamming)
      
    clstr_name = 'pyclustering k-means (' + clstr_distance + ')'
    X = df.to_numpy()
    initial_centers = random_center_initializer(X, n_clusters, random_state=1).initialize()
    kmeans_instance = kmeans(X, initial_centers, metric=metric)
    kmeans_instance.process()
    pyCenters = kmeans_instance.get_centers()
    for cent in pyCenters:
      df_centroids.loc[len(df_centroids.index)] = cent
    pyClusters = kmeans_instance.get_clusters()
    pyEncoding = kmeans_instance.get_cluster_encoding()
    pyEncoder = cluster_encoder(pyEncoding, pyClusters, X)
    labels = pyEncoder.set_encoding(0).get_clusters()
  elif clstr_eng == 'scratch':
    metric = None
    clstr_name = 'scratch k-means (' + clstr_distance + ')'
    (labels, centroids) = my_kmeans(df, n_clusters, 100, clstr_distance)
    for cent in centroids:
      df_centroids.loc[len(df_centroids.index)] = cent

  # metrics
  #size_clusters = [sum(labels == label) for label in range(0, n_clusters)]
  size_clusters = [0 for label in range(0, n_clusters)]
  for label in labels:
    size_clusters[label] = size_clusters[label] + 1
  print ('------ ' + clstr_name)
  print('size clusters: ' + str(size_clusters))
  try:
    print('davies bouldin: ' + str(round(davies_bouldin_score(df, labels), 2)))
    print('silhouette: ' + str(round(silhouette_score(df, labels), 2)))
    print('calinski harabasz: ' + str(round(calinski_harabasz_score(df, labels))))
  except:
    print('error index calculation')
  
  # plots
  my_colors = ['b', 'g', 'r', 'c', 'm', 'y']
  fig, axs = plt.subplots(1, 3, figsize=(15, 8))
  fig.suptitle(clstr_name, fontsize=20)
  for ax_idx in range(0, 3):
    my_plot = None
    if ax_idx == 0:
      my_plot = MDS(n_components=2, normalized_stress='auto', random_state=0)
    elif ax_idx == 1:
      my_plot = PCA(n_components=2)
    elif ax_idx == 2:
      my_plot = TSNE(n_components=2, perplexity=1.5, random_state=1)

    # plot items
    df_transform = my_plot.fit_transform(df)
    plot_x = [row[0] for row in df_transform]
    plot_y = [row[1] for row in df_transform]
    df_plot = pd.DataFrame({'X': plot_x, 'Y': plot_y, 'label': labels})
    u_labels = np.unique(labels)
    for i in u_labels:
      axs[ax_idx].scatter(df_plot[df_plot.label == i]['X'], df_plot[df_plot.label == i]['Y'], label=i, color=my_colors[i])

    # plot centers      
    transform_centroids = my_plot.fit_transform(df_centroids)
    for i, cent in enumerate(transform_centroids):
      axs[ax_idx].scatter(cent[0], cent[1], marker='x', color=my_colors[i])

    axs[ax_idx].set_title(my_plot.__class__.__name__, fontsize=16)
    leg = axs[ax_idx].legend()
    for i in u_labels:
      leg.legend_handles[i].set_color(my_colors[i])
    axs[ax_idx].xaxis.set_tick_params(labelbottom=False)
    axs[ax_idx].yaxis.set_tick_params(labelleft=False)

  if save_imgs:
    plt.savefig('./images/' + clstr_eng + '_' + clstr_distance + '.png')

#
# MAIN
# 

# %% read data
df = pd.read_csv('./data/data.txt', sep='\t')

# %% show data features groups
clstr_cols = list(df.columns[df.columns.str.startswith('C')])
df_clstr = df[clstr_cols]
print('ngroups: ' + str(df.groupby(clstr_cols).ngroups))
print(df.groupby(clstr_cols)['Item'].agg('count').to_frame())

# %% test with single distance values
idx_a = 0
idx_b = 8
print(cdist([df_clstr.loc[idx_a]], [df_clstr.loc[idx_b]], 'euclidean')[0][0])
print(my_euclidean(df_clstr.loc[idx_a], df_clstr.loc[idx_b]))

print(cdist([df_clstr.loc[idx_a]], [df_clstr.loc[idx_b]], 'jaccard')[0][0])
print(my_jaccard_tanimoto(df_clstr.loc[idx_a], df_clstr.loc[idx_b]))

# %% distances
dists = pd.DataFrame(columns = MY_DISTANCES)
for dist_name in dists:
  dist_matrix = cdist(df_clstr, df_clstr, dist_name)
  dist_values = dist_matrix[np.triu_indices(len(dist_matrix), k = 1)] # get only upper triangle values (no diagonal)
  dists[dist_name] = dist_values

print(len(dists))
for col in dists:
  print(col + ': unique=' + str(dists[col].nunique()) + ', na=' + str(dists[col].isna().sum()))
dists.hist(bins=10)
plt.tight_layout()
plt.show()

# %% single clustering test
clustering('scratch', 'hamming', df_clstr, 3, False)

# %% clustering tests
n_clusters = 3
cluster_tests = [
  ['sklearn', 'euclidean'],
  ['pyclustering', 'euclidean'],
  #['pyclustering', 'jaccard'],
]
for dist_name in MY_DISTANCES:
  cluster_tests.append(['scratch', dist_name])

for cluster_test in cluster_tests:
  print(cluster_test)
  clustering(cluster_test[0], cluster_test[1], df_clstr, n_clusters, True)
