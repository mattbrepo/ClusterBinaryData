# ClusterBinaryData
Cluster on binary data (also using specific distances)

**Language: Python**

**Start: 2023**

## Why
I needed to perform a [cluster analysis](https://en.wikipedia.org/wiki/Cluster_analysis) on data composed of binary features. I wanted to test how the following distances would affect the cluster performance:

- [euclidean](https://en.wikipedia.org/wiki/Euclidean_distance)
- [chebyshev](https://en.wikipedia.org/wiki/Chebyshev_distance)
- [cityblock](https://en.wikipedia.org/wiki/Taxicab_geometry) (aka Manhattan)
- hamming
- [jaccard](https://en.wikipedia.org/wiki/Jaccard_index#Tanimoto_similarity_and_distance) (aka Jaccard-Tanimoto)
- mahalanobis
- rogerstanimoto
- russellrao
- seuclidean
- sqeuclidean
- yule

## Dataset
The dataset has 100 rows and 6 columns: a unique ID (_Item_) and 5 binary features (_C1...C5_). These features can be grouped in 24 groups:

```
ngroups: 24
                   
C1 C2 C3 C4 C5 C6  Item (count)
0  0  0  0  0  0     10
0  0  0  0  1  0      3
0  0  0  1  0  0      4
0  0  1  0  0  0      5
0  1  0  0  0  0      5
0  1  0  0  0  1      3
0  1  0  0  1  0      8
0  1  0  1  0  0      1
0  1  1  0  0  0      1
0  1  1  1  0  0      1
0  1  1  1  1  0      8
1  0  0  0  0  0      5
1  0  1  1  1  1      1
1  1  0  0  0  0     11
1  1  0  0  0  1      7
1  1  0  0  1  1     11
1  1  0  1  0  0      2
1  1  0  1  0  1      1
1  1  1  0  0  0      1
1  1  1  0  1  1      1
1  1  1  1  0  0      1
1  1  1  1  0  1      1
1  1  1  1  1  0      1
1  1  1  1  1  1      8
```

## Distances
I calculated the distances for all 4,950 pairs and prepared the histograms:

![distance histograms](/images/distance_histos.png)

From which it is quite evident that, for binary data, some distances are the same or highly correlated (e.g., euclidean/minkowski, hamming/matching, rogerstanimoto/sokalmichener).

## Cluster analysis
First, I used [scikit-learn](https://scikit-learn.org/) to apply the [k-means](https://en.wikipedia.org/wiki/K-means_clustering) on the data and tested some [internal evaluation](https://en.wikipedia.org/wiki/Cluster_analysis#Internal_evaluation) metrics like the [Davies–Bouldin index](https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index), [Silhouette coefficient](https://en.wikipedia.org/wiki/Silhouette_(clustering)) and [Calinski-Harabasz index](https://fr.wikipedia.org/wiki/Indice_de_Calinski-Harabasz).

Then, I used another library called [pyclustering](https://pyclustering.github.io/), to choose specific distances for the k-means.

Unfortunately, pyclustering shows some issues when dealing with distances like the Jaccard-Tanimoto. Therefore, I also implemented a simple K-means from scratch.

Results:

 Library      | Distance       | Cluster sizes  |Davies–Bouldin index | Silhouette coefficient | Calinski Harabasz |
--------------|----------------|----------------|---------------------|------------------------|-------------------|
 sklearn      | euclidean      | 44, 22, 34     | 1.31                | 0.34                   | 42                |
 pyclustering | euclidean      | 49, 24, 27     | 1.53                | 0.29                   | 32                |
 scratch      | euclidean      | 13, 41, 46     | 1.29                | 0.3                    | 35                |
 scratch      | chebyshev      | 85, 7, 8       | 0.98                | 0.06                   | 10                |
 scratch      | cityblock      | 14, 55, 31     | 1.51                | 0.25                   | 27                |
 scratch      | hamming        | 33, 56, 11     | 1.31                | 0.26                   | 28                |
 scratch      | jaccard        | 45, 39, 16     | 1.7                 | 0.21                   | 20                |
 scratch      | mahalanobis    | 13, 56, 31     | 1.49                | 0.26                   | 28                |
 scratch      | rogerstanimoto | 33, 56, 11     | 1.31                | 0.26                   | 28                |
 scratch      | russellrao     | 100, 0, 0      | -                   | -                      | -                 |
 scratch      | seuclidean     | 13, 56, 31     | 1.49                | 0.26                   | 28                |
 scratch      | sqeuclidean    | 13, 41, 46     | 1.29                | 0.3                    | 35                |
 scratch      | yule           | 65, 19, 16     | 1.97                | 0.19                   | 14                |

Considering that:
- Davies–Bouldin index has a maximum value of 0 and is better when smaller
- Silhouette coefficient has a maximum value of 1 and is better when bigger
- Calinski-Harabasz index is better when bigger

it's possible to say that the euclidean distance seems to be the best choice on this dataset.

## Cluster visualization
To visualize the clustering 3 different techniques where used: [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis), [TSNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) and [MDS](https://en.wikipedia.org/wiki/Multidimensional_scaling).

**sklearn euclidean**
![sklearn euclidean](/images/sklearn_euclidean.png)

**pyclustering euclidean**
![pyclustering euclidean](/images/pyclustering_euclidean.png)

**scratch euclidean**
![scratch euclidean](/images/scratch_euclidean.png)

**chebyshev**
![manhattan](/images/scratch_chebyshev.png)

**cityblock**
![cityblock](/images/scratch_cityblock.png)

**hamming**
![hamming](/images/scratch_hamming.png)

**jaccard**
![jaccard](/images/scratch_jaccard.png)

**mahalanobis**
![mahalanobis](/images/scratch_mahalanobis.png)

**rogerstanimoto**
![rogerstanimoto](/images/scratch_rogerstanimoto.png)

**seuclidean**
![seuclidean](/images/scratch_seuclidean.png)

**sqeuclidean**
![sqeuclidean](/images/scratch_sqeuclidean.png)

**yule**
![yule](/images/scratch_yule.png)
