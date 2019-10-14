# Further Evalution and Clustering Functions

```@docs
distance_matrix
MCBB.distance_matrix_mmap
MCBB.compute_distance
MCBB.distance_matrix_sparse
AbstractDistanceMatrix
DistanceMatrix
DistanceMatrixHist
MCBB.metadata!(dm::AbstractDistanceMatrix)
wasserstein_histogram_distance
ecdf_hist
cluster_distance
cluster_means(sol::myMCSol, clusters::DbscanResult)
cluster_membership
MCBB.ClusterMembershipResult
Base.sort!(cm::ClusterMembershipResult; ignore_first::Bool)
Base.sum(cm::ClusterMembershipResult, indices::AbstractArray{Int,1})
get_trajectory
cluster_measure_mean
cluster_measure_std
cluster_measures
MCBB.ClusterMeasureResult
cluster_measures_sliding_histograms
MCBB.ClusterMeasureHistogramResult
ClusterICSpaces
cluster_n_noise
measure_on_parameter_sliding_window
k_dist(D::AbstractArray, k::Int=4)
KNN_dist
KNN_dist_relative
```
