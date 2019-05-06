# Further Evalution and Clustering Functions

```@docs
distance_matrix
compute_distance
AbstractDistanceMatrix
DistanceMatrix
DistanceMatrixHist
wasserstein_histogram_distance
ecdf_hist
cluster_distance
cluster_means(sol::myMCSol, clusters::DbscanResult)
cluster_membership(par::AbstractArray, clusters::DbscanResult)
cluster_membership(prob::myMCProblem, clusters::DbscanResult, window_size::AbstractArray, window_offset::AbstractArray, normalize::Bool=true)
MCBB.ClusterMembershipResult
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
