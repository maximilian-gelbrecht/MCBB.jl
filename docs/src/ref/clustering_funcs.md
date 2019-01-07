# Further Evalution and Clustering Functions

```@docs
distance_matrix
cluster_means(sol::myMCSol, clusters::DbscanResult)
cluster_membership(par::AbstractArray, clusters::DbscanResult)
cluster_membership(prob::myMCProblem, clusters::DbscanResult, window_size::AbstractArray, window_offset::AbstractArray, normalize::Bool=true)
cluster_measures
ClusterICSpaces
cluster_n_noise
measure_on_parameter_sliding_window
k_dist(D::AbstractArray, k::Int=4)
KNN_dist
KNN_dist_relative
```
