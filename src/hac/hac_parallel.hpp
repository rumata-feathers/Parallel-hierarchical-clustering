#pragma once
#include <array>
#include <vector>

#include "linkage.hpp"

/*
    * @param dist : n x n distance matrix
    * @param linkage : linkage criterion
    * @param n_threads : number of threads to use
    * @returns (n-1) x 4 linkage matrix: {cluster_i, cluster_j, distance, merged_size}
*/
std::vector<std::tuple<int, int, double, int>> hac_parallel(
    const std::vector<std::vector<double>>& dist, const std::vector<std::vector<double>>& data,  Linkage linkage, int n_threads);
