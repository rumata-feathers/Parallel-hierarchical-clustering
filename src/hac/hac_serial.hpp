#pragma once
#include <array>
#include <vector>

#include "linkage.hpp"

/*
* @param dist : n x n distance matrix (passed by value — modified in place)
* @param df : n x d data matrix (used for centroid linkage)
* @param linkage : linkage criterion
* @returns (n-1) x 4 linkage matrix: {cluster_i, cluster_j, distance, merged_size}
*/
std::vector<std::array<double, 4>> hac_serial(
    const std::vector<std::vector<double>>& dist, const std::vector<std::vector<double>>& df, Linkage linkage);