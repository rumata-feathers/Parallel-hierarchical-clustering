#pragma once
#include <array>
#include <vector>

#include "linkage.hpp"

// Returns the (n-1) x 4 linkage matrix: {cluster_i, cluster_j, distance,
// merged_size}
std::vector<std::array<double, 4>> hac_parallel(
    std::vector<std::vector<double>> dist, Linkage linkage, int n_threads);
