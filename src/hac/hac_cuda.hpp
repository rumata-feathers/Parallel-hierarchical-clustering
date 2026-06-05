#pragma once
#include <array>
#include <vector>
#include "linkage.hpp"

// Returns the (n-1) x 4 linkage matrix: {cluster_i, cluster_j, distance, merged_size}
std::vector<std::tuple<int, int, double, int>> hac_cuda(
    std::vector<std::vector<double>> dist, const std::vector<std::vector<double>>& data,
    Linkage linkage);


