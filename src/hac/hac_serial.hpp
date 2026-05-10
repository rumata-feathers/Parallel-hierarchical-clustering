#pragma once
#include <array>
#include <vector>

#include "linkage.hpp"

// Returns (n-1) x 4 linkage matrix: {cluster_i, cluster_j, distance, merged_size}
// dist is the n x n point-to-point distance matrix and should never be modified.
std::vector<std::array<double, 4>> hac_serial(
    const std::vector<std::vector<double>>& dist, Linkage linkage);