#pragma once
#include <array>
#include <vector>
#include "linkage.hpp"

// Returns (n-1) x 4 linkage matrics: {cluster_i, cluster_j, distance, merged_size}
std::vector<std::array<double, 4>> hac_serial(
    std::vector<std::vector<double>> dist,
    Linkage linkage);