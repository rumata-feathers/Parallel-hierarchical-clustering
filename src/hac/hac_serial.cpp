#include "hac_serial.hpp"

#include <algorithm>
#include <limits>
#include <iostream>
#include "distance.hpp"

/*
 * @param dist : n x n distance matrix (passed by value — modified in place)
 * @param linkage : linkage criterion
 * @returns (n-1) x 4 linkage matrix: {cluster_i, cluster_j, distance,
 * merged_size}
 */
std::vector<std::tuple<int, int, double, int>> hac_serial(
    const std::vector<std::vector<double>> &dist,
    const std::vector<std::vector<double>> &df, Linkage linkage)
{
    int n = static_cast<int>(dist.size());

    std::vector<std::vector<int>> cluster_nodes(n);
    for (int i = 0; i < n; ++i)
        cluster_nodes[i] = {i};

    std::vector<int> id(n);
    for (int i = 0; i < n; ++i)
        id[i] = i;

    std::vector<std::vector<double>> dist_matrix(n,
                                                 std::vector<double>(n, std::numeric_limits<double>::infinity()));
    for (int i = 0; i < n; ++i)
    {
        dist_matrix[i][i] = 0.0;
        for (int j = i + 1; j < n; ++j)
        {
            double d = compute_cluster_dist(
                cluster_nodes[i], cluster_nodes[j], dist, df, linkage);
            dist_matrix[i][j] = d;
            dist_matrix[j][i] = d;
        }
    }

    std::vector<std::tuple<int, int, double, int>> result;
    result.reserve(n - 1);

    // for scipy dendrogram
    int next_id = n;
    for (int step = 0; step < n - 1; ++step)
    {
        // find clusters to merge
        double best_d = std::numeric_limits<double>::infinity();
        int cluster_i = -1, cluster_j = -1;
        for (int i = 0; i < n; ++i)
        {
            if (cluster_nodes[i].empty())
                continue;
            for (int j = i + 1; j < n; ++j)
            {
                if (cluster_nodes[j].empty())
                    continue;
                if (dist_matrix[i][j] < best_d)
                {
                    best_d = dist_matrix[i][j];
                    cluster_i = i;
                    cluster_j = j;
                }
            }
        }

        if (cluster_i == -1 || cluster_j == -1)
            break;

        result.emplace_back(id[cluster_i], id[cluster_j], best_d, static_cast<int>(cluster_nodes[cluster_i].size() + cluster_nodes[cluster_j].size()));

        // merge
        for (auto &node : cluster_nodes[cluster_j])
            cluster_nodes[cluster_i].push_back(node);
        cluster_nodes[cluster_j].clear();
        id[cluster_i] = next_id++;

        update_distances(cluster_i, cluster_j, cluster_nodes, dist, dist_matrix, df, linkage);
    }

    return result;
}
