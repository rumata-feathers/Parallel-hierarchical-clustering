#include "hac_serial.hpp"

#include <algorithm>
#include <limits>
#include <iostream>
#include "distance.hpp"

/*
 *   @param cluster_i, cluster_j : two clusters to merge
 *   @param dist : n x n distance matrix
 *   @param linkage : linkage criterion
 *   @returns distance between the new cluster and any other cluster (depends on
 the linkage) cluster
*/
double compute_cluster_dist(const std::vector<int>& cluster_i,
                            const std::vector<int>& cluster_j,
                            const std::vector<std::vector<double>>& dist,
                            const std::vector<std::vector<double>>& data,
                            Linkage linkage) {
    double d = 0.0;
    if (cluster_i.empty() || cluster_j.empty())
        return std::numeric_limits<double>::infinity();

    switch (linkage) {
        case Linkage::SINGLE: {
            d = std::numeric_limits<double>::infinity();
            for (auto& a : cluster_i)
                for (auto& b : cluster_j) d = std::min(d, dist[a][b]);
            break;
        }
        case Linkage::CENTROID: {
            std::vector<double> centroid_i(data[0].size(), 0.0);
            std::vector<double> centroid_j(data[0].size(), 0.0);
            for (auto& a : cluster_i)
                for (size_t k = 0; k < centroid_i.size(); ++k)
                    centroid_i[k] += data[a][k];
            for (auto& b : cluster_j)
                for (size_t k = 0; k < centroid_j.size(); ++k)
                    centroid_j[k] += data[b][k];
            for (size_t k = 0; k < centroid_i.size(); ++k) {
                centroid_i[k] /= cluster_i.size();
                centroid_j[k] /= cluster_j.size();
            }

            d = euclidean(centroid_i, centroid_j);
            break;
        }
        case Linkage::COMPLETE: {
            double d = 0.0;
            for (auto a : cluster_i)
                for (auto b : cluster_j)
                    d = std::max(d, dist[a][b]);
            return d;
        }
        case Linkage::AVERAGE: {
            double sum = 0.0;
            int count = 0;
            for (auto a : cluster_i)
                for (auto b : cluster_j) {
                    sum += dist[a][b];
                    ++count;
                }
            return sum / count;
        }
        default:
            throw std::runtime_error("Unsupported linkage for serial mode");
    }

    return d;
}

/*
 * @param dist : n x n distance matrix (passed by value — modified in place)
 * @param linkage : linkage criterion
 * @returns (n-1) x 4 linkage matrix: {cluster_i, cluster_j, distance,
 * merged_size}
 */
std::vector<std::array<double, 4>> hac_serial(
    const std::vector<std::vector<double>>& dist,
    const std::vector<std::vector<double>>& df, Linkage linkage) {
    int n = static_cast<int>(dist.size());

    std::vector<std::vector<int>> cluster_nodes(n);
    for (int i = 0; i < n; ++i) cluster_nodes[i] = {i};

    std::vector<int> id(n);
    for (int i = 0; i < n; ++i) id[i] = i;

    std::vector<std::array<double, 4>> result;
    result.reserve(n - 1);

    // for scipy dendrogram
    int next_id = n;
    for (int step = 0; step < n - 1; ++step) {
        // find clusters to merge
        double best_d = std::numeric_limits<double>::infinity();
        int cluster_i = -1, cluster_j = -1;
        for (int i = 0; i < n; ++i) {
            if (cluster_nodes[i].empty()) continue;
            for (int j = i + 1; j < n; ++j) {
                if (cluster_nodes[j].empty()) continue;
                try {
                    double d = compute_cluster_dist(
                        cluster_nodes[i], cluster_nodes[j], dist, df, linkage);
                    if (d < best_d) {
                        best_d = d;
                        cluster_i = i;
                        cluster_j = j;
                    }
                } catch(const std::runtime_error& e) {
                    std::cerr << "[hac_serial] " << e.what() << " — skipping\n";
                    return {};
                }
            }
        }

        if (cluster_i == -1 || cluster_j == -1) break;

        result.push_back(
            {static_cast<double>(id[cluster_i]),
             static_cast<double>(id[cluster_j]), best_d,
             static_cast<double>(cluster_nodes[cluster_i].size() +
                                 cluster_nodes[cluster_j].size())});

        // merge
        for (auto& node_ind : cluster_nodes[cluster_j])
            cluster_nodes[cluster_i].push_back(node_ind);
        cluster_nodes[cluster_j].clear();
        id[cluster_i] = next_id++;
    }

    return result;
}
