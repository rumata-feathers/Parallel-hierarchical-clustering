#include "hac_parallel.hpp"

#include <algorithm>
#include <atomic>
#include <iostream>
#include <limits>
#include <thread>

#include "distance.hpp"

static double compute_cluster_dist(const std::vector<int> &cluster_i,
                                   const std::vector<int> &cluster_j,
                                   const std::vector<std::vector<double>> &dist,
                                   const std::vector<std::vector<double>> &data,
                                   Linkage linkage)
{
    if (cluster_i.empty() || cluster_j.empty())
        return std::numeric_limits<double>::infinity();

    switch (linkage)
    {
    case Linkage::SINGLE:
    {
        double d = std::numeric_limits<double>::infinity();
        for (auto a : cluster_i)
            for (auto b : cluster_j)
                d = std::min(d, dist[a][b]);
        return d;
    }
    case Linkage::CENTROID:
    {
        std::vector<double> centroid_i(data[0].size(), 0.0);
        std::vector<double> centroid_j(data[0].size(), 0.0);

        for (auto idx : cluster_i)
            for (size_t d = 0; d < data[0].size(); ++d)
                centroid_i[d] += data[idx][d];
        for (auto idx : cluster_j)
            for (size_t d = 0; d < data[0].size(); ++d)
                centroid_j[d] += data[idx][d];

        for (size_t d = 0; d < data[0].size(); ++d)
        {
            centroid_i[d] /= cluster_i.size();
            centroid_j[d] /= cluster_j.size();
        }

        return euclidean(centroid_i, centroid_j);
    }
    case Linkage::COMPLETE:
    {
        double d = 0.0;
        for (auto a : cluster_i)
            for (auto b : cluster_j)
                d = std::max(d, dist[a][b]);
        return d;
    }

    case Linkage::AVERAGE:
    {
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
        throw std::runtime_error("Unsupported linkage for parallel mode");
    }
}

/*
 * @param dist      : n x n distance matrix
 * @param data      : n x d raw data (for future linkages that need it)
 * @param linkage   : linkage criterion (only SINGLE supported for now)
 * @param n_threads : number of std::thread workers
 * @returns (n-1) x 4 linkage matrix: {cluster_i, cluster_j, distance, merged_size}
 */
std::vector<std::array<double, 4>> hac_parallel(
    const std::vector<std::vector<double>> &dist,
    const std::vector<std::vector<double>> &data,
    Linkage linkage, int n_threads)
{
    // same logic as with serial
    // just adding threads to compute distances in parallel
    // also I realised, that I need some comments to understand what im writing, so I'll do it here and eventually will comment all

    int n = static_cast<int>(dist.size());

    // initialy each node is its own cluster
    std::vector<std::vector<int>> cluster_nodes(n);
    for (int i = 0; i < n; ++i)
        cluster_nodes[i] = {i};

    // active[i] = false means that cluster i is already merged into another cluster
    std::vector<bool> active(n, true);

    // id[i] - id of the cluster at node n
    std::vector<int> id(n);
    for (int i = 0; i < n; ++i)
        id[i] = i;
    int next_id = n;

    std::vector<std::array<double, 4>> result;
    result.reserve(n - 1);

    // pre-allocated per thread outputs
    std::vector<double> t_best(n_threads);
    std::vector<int> t_ci(n_threads), t_cj(n_threads);
    std::vector<std::thread> threads(n_threads);
    // atomic boolean to track if linkage not supported
    std::atomic<bool> unsupported{false};

    for (int step = 0; step < n - 1; ++step)
    {
        int chunk = (n + n_threads - 1) / n_threads;

        for (int t = 0; t < n_threads; ++t)
        {
            int row_start = t * chunk;
            int row_end = std::min(row_start + chunk, n);

            threads[t] = std::thread([&, t, row_start, row_end]()
                                     {
                double best = std::numeric_limits<double>::infinity();
                int ci = -1, cj = -1;

                for (int i = row_start; i < row_end; ++i) {
                    if (!active[i]) continue;
                    for (int j = i + 1; j < n; ++j) {
                        if (!active[j]) continue;
                        // try-bloack with atomic boolean
                        // if fails -> mark boolean
                        try {
                            double d = compute_cluster_dist(
                                cluster_nodes[i], cluster_nodes[j], dist, data, linkage);
                            if (d < best) { best = d; ci = i; cj = j; }
                        } catch (const std::runtime_error& e) {
                            if (!unsupported.exchange(true))
                                std::cerr << "[hac_parallel] " << e.what()
                                          << " — skipping\n";
                            return;
                        }
                    }
                }

                t_best[t] = best;
                t_ci[t]   = ci;
                t_cj[t]   = cj; });
        }

        for (auto &th : threads)
            th.join();

        if (unsupported)
            return {};

        // find closest clusters
        double best_d = std::numeric_limits<double>::infinity();
        int ci = -1, cj = -1;
        for (int t = 0; t < n_threads; ++t)
        {
            if (t_best[t] < best_d)
            {
                best_d = t_best[t];
                ci = t_ci[t];
                cj = t_cj[t];
            }
        }
        // no clusters -> done
        if (ci == -1 || cj == -1)
            break;

        result.push_back({static_cast<double>(id[ci]),
                          static_cast<double>(id[cj]),
                          best_d,
                          static_cast<double>(cluster_nodes[ci].size() + cluster_nodes[cj].size())});

        // merge
        for (auto node : cluster_nodes[cj])
            cluster_nodes[ci].push_back(node);
        cluster_nodes[cj].clear();
        active[cj] = false;
        id[ci] = next_id++;
    }

    return result;
}
