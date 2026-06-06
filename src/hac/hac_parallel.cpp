#include "hac_parallel.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <limits>
#include <mutex>
#include <thread>

#include "distance.hpp"

/*
 * @param dist      : n x n distance matrix
 * @param data      : n x d raw data (for future linkages that need it)
 * @param linkage   : linkage criterion (only SINGLE supported for now)
 * @param n_threads : number of std::thread workers
 * @returns (n-1) x 4 linkage matrix: {cluster_i, cluster_j, distance, merged_size}
 */
std::vector<std::tuple<int, int, double, int>> hac_parallel(
    const std::vector<std::vector<double>> &dist,
    const std::vector<std::vector<double>> &data,
    Linkage linkage, int n_threads)
{
    // same logic as with serial
    // just adding threads to compute distances in parallel
    // also I realised, that I need some comments to understand what im writing, so I'll do it here and eventually will comment all

    int n = static_cast<int>(dist.size());
    // mutable copy of the distance matrix 
    //the next step only scans cached values instead of recomputing everything

    
    std::vector<std::vector<double>> dist_matrix = dist;
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

    std::vector<std::tuple<int, int, double, int>> result;
    result.reserve(n - 1);

    // pre-allocated per thread outputs
    std::vector<double> t_best(n_threads);
    std::vector<int> t_ci(n_threads), t_cj(n_threads);
    std::vector<std::thread> threads(n_threads);
    std::atomic<int> next_row{0};


    for (int step = 0; step < n - 1; ++step)
    {
        next_row = 0;

        for (int t = 0; t < n_threads; ++t)
        {
            threads[t] = std::thread([&, t]()
            {
                double best = std::numeric_limits<double>::infinity();
                int ci = -1, cj = -1;

                int i;

                while ((i = next_row.fetch_add(1)) < n) {
                    if (!active[i]) continue;
                    for (int k = 1; k <= n / 2; ++k) {
                        int j = (i + k) % n;
                        if (!active[j]) continue;
                        int a = std::min(i, j);
                        int b = std::max(i, j);
                        if (a == b) continue;
                        // read cached distance no recompute here
                        double d = dist_matrix[a][b];
                        if (d < best) { best = d; ci = a; cj = b; }
                    }
                }

                t_best[t] = best;
                t_ci[t]   = ci;
                t_cj[t]   = cj;
            });
        }

        for (auto &th : threads)
            th.join();


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

        result.emplace_back(id[ci], id[cj], best_d, static_cast<int>(cluster_nodes[ci].size() + cluster_nodes[cj].size()));

        // merge
        for (int node : cluster_nodes[cj])
            cluster_nodes[ci].push_back(node);
        cluster_nodes[cj].clear();
        active[cj] = false;
        id[ci] = next_id++;
        
        for (int k = 0; k < n; ++k)
            dist_matrix[cj][k] = dist_matrix[k][cj] = std::numeric_limits<double>::infinity();
        for (int k = 0; k < n; ++k) {
            if (!active[k] || k == ci) continue;
            double d = compute_cluster_dist(
                cluster_nodes[ci], cluster_nodes[k],
                dist_matrix, data, linkage);
            dist_matrix[ci][k] = d;
            dist_matrix[k][ci] = d;
        }
    }

    return result;
}
