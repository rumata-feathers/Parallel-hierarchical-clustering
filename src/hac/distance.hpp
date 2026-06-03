#pragma once
#include <cmath>
#include <vector>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include "linkage.hpp" 

// TODO: add parallel version
inline double euclidean(const std::vector<double>& a,
                        const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t k = 0; k < a.size(); ++k) {
        double d = a[k] - b[k];
        sum += d * d;
    }
    return std::sqrt(sum);
}
// TODO: add parallel version
inline std::vector<std::vector<double>> compute_distance_matrix(
    const std::vector<std::vector<double>>& points) {
    size_t n = points.size();
    std::vector<std::vector<double>> dist(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            dist[i][j] = dist[j][i] = euclidean(points[i], points[j]);
    return dist;
}


/*
 *   @param cluster_i, cluster_j : two clusters to merge
 *   @param dist : n x n distance matrix
 *   @param linkage : linkage criterion
 *   @returns distance between the new cluster and any other cluster (depends on
 the linkage) cluster
*/
inline double compute_cluster_dist(
    const std::vector<int>& cluster_i,
    const std::vector<int>& cluster_j,
    const std::vector<std::vector<double>>& dist,
    const std::vector<std::vector<double>>& data,
    Linkage linkage)
{
    if (cluster_i.empty() || cluster_j.empty())
        return std::numeric_limits<double>::infinity();

    switch (linkage) {
    case Linkage::SINGLE: {
        double d = std::numeric_limits<double>::infinity();
        for (auto a : cluster_i)
            for (auto b : cluster_j)
                d = std::min(d, dist[a][b]);
        return d;
    }
    case Linkage::COMPLETE: {
        double d = 0.0;
        for (auto a : cluster_i)
            for (auto b : cluster_j)
                d = std::max(d, dist[a][b]);
        return d;
    }
    case Linkage::AVERAGE: {
        double sum = 0.0; int count = 0;
        for (auto a : cluster_i)
            for (auto b : cluster_j) { sum += dist[a][b]; ++count; }
        return sum / count;
    }
    case Linkage::WARD: {
        size_t dims = data[0].size();
        std::vector<double> ci(dims, 0.0), cj(dims, 0.0);
        for (auto a : cluster_i) for (size_t k = 0; k < dims; ++k) ci[k] += data[a][k];
        for (auto b : cluster_j) for (size_t k = 0; k < dims; ++k) cj[k] += data[b][k];
        double sq = 0.0;
        for (size_t k = 0; k < dims; ++k) {
            double diff = ci[k]/cluster_i.size() - cj[k]/cluster_j.size();
            sq += diff * diff;
        }
        double ni = cluster_i.size(), nj = cluster_j.size();
        return std::sqrt(ni * nj / (ni + nj) * sq);
    }
    case Linkage::CENTROID:
    case Linkage::MEDIAN: {
        size_t dims = data[0].size();
        std::vector<double> ci(dims, 0.0), cj(dims, 0.0);
        for (auto a : cluster_i) for (size_t k = 0; k < dims; ++k) ci[k] += data[a][k];
        for (auto b : cluster_j) for (size_t k = 0; k < dims; ++k) cj[k] += data[b][k];
        for (size_t k = 0; k < dims; ++k) {
            ci[k] /= cluster_i.size();
            cj[k] /= cluster_j.size();
        }
        return euclidean(ci, cj);
    }
    default:
        throw std::runtime_error("Unsupported linkage");
    }
}