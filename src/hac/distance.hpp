#pragma once
#include <cmath>
#include <vector>

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
