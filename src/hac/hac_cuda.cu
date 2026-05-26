#include "hac_cuda.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "cuda error: " << cudaGetErrorString(err)           \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";     \
            throw std::runtime_error("cuda error");                         \
        }                                                                    \
    } while (0)

static constexpr double INF = 1e300;


__global__ void pairwise_point_linkage_kernel(
    const double* point_dist,
    const int* active,
    const int* cluster_start,
    const int* cluster_size,
    const int* cluster_points,
    int n,
    int linkage_type,
    double* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n || j >= n) return;
    if (i >= j || !active[i] || !active[j]) {
        out[i * n + j] = INF;
        return;
    }

    int si = cluster_start[i];
    int ei = si + cluster_size[i];
    int sj = cluster_start[j];
    int ej = sj + cluster_size[j];

    double value;

    if (linkage_type == 0) { 
        // single
        value = INF;
        for (int a = si; a < ei; ++a) {
            int pa = cluster_points[a];
            for (int b = sj; b < ej; ++b) {
                int pb = cluster_points[b];
                value = fmin(value, point_dist[pa * n + pb]);
            }
        }
    } else if (linkage_type == 1) {
        // complete
        value = 0.0;
        for (int a = si; a < ei; ++a) {
            int pa = cluster_points[a];
            for (int b = sj; b < ej; ++b) {
                int pb = cluster_points[b];
                value = fmax(value, point_dist[pa * n + pb]);
            }
        }
    } else {
        // average
        double sum = 0.0;
        int count = 0;
        for (int a = si; a < ei; ++a) {
            int pa = cluster_points[a];
            for (int b = sj; b < ej; ++b) {
                int pb = cluster_points[b];
                sum += point_dist[pa * n + pb];
                ++count;
            }
        }
        if (count > 0) {
            value = sum / count;
        } else {
            value = INF;
        }
    }

    out[i * n + j] = value;
}


// centroid/ward/median

__global__ void pairwise_centroid_kernel( const double* centroids, const int* active, const int* cluster_sizes, int n, int dim, int linkage_type, double* out){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n || j >= n) return;
    if (i >= j || !active[i] || !active[j]) {
        out[i * n + j] = INF;
        return;
    }

    double sum = 0.0;
    for (int k = 0; k < dim; ++k) {
        double diff = centroids[i * dim + k] - centroids[j * dim + k];
        sum += diff * diff;
    }
    if (linkage_type == 1) {
        // ward: sqrt(ni * nj / (ni + nj) * ||ci - cj||^2)
        double ni = static_cast<double>(cluster_sizes[i]);
        double nj = static_cast<double>(cluster_sizes[j]);
        out[i * n + j] = sqrt(ni * nj / (ni + nj) * sum);
    } else {
    out[i * n + j] = sqrt(sum);
    }
}

__global__ void find_min_kernel(
    const double* D,
    int n,
    double* block_val,
    int* block_i,
    int* block_j) {
    extern __shared__ unsigned char smem[];

    double* s_val = reinterpret_cast<double*>(smem);
    int* s_i = reinterpret_cast<int*>(s_val + blockDim.x);
    int* s_j = s_i + blockDim.x;

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int total = n * n;

    double best = INF;
    int bi = -1;
    int bj = -1;

    for (int idx = gid; idx < total; idx += blockDim.x * gridDim.x) {
        int i = idx / n;
        int j = idx % n;

        if (i < j) {
            double d = D[idx];
            if (d < best) {
                best = d;
                bi = i;
                bj = j;
            }
        }
    }

    s_val[tid] = best;
    s_i[tid] = bi;
    s_j[tid] = bj;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride && s_val[tid + stride] < s_val[tid]) {
            s_val[tid] = s_val[tid + stride];
            s_i[tid] = s_i[tid + stride];
            s_j[tid] = s_j[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_val[blockIdx.x] = s_val[0];
        block_i[blockIdx.x] = s_i[0];
        block_j[blockIdx.x] = s_j[0];
    }
}

static std::vector<double> flatten_matrix( const std::vector<std::vector<double>>& mat){
    int n = static_cast<int>(mat.size());
    std::vector<double> flat((size_t)n * n);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            flat[(size_t)i * n + j] = mat[i][j];

    return flat;
}

std::vector<std::array<double, 4>> hac_cuda( std::vector<std::vector<double>> dist, const std::vector<std::vector<double>>& data, Linkage linkage){


    const int n = static_cast<int>(dist.size());
    const int dim = data.empty() ? 0 : static_cast<int>(data[0].size());

    if (n == 0) return {};

    bool point_based =
        linkage == Linkage::SINGLE ||
        linkage == Linkage::COMPLETE ||
        linkage == Linkage::AVERAGE;

    bool centroid_based = linkage == Linkage::CENTROID ||
                          linkage == Linkage::WARD ||
                          linkage == Linkage::MEDIAN;

    double* d_point_dist = nullptr;
    double* d_D = nullptr;
    double* d_centroids = nullptr;
    double* d_block_val = nullptr;

    int* d_active = nullptr;
    int* d_cluster_start = nullptr;
    int* d_cluster_size = nullptr;
    int* d_cluster_points = nullptr;
    int* d_block_i = nullptr;
    int* d_block_j = nullptr;
    int* d_cluster_sizes_centroid = nullptr;
    const int REDUCE_BLOCK = 256;
    const int REDUCE_GRID = std::max(1, (n * n + REDUCE_BLOCK - 1) / REDUCE_BLOCK);

    CUDA_CHECK(cudaMalloc(&d_D, (size_t)n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_active, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_val, REDUCE_GRID * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_i, REDUCE_GRID * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_j, REDUCE_GRID * sizeof(int)));

    if (point_based) {
        std::vector<double> flat_dist = flatten_matrix(dist);
        CUDA_CHECK(cudaMalloc(&d_point_dist, (size_t)n * n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_point_dist, flat_dist.data(), (size_t)n * n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_cluster_start, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cluster_size, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cluster_points, n * sizeof(int)));
    }

    if (centroid_based) {
        CUDA_CHECK(cudaMalloc(&d_centroids, (size_t)n * dim * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cluster_sizes_centroid, n * sizeof(int)));
    }

    std::vector<std::vector<int>> clusters(n);
    for (int i = 0; i < n; ++i) clusters[i] = {i};

    std::vector<bool> active(n, true);

    std::vector<int> id(n);
    for (int i = 0; i < n; ++i) id[i] = i;

    int next_id = n;

    std::vector<double> centroids((size_t)n * dim, 0.0);
    if (centroid_based) {
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < dim; ++k)
                centroids[i * dim + k] = data[i][k];
    }

    std::vector<std::array<double, 4>> result;
    result.reserve(n - 1);

    for (int step = 0; step < n - 1; ++step) {
        std::vector<int> active_int(n);
        for (int i = 0; i < n; ++i)
            active_int[i] = active[i] ? 1 : 0;

        CUDA_CHECK(cudaMemcpy(d_active, active_int.data(), n * sizeof(int), cudaMemcpyHostToDevice));

        if (point_based) {
            std::vector<int> cluster_start(n);
            std::vector<int> cluster_size(n);
            std::vector<int> cluster_points;
            cluster_points.reserve(n);

            int offset = 0;
            for (int i = 0; i < n; ++i) {
                cluster_start[i] = offset;
                cluster_size[i] = static_cast<int>(clusters[i].size());

                for (int p : clusters[i])
                    cluster_points.push_back(p);

                offset += cluster_size[i];
            }

            CUDA_CHECK(cudaMemcpy(d_cluster_start, cluster_start.data(), n * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_cluster_size, cluster_size.data(), n * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_cluster_points, cluster_points.data(), cluster_points.size() * sizeof(int), cudaMemcpyHostToDevice));
        }

        if (centroid_based) {
            CUDA_CHECK(cudaMemcpy(d_centroids, centroids.data(), (size_t)n * dim * sizeof(double),cudaMemcpyHostToDevice));
            std::vector<int> csizes(n);
            for (int i = 0; i < n; ++i)
                csizes[i] = static_cast<int>(clusters[i].size());
            CUDA_CHECK(cudaMemcpy(d_cluster_sizes_centroid, csizes.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        }

        dim3 block2d(16, 16);
        dim3 grid2d((n + 15) / 16, (n + 15) / 16);

        if (point_based) {
            int linkage_type = 0;
            if (linkage == Linkage::COMPLETE) linkage_type = 1;
            if (linkage == Linkage::AVERAGE) linkage_type = 2;

            pairwise_point_linkage_kernel<<<grid2d, block2d>>>(
                d_point_dist,
                d_active,
                d_cluster_start,
                d_cluster_size,
                d_cluster_points,
                n,
                linkage_type,
                d_D);
        } else {
            int centroid_linkage_type = 0;
            if (linkage == Linkage::WARD) centroid_linkage_type = 1;
            if (linkage == Linkage::MEDIAN) centroid_linkage_type = 2;
            pairwise_centroid_kernel<<<grid2d, block2d>>>(
                d_centroids,
                d_active,
                d_cluster_sizes_centroid,
                n,
                dim,
                centroid_linkage_type,
                d_D);
        }

        CUDA_CHECK(cudaGetLastError());

        size_t shared_mem = REDUCE_BLOCK * sizeof(double) + 2 * REDUCE_BLOCK * sizeof(int);

        find_min_kernel<<<REDUCE_GRID, REDUCE_BLOCK, shared_mem>>>(
            d_D,
            n,
            d_block_val,
            d_block_i,
            d_block_j);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> h_block_val(REDUCE_GRID);
        std::vector<int> h_block_i(REDUCE_GRID);
        std::vector<int> h_block_j(REDUCE_GRID);

        CUDA_CHECK(cudaMemcpy(h_block_val.data(), d_block_val, REDUCE_GRID * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_block_i.data(), d_block_i, REDUCE_GRID * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_block_j.data(), d_block_j, REDUCE_GRID * sizeof(int), cudaMemcpyDeviceToHost));

        double best_d = INF;
        int ci = -1;
        int cj = -1;

        for (int b = 0; b < REDUCE_GRID; ++b) {
            if (h_block_val[b] < best_d) {
                best_d = h_block_val[b];
                ci = h_block_i[b];
                cj = h_block_j[b];
            }
        }

        if (ci == -1 || cj == -1 || best_d >= INF / 2){
            break;
        }

        result.push_back({
            static_cast<double>(id[ci]),
            static_cast<double>(id[cj]),
            best_d,
            static_cast<double>(clusters[ci].size() + clusters[cj].size())
        });

        if (centroid_based) {
            double ni = static_cast<double>(clusters[ci].size());
            double nj = static_cast<double>(clusters[cj].size());
            double total = ni + nj;

            for (int k = 0; k < dim; ++k) {
                centroids[ci * dim + k] = (ni * centroids[ci * dim + k] + nj * centroids[cj * dim + k]) / total;
            }
        }

        for (int p : clusters[cj]){
            clusters[ci].push_back(p);
        }
        clusters[cj].clear();
        active[cj] = false;
        id[ci] = next_id++;
    }

    cudaFree(d_point_dist);
    cudaFree(d_D);
    cudaFree(d_centroids);
    cudaFree(d_active);
    cudaFree(d_cluster_start);
    cudaFree(d_cluster_size);
    cudaFree(d_cluster_points);
    cudaFree(d_cluster_sizes_centroid);
    cudaFree(d_block_val);
    cudaFree(d_block_i);
    cudaFree(d_block_j);
    return result;
}