#include "hac_cuda.hpp"
#include <cuda_runtime.h>
#include "distance.hpp"

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


__global__ void read_dist_kernel(
    const double* dist,
    const int* active,
    int n,
    double* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n || j >= n) return;
    if (i >= j || !active[i] || !active[j]) {
        out[i * n + j] = INF;
        return;
    }
    out[i * n + j] = dist[i * n + j];
}

__global__ void copy_column_kernel(
    double* dist,
    int n,
    int ci)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    dist[k * n + ci] = dist[ci * n + k];
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
            bool better = d < best ||
              (d == best && (i < bi || (i == bi && j < bj)));
            if (better) {
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
        if (tid < stride) {
        bool take = s_val[tid + stride] < s_val[tid] ||
                    (s_val[tid + stride] == s_val[tid] &&
                    (s_i[tid + stride] < s_i[tid] ||
                    (s_i[tid + stride] == s_i[tid] &&
                    s_j[tid + stride] < s_j[tid])));
        if (take) {
            s_val[tid] = s_val[tid + stride];
            s_i[tid]   = s_i[tid + stride];
            s_j[tid]   = s_j[tid + stride];
        }
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

std::vector<std::tuple<int, int, double, int>> hac_cuda( std::vector<std::vector<double>> dist, const std::vector<std::vector<double>>& data, Linkage linkage){


    const int n = static_cast<int>(dist.size());
    const int dim = data.empty() ? 0 : static_cast<int>(data[0].size());

    if (n == 0) return {};

    float total_copy_ms = 0.0f;
    float total_compute_ms = 0.0f;
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    float ms = 0.0f;

    double* d_dist     = nullptr;  
    double* d_D        = nullptr;  
    double* d_block_val = nullptr;
    int*    d_active   = nullptr;
    int*    d_block_i  = nullptr;
    int*    d_block_j  = nullptr;

    const int REDUCE_BLOCK = 256;
    const int REDUCE_GRID = std::max(1, (n * n + REDUCE_BLOCK - 1) / REDUCE_BLOCK);

    CUDA_CHECK(cudaMalloc(&d_D, (size_t)n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_active, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_val, REDUCE_GRID * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_i, REDUCE_GRID * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_j, REDUCE_GRID * sizeof(int)));

    

    std::vector<std::vector<int>> clusters(n);
    for (int i = 0; i < n; ++i) clusters[i] = {i};
    std::vector<bool> active(n, true);
    std::vector<int> id(n);
    for (int i = 0; i < n; ++i) id[i] = i;
    int next_id = n;

    std::vector<double> flat_dist = flatten_matrix(dist);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double d = compute_cluster_dist(clusters[i], clusters[j], dist, data, linkage);
            flat_dist[i * n + j] = d;
            flat_dist[j * n + i] = d;
        }
    }
    CUDA_CHECK(cudaMalloc(&d_dist, (size_t)n * n * sizeof(double)));
    CUDA_CHECK(cudaEventRecord(ev_start));
    CUDA_CHECK(cudaMemcpy(d_dist, flat_dist.data(), (size_t)n * n * sizeof(double), cudaMemcpyHostToDevice));
        


    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    cudaEventElapsedTime(&ms, ev_start, ev_stop);
    total_copy_ms += ms;


    bool centroid_based = linkage == Linkage::CENTROID || linkage == Linkage::WARD || linkage == Linkage::MEDIAN;

    double* d_centroids = nullptr;
    int* d_cluster_sizes_centroid = nullptr;


    std::vector<double> centroids((size_t)n * dim, 0.0);
    if (centroid_based) {
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < dim; ++k)
                centroids[i * dim + k] = data[i][k];
        CUDA_CHECK(cudaMalloc(&d_centroids, (size_t)n * dim * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cluster_sizes_centroid, n * sizeof(int)));
    }


    std::vector<std::tuple<int, int, double, int>> result;
    result.reserve(n - 1);

    dim3 block2d(16, 16);
    dim3 grid2d((n + 15) / 16, (n + 15) / 16);

    for (int step = 0; step < n - 1; ++step) {
        std::vector<int> active_int(n);
        for (int i = 0; i < n; ++i)
            active_int[i] = active[i] ? 1 : 0;

        CUDA_CHECK(cudaEventRecord(ev_start));
        CUDA_CHECK(cudaMemcpy(d_active, active_int.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        total_copy_ms += ms;

        CUDA_CHECK(cudaEventRecord(ev_start));
        read_dist_kernel<<<grid2d, block2d>>>(d_dist, d_active, n, d_D);
        CUDA_CHECK(cudaGetLastError());

        size_t shared_mem = REDUCE_BLOCK * sizeof(double) + 2 * REDUCE_BLOCK * sizeof(int);
        find_min_kernel<<<REDUCE_GRID, REDUCE_BLOCK, shared_mem>>>(d_D, n, d_block_val, d_block_i, d_block_j);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        total_compute_ms += ms;

        std::vector<double> h_block_val(REDUCE_GRID);
        std::vector<int> h_block_i(REDUCE_GRID), h_block_j(REDUCE_GRID);
        CUDA_CHECK(cudaMemcpy(h_block_val.data(), d_block_val, REDUCE_GRID * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_block_i.data(), d_block_i, REDUCE_GRID * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_block_j.data(), d_block_j, REDUCE_GRID * sizeof(int), cudaMemcpyDeviceToHost));

        double best_d = INF;
        int ci = -1, cj = -1;
        for (int b = 0; b < REDUCE_GRID; ++b) {
            bool take = h_block_val[b] < best_d ||
                        (h_block_val[b] == best_d &&
                        (h_block_i[b] < ci ||
                        (h_block_i[b] == ci && h_block_j[b] < cj)));
            if (take) {
                best_d = h_block_val[b];
                ci = h_block_i[b];
                cj = h_block_j[b];
            }
        }

        if (ci == -1 || cj == -1 || best_d >= INF / 2) break;

        std::vector<int> ci_before = clusters[ci];
        std::vector<int> cj_before = clusters[cj];

        result.emplace_back(id[ci], id[cj], best_d,
                            static_cast<int>(clusters[ci].size() + clusters[cj].size()));

        for (int p : clusters[cj]) clusters[ci].push_back(p);
        clusters[cj].clear();
        active[cj] = false;
        id[ci] = next_id++;

        for (int k = 0; k < n; ++k) {
            if (!active[k] || k == ci) continue;
            double d = compute_cluster_dist(clusters[ci], clusters[k], dist, data, linkage);
            flat_dist[ci * n + k] = d;
            flat_dist[k * n + ci] = d;
        }

        if (centroid_based) {

            for (int k = 0; k < dim; ++k) centroids[ci * dim + k] = 0.0;
            for (int p : clusters[ci])
                for (int k = 0; k < dim; ++k)
                    centroids[ci * dim + k] += data[p][k];
            for (int k = 0; k < dim; ++k)
                centroids[ci * dim + k] /= clusters[ci].size();
        }


        for (int k = 0; k < n; ++k) {
            flat_dist[cj * n + k] = INF;
            flat_dist[k * n + cj] = INF;
        }

        CUDA_CHECK(cudaEventRecord(ev_start));
        CUDA_CHECK(cudaMemcpy(d_dist + ci * n, flat_dist.data() + ci * n, n * sizeof(double), cudaMemcpyHostToDevice));

        int col_grid = (n + 255) / 256;
        copy_column_kernel<<<col_grid, 256>>>(d_dist, n, ci);
        CUDA_CHECK(cudaMemcpy(d_dist + cj * n, flat_dist.data() + cj * n, n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        total_copy_ms += ms;
    }

    std::cerr << "[cuda] total copy:    " << total_copy_ms << " ms\n"
              << "[cuda] total compute: " << total_compute_ms << " ms\n"
              << "[cuda] copy fraction: "
              << 100.0f * total_copy_ms / (total_copy_ms + total_compute_ms)
              << "%\n";

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    cudaFree(d_dist);
    cudaFree(d_D);
    cudaFree(d_active);
    cudaFree(d_block_val);
    cudaFree(d_block_i);
    cudaFree(d_block_j);
    cudaFree(d_centroids);
    cudaFree(d_cluster_sizes_centroid);

    return result;
}



