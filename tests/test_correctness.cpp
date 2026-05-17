#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "hac/distance.hpp"
#include "hac/hac_parallel.hpp"
#include "hac/hac_serial.hpp"
#include "hac/linkage.hpp"

#ifdef ENABLE_CUDA
#include "hac/hac_cuda.hpp"
#endif

// ---------------------------------------------------------------
// Minimal test harness
// ---------------------------------------------------------------

static int n_failed = 0;

#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::cerr << "  FAIL: " << (msg) << "  (line " << __LINE__    \
                      << ")\n";                                            \
            ++n_failed;                                                    \
        }                                                                  \
    } while (0)

static bool matrices_equal(const std::vector<std::array<double, 4>>& a,
                            const std::vector<std::array<double, 4>>& b,
                            double tol = 1e-9) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i][0] != b[i][0] || a[i][1] != b[i][1]) return false;
        if (std::abs(a[i][2] - b[i][2]) > tol)         return false;
        if (a[i][3] != b[i][3])                         return false;
    }
    return true;
}

// ---------------------------------------------------------------
// Test datasets
// ---------------------------------------------------------------

/*
 * Six points on a line forming two tight clusters far apart.
 * All pairwise distances are distinct so there are no tie-breaking ambiguities.
 *
 *   Cluster L: x = 0, 1, 4      (max within-dist = 4)
 *   Cluster R: x = 20, 22, 25   (max within-dist = 5)
 *   Between clusters: min dist = 16  (point 4 to point 2)
 *
 * Expected single-linkage merge order:
 *   step 1 : (0,1)  d=1,  size 2  → id 6
 *   step 2 : (3,4)  d=2,  size 2  → id 7
 *   step 3 : (6,2)  d=3,  size 3  → id 8
 *   step 4 : (7,5)  d=3,  size 3  → id 9
 *   step 5 : (8,9)  d=16, size 6  → id 10
 */
static std::vector<std::vector<double>> make_two_cluster_data() {
    return {{0, 0}, {1, 0}, {4, 0}, {20, 0}, {22, 0}, {25, 0}};
}

static std::vector<std::array<double, 4>> expected_two_cluster() {
    return {
        {0, 1,  1.0, 2},
        {3, 4,  2.0, 2},
        {6, 2,  3.0, 3},
        {7, 5,  3.0, 3},
        {8, 9, 16.0, 6},
    };
}

/*
 * Nine points: 3 well-separated 2-D clusters (3 points each).
 * Used to cross-check that parallel produces the same merge tree
 * as serial on a 2-D dataset.
 *
 *   A: (0,0) (1,0) (0,1)      within-max ≈ 1.41
 *   B: (50,0) (51,0) (50,1)   within-max ≈ 1.41
 *   C: (0,50) (1,50) (0,51)   within-max ≈ 1.41
 *   Between clusters: all distances ≥ 49
 */
static std::vector<std::vector<double>> make_three_cluster_data() {
    return {
        {0, 0}, {1, 0}, {0, 1},
        {50, 0}, {51, 0}, {50, 1},
        {0, 50}, {1, 50}, {0, 51},
    };
}

// ---------------------------------------------------------------
// Tests
// ---------------------------------------------------------------

static void test_serial_expected_output() {
    std::cout << "[serial] expected output on two-cluster dataset\n";
    auto data = make_two_cluster_data();
    auto dist = compute_distance_matrix(data);
    auto result = hac_serial(dist, data, Linkage::SINGLE);
    auto expected = expected_two_cluster();

    CHECK(result.size() == expected.size(), "result row count");
    if (matrices_equal(result, expected))
        std::cout << "  PASS\n";
    else {
        ++n_failed;
        std::cerr << "  FAIL: linkage matrix mismatch\n";
        for (auto& r : result)
            std::cerr << "    " << r[0] << " " << r[1] << " " << r[2] << " " << r[3] << "\n";
    }
}

static void test_parallel_matches_serial(int n_threads) {
    std::cout << "[parallel " << n_threads << "t] matches serial (two-cluster dataset)\n";
    auto data = make_two_cluster_data();
    auto dist = compute_distance_matrix(data);
    auto serial   = hac_serial(dist, data, Linkage::SINGLE);
    auto parallel = hac_parallel(dist, data, Linkage::SINGLE, n_threads);

    CHECK(!serial.empty(),   "serial produced result");
    CHECK(!parallel.empty(), "parallel produced result");
    CHECK(matrices_equal(serial, parallel),
          "parallel result matches serial (n_threads=" + std::to_string(n_threads) + ")");
    if (n_failed == 0) std::cout << "  PASS\n";
}

static void test_parallel_three_clusters(int n_threads) {
    std::cout << "[parallel " << n_threads << "t] matches serial (three-cluster 2D dataset)\n";
    auto data = make_three_cluster_data();
    auto dist = compute_distance_matrix(data);
    auto serial   = hac_serial(dist, data, Linkage::SINGLE);
    auto parallel = hac_parallel(dist, data, Linkage::SINGLE, n_threads);

    CHECK(!serial.empty(),   "serial produced result");
    CHECK(!parallel.empty(), "parallel produced result");
    CHECK(matrices_equal(serial, parallel),
          "parallel result matches serial (n_threads=" + std::to_string(n_threads) + ")");
    if (n_failed == 0) std::cout << "  PASS\n";
}

static void test_parallel_unsupported_linkage_does_not_crash() {
    std::cout << "[parallel] unsupported linkage returns empty without crash\n";
    auto data = make_two_cluster_data();
    auto dist = compute_distance_matrix(data);
    // COMPLETE is not implemented in parallel — must return {} gracefully
    auto result = hac_parallel(dist, data, Linkage::COMPLETE, 2);
    CHECK(result.empty(), "unsupported linkage returns empty result");
    if (n_failed == 0) std::cout << "  PASS\n";
}

#ifdef ENABLE_CUDA
static void test_cuda_matches_serial() {
    std::cout << "[cuda] matches serial (two-cluster dataset)\n";
    auto data = make_two_cluster_data();
    auto dist = compute_distance_matrix(data);
    auto serial = hac_serial(dist, data, Linkage::SINGLE);
    auto cuda   = hac_cuda(dist, Linkage::SINGLE);

    CHECK(!serial.empty(), "serial produced result");
    CHECK(!cuda.empty(),   "cuda produced result");
    CHECK(matrices_equal(serial, cuda), "cuda result matches serial");
    if (n_failed == 0) std::cout << "  PASS\n";
}

static void test_cuda_three_clusters() {
    std::cout << "[cuda] matches serial (three-cluster 2D dataset)\n";
    auto data = make_three_cluster_data();
    auto dist = compute_distance_matrix(data);
    auto serial = hac_serial(dist, data, Linkage::SINGLE);
    auto cuda   = hac_cuda(dist, Linkage::SINGLE);

    CHECK(!serial.empty(), "serial produced result");
    CHECK(!cuda.empty(),   "cuda produced result");
    CHECK(matrices_equal(serial, cuda), "cuda result matches serial");
    if (n_failed == 0) std::cout << "  PASS\n";
}
#endif

// ---------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------

int main() {
    std::cout << "=== HAC correctness tests ===\n\n";

    test_serial_expected_output();
    test_parallel_matches_serial(1);
    test_parallel_matches_serial(2);
    test_parallel_matches_serial(4);
    test_parallel_three_clusters(2);
    test_parallel_three_clusters(4);
    test_parallel_unsupported_linkage_does_not_crash();

#ifdef ENABLE_CUDA
    test_cuda_matches_serial();
    test_cuda_three_clusters();
#endif

    std::cout << "\n";
    if (n_failed == 0)
        std::cout << "All tests passed.\n";
    else
        std::cerr << n_failed << " test(s) FAILED.\n";

    return n_failed == 0 ? 0 : 1;
}
