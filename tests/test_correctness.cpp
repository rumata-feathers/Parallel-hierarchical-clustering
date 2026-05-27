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
static int g_step   = 0;
static int g_total  = 0;  // set in main() before first test runs

static void print_progress() {
    const int width = 40;
    int filled = (g_total > 0) ? width * g_step / g_total : 0;
    int pct    = (g_total > 0) ? 100 * g_step / g_total   : 0;
    std::cout << "  [";
    for (int i = 0; i < width; ++i)
        std::cout << (i < filled ? "█" : "░");
    std::cout << "] " << g_step << '/' << g_total
              << "  (" << pct << "%)\n";
}

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

static void test_parallel_ward_matches_serial() {
    std::cout << "[parallel] ward linkage matches serial (two-cluster dataset)\n";
    auto data = make_two_cluster_data();
    auto dist = compute_distance_matrix(data);
    auto serial   = hac_serial(dist, data, Linkage::WARD);
    auto parallel = hac_parallel(dist, data, Linkage::WARD, 2);

    CHECK(!serial.empty(),   "serial produced result");
    CHECK(!parallel.empty(), "parallel produced result");
    CHECK(matrices_equal(serial, parallel),
          "parallel ward result matches serial");
    if (n_failed == 0) std::cout << "  PASS\n";
}

#ifdef ENABLE_CUDA

// ---------------------------------------------------------------
// Reference HAC (CPU) — implements all linkages so CUDA can be
// validated even for linkages not yet in hac_serial.
// Uses the same scanning order as hac_serial / hac_cuda so cluster
// IDs match exactly when there are no distance ties.
// ---------------------------------------------------------------
static double compute_cluster_dist_ref(
    const std::vector<int>& ci, const std::vector<int>& cj,
    const std::vector<std::vector<double>>& dist,
    const std::vector<std::vector<double>>& data,
    Linkage linkage)
{
    if (ci.empty() || cj.empty()) return std::numeric_limits<double>::infinity();

    switch (linkage) {
        case Linkage::SINGLE: {
            double d = std::numeric_limits<double>::infinity();
            for (auto a : ci) for (auto b : cj) d = std::min(d, dist[a][b]);
            return d;
        }
        case Linkage::COMPLETE: {
            double d = 0.0;
            for (auto a : ci) for (auto b : cj) d = std::max(d, dist[a][b]);
            return d;
        }
        case Linkage::AVERAGE: {
            double sum = 0.0;
            for (auto a : ci) for (auto b : cj) sum += dist[a][b];
            return sum / (double(ci.size()) * cj.size());
        }
        case Linkage::WARD: {
            size_t dims = data[0].size();
            std::vector<double> centi(dims, 0), centj(dims, 0);
            for (auto a : ci) for (size_t d = 0; d < dims; ++d) centi[d] += data[a][d];
            for (auto b : cj) for (size_t d = 0; d < dims; ++d) centj[d] += data[b][d];
            double sq = 0.0;
            for (size_t d = 0; d < dims; ++d) {
                double diff = centi[d] / ci.size() - centj[d] / cj.size();
                sq += diff * diff;
            }
            return std::sqrt(double(ci.size() * cj.size()) / (ci.size() + cj.size()) * sq);
        }
        case Linkage::CENTROID: {
            size_t dims = data[0].size();
            std::vector<double> centi(dims, 0), centj(dims, 0);
            for (auto a : ci) for (size_t d = 0; d < dims; ++d) centi[d] += data[a][d];
            for (auto b : cj) for (size_t d = 0; d < dims; ++d) centj[d] += data[b][d];
            double sq = 0.0;
            for (size_t d = 0; d < dims; ++d) {
                double diff = centi[d] / ci.size() - centj[d] / cj.size();
                sq += diff * diff;
            }
            return std::sqrt(sq);
        }
        case Linkage::MEDIAN: {
            // Computed from scratch: centroid-to-centroid (same formula as CENTROID)
            size_t dims = data[0].size();
            std::vector<double> centi(dims, 0), centj(dims, 0);
            for (auto a : ci) for (size_t d = 0; d < dims; ++d) centi[d] += data[a][d];
            for (auto b : cj) for (size_t d = 0; d < dims; ++d) centj[d] += data[b][d];
            double sq = 0.0;
            for (size_t d = 0; d < dims; ++d) {
                double diff = centi[d] / ci.size() - centj[d] / cj.size();
                sq += diff * diff;
            }
            return std::sqrt(sq);
        }
    }
    return std::numeric_limits<double>::infinity();
}

static std::vector<std::array<double, 4>> hac_ref(
    const std::vector<std::vector<double>>& dist,
    const std::vector<std::vector<double>>& data,
    Linkage linkage)
{
    int n = static_cast<int>(dist.size());
    int dims = data.empty() ? 0 : static_cast<int>(data[0].size());
    std::vector<std::vector<int>> clusters(n);
    for (int i = 0; i < n; ++i) clusters[i] = {i};
    std::vector<int> id(n);
    for (int i = 0; i < n; ++i) id[i] = i;
    int next_id = n;
// for median we track cluster centers (unweighted average)
// rather than recomputing from raw points
    std::vector<std::vector<double>> centers(n, std::vector<double>(dims, 0.0));
    if (linkage == Linkage::MEDIAN) {
        for (int i = 0; i < n; ++i) centers[i] = data[i];
    }
    std::vector<std::array<double, 4>> result;
    result.reserve(n - 1);

    for (int step = 0; step < n - 1; ++step) {
        double best_d = std::numeric_limits<double>::infinity();
        int ci = -1, cj = -1;
        for (int i = 0; i < n; ++i) {
            if (clusters[i].empty()) continue;
            for (int j = i + 1; j < n; ++j) {
                if (clusters[j].empty()) continue;
                double d;
                if (linkage == Linkage::MEDIAN) {
                    // distance between stored median centers
                    double sq = 0.0;
                    for (int k = 0; k < dims; ++k) {
                        double diff = centers[i][k] - centers[j][k];
                        sq += diff * diff;
                    }
                    d = std::sqrt(sq);
                } else {
                    d = compute_cluster_dist_ref(
                        clusters[i], clusters[j], dist, data, linkage);
                }
                if (d < best_d) { best_d = d; ci = i; cj = j; }
            }
        }
        if (ci == -1) break;

        result.push_back({double(id[ci]), double(id[cj]), best_d,
                          double(clusters[ci].size() + clusters[cj].size())});
                          
        // update median center: unweighted average                         
        if (linkage == Linkage::MEDIAN) {
            for (int k = 0; k < dims; ++k)
                centers[ci][k] = (centers[ci][k] + centers[cj][k]) / 2.0;
        }
        
        for (auto x : clusters[cj]) clusters[ci].push_back(x);
        clusters[cj].clear();
        id[ci] = next_id++;
    }
    return result;
}

// ---------------------------------------------------------------
// CUDA tests — each linkage compared against hac_ref.
// The two-cluster dataset is used because it has no distance ties
// for any linkage, so cluster IDs match the reference exactly.
// ---------------------------------------------------------------

static void test_cuda_linkage(Linkage linkage, const std::string& name) {
    int before = n_failed;
    std::cout << "[cuda] " << name << " linkage — two-cluster dataset\n";
    auto data = make_two_cluster_data();
    auto dist = compute_distance_matrix(data);
    auto ref  = hac_ref(dist, data, linkage);
    auto cuda = hac_cuda(dist, data, linkage);

    CHECK(ref.size()  == size_t(5), name + " ref: expected 5 merges");
    CHECK(cuda.size() == size_t(5), name + " cuda: expected 5 merges");
    CHECK(matrices_equal(ref, cuda), name + " cuda matches reference");
    if (n_failed == before) std::cout << "  PASS\n";
}

static void test_cuda_single_vs_serial() {
    // Extra check: for SINGLE, CUDA must also match hac_serial exactly.
    int before = n_failed;
    std::cout << "[cuda] SINGLE matches hac_serial (three-cluster 2D dataset)\n";
    auto data   = make_three_cluster_data();
    auto dist   = compute_distance_matrix(data);
    auto serial = hac_serial(dist, data, Linkage::SINGLE);
    auto cuda   = hac_cuda(dist, data, Linkage::SINGLE);

    CHECK(!serial.empty(), "serial produced result");
    CHECK(!cuda.empty(),   "cuda produced result");
    CHECK(matrices_equal(serial, cuda), "cuda matches hac_serial");
    if (n_failed == before) std::cout << "  PASS\n";
}

static void test_cuda_centroid_vs_serial() {
    // Extra check: for CENTROID, CUDA must also match hac_serial exactly.
    int before = n_failed;
    std::cout << "[cuda] CENTROID matches hac_serial (two-cluster dataset)\n";
    auto data   = make_two_cluster_data();
    auto dist   = compute_distance_matrix(data);
    auto serial = hac_serial(dist, data, Linkage::CENTROID);
    auto cuda   = hac_cuda(dist, data, Linkage::CENTROID);

    CHECK(!serial.empty(), "serial produced result");
    CHECK(!cuda.empty(),   "cuda produced result");
    CHECK(matrices_equal(serial, cuda), "cuda matches hac_serial");
    if (n_failed == before) std::cout << "  PASS\n";
}

static void test_cuda_result_is_valid_linkage_matrix() {
    // Structural sanity: for each linkage the result must have exactly n-1
    // rows, monotonically non-decreasing distances, and correct merged sizes.
    int before = n_failed;
    std::cout << "[cuda] structural validity for all linkages\n";
    auto data = make_two_cluster_data();
    auto dist = compute_distance_matrix(data);
    int  n    = static_cast<int>(data.size());

    for (auto [lk, name] : std::vector<std::pair<Linkage, std::string>>{
             {Linkage::SINGLE,   "single"},
             {Linkage::COMPLETE, "complete"},
             {Linkage::AVERAGE,  "average"},
             {Linkage::WARD,     "ward"},
             {Linkage::CENTROID, "centroid"},
             {Linkage::MEDIAN,   "median"},
         }) {
        auto Z = hac_cuda(dist, data, lk);
        CHECK(Z.size() == size_t(n - 1),
              name + ": expected " + std::to_string(n - 1) + " merges, got " +
              std::to_string(Z.size()));
        for (size_t i = 0; i < Z.size(); ++i) {
            CHECK(Z[i][2] >= 0.0, name + ": row " + std::to_string(i) + " distance < 0");
            if (i > 0)
                CHECK(Z[i][2] >= Z[i - 1][2] - 1e-9,
                      name + ": distances not non-decreasing at row " + std::to_string(i));
            CHECK(Z[i][3] >= 2.0, name + ": merged size < 2 at row " + std::to_string(i));
        }
    }
    if (n_failed == before) std::cout << "  PASS\n";
}

#endif

// ---------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------

int main() {
    g_total = 7;
#ifdef ENABLE_CUDA
    g_total += 9;
#endif
    std::cout << "=== HAC correctness tests ===  (" << g_total << " tests)\n\n";

// Print [n/total] header, run the test, then print the cumulative progress bar.
#define RUN(call)                                          \
    do {                                                   \
        ++g_step;                                          \
        std::cout << '[' << g_step << '/' << g_total << "] "; \
        call;                                              \
        print_progress();                                  \
    } while (0)

    RUN(test_serial_expected_output());
    RUN(test_parallel_matches_serial(1));
    RUN(test_parallel_matches_serial(2));
    RUN(test_parallel_matches_serial(4));
    RUN(test_parallel_three_clusters(2));
    RUN(test_parallel_three_clusters(4));
    RUN(test_parallel_ward_matches_serial());

#ifdef ENABLE_CUDA
    RUN(test_cuda_linkage(Linkage::SINGLE,   "single"));
    RUN(test_cuda_linkage(Linkage::COMPLETE, "complete"));
    RUN(test_cuda_linkage(Linkage::AVERAGE,  "average"));
    RUN(test_cuda_linkage(Linkage::WARD,     "ward"));
    RUN(test_cuda_linkage(Linkage::CENTROID, "centroid"));
    RUN(test_cuda_linkage(Linkage::MEDIAN,   "median"));
    RUN(test_cuda_single_vs_serial());
    RUN(test_cuda_centroid_vs_serial());
    RUN(test_cuda_result_is_valid_linkage_matrix());
#endif

#undef RUN

    std::cout << '\n';
    if (n_failed == 0)
        std::cout << "All tests passed.\n";
    else
        std::cerr << n_failed << " test(s) FAILED.\n";

    return n_failed == 0 ? 0 : 1;
}
