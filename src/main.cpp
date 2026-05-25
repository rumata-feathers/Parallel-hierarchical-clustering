#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

// huge includes oh well
#include "hac/distance.hpp"
#include "hac/hac_parallel.hpp"
#include "hac/hac_serial.hpp"
#include "hac/linkage.hpp"
#include "io/data_loader.hpp"
#include "io/output_writer.hpp"
#ifdef ENABLE_CUDA
#include "hac/hac_cuda.hpp"
#endif

namespace fs = std::filesystem;

void usage(const char* prog){
    std::cerr
    << "Usage: " << prog << "\n"
    << "  --dataset  <path.csv>                      input file\n"
    << "  --linkage  single|complete|average|ward|centroid|median\n"
    << "  --mode     serial|parallel|cuda\n"
    << "  --out-dir  <dir>                            output directory\n"
    << "  --threads  <n>                              (parallel mode, default "
       "4)\n";}

int main(int argc, char** argv) {
    std::string dataset_path, linkage_str, mode, out_dir;
    int n_threads = 4;
    bool save_labels = false;
    bool has_header = true;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--dataset" && i + 1 < argc)
            dataset_path = argv[++i];
        else if (a == "--linkage" && i + 1 < argc)
            linkage_str = argv[++i];
        else if (a == "--mode" && i + 1 < argc)
            mode = argv[++i];
        else if (a == "--out-dir" && i + 1 < argc)
            out_dir = argv[++i];
        else if (a == "--threads" && i + 1 < argc)
            n_threads = std::stoi(argv[++i]);
        else if (a == "--save-labels")
            save_labels = true;
        else if (a == "--no-header")
            has_header = false;
        else {
            std::cerr << "Unknown argument: " << a << "\n";
            usage(argv[0]);
            return 1;
        }
    }

    if (dataset_path.empty() || linkage_str.empty() || mode.empty() ||
        out_dir.empty()) {
        usage(argv[0]);
        return 1;
    }

    // load
    Dataset df;
    try {
        df = load_csv(dataset_path, has_header);
    } catch (const std::exception& e) {
        std::cerr << "Load error: " << e.what() << "\n";
        return 1;
    }
    std::cout << "Loaded " << df.features.size() << " points from "
              << dataset_path << "\n";

    auto distances = compute_distance_matrix(df.features);

    Linkage linkage = parse_linkage(linkage_str);

    // set-up time
    auto start_time = std::chrono::high_resolution_clock::now();
    // let's go baby
    std::vector<std::array<double, 4>> result;
    if (mode == "serial") {
        result = hac_serial(distances, df.features, linkage);
    } else if (mode == "parallel") {
        result = hac_parallel(distances, df.features, linkage, n_threads);
    } else if (mode == "cuda") {
#ifdef ENABLE_CUDA
        result = hac_cuda(distances, df.features, linkage);
#else
        std::cerr << "Binary was built without CUDA support. "
                     "Rebuild with -DENABLE_CUDA=ON.\n";
        return 1;
#endif
    } else {
        std::cerr << "Unknown mode: " << mode << "\n";
        return 1;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double wall = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    fs::create_directories(out_dir);
    std::string stem = fs::path(dataset_path).stem().string();
    std::string linkage_matrix_path =
        out_dir + "/" + stem + "_" + linkage_str + "_" + mode + "_linkage.csv";

    write_linkage_matrix(result, linkage_matrix_path);

    if (save_labels)
        write_labels(df.labels, out_dir + "/" + stem + "_labels.csv");

    append_timing(out_dir + "/timings.csv", stem, linkage_str, mode, n_threads,
                  static_cast<int>(df.features.size()), wall);

    std::cout << "Done [" << mode << "/" << linkage_str << "] " << wall
              << " ms  →  " << linkage_matrix_path << "\n";
    return 0;
}
