#include "output_writer.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

void write_linkage_matrix(const std::vector<std::tuple<int, int, double, int>>& matrix,
                          const std::string& path) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write: " + path);

    f << "cluster_i,cluster_j,distance,count\n";
    for (const auto& row : matrix) {
        f << static_cast<int>(std::get<0>(row)) << "," << static_cast<int>(std::get<1>(row)) << ","
          << std::fixed << std::setprecision(6) << std::get<2>(row) << ","
          << static_cast<int>(std::get<3>(row)) << "\n";
    }
}

void write_labels(const std::vector<std::string>& labels,
                  const std::string& path) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write: " + path);

    f << "label\n";
    for (const auto& l : labels) f << l << "\n";
}

void append_timing(const std::string& path, const std::string& dataset,
                   const std::string& linkage, const std::string& mode,
                   int n_threads, int n_points, double wall_ms) {
    bool write_header = false;
    {
        std::ifstream check(path);
        write_header = !check.good();
    }

    std::ofstream f(path, std::ios::app);
    if (!f) throw std::runtime_error("Cannot write: " + path);

    if (write_header) f << "dataset,linkage,mode,threads,n_points,wall_ms\n";

    f << dataset << "," << linkage << "," << mode << "," << n_threads << ","
      << n_points << ","
      << std::fixed << std::setprecision(2) << wall_ms << "\n";
}
