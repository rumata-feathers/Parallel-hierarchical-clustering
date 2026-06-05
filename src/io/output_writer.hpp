#pragma once
#include <tuple>
#include <string>
#include <vector>

void write_linkage_matrix(const std::vector<std::tuple<int, int, double, int>>& matrix,
                          const std::string& path);

void write_labels(const std::vector<std::string>& labels,
                  const std::string& path);

// appends one run to the shared timings CSV
void append_timing(const std::string& path, const std::string& dataset,
                   const std::string& linkage, const std::string& mode,
                   int n_threads, int n_points, double wall_ms);
