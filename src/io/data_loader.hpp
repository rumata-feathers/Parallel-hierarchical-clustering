#pragma once
#include <string>
#include <vector>

struct Dataset {
    std::vector<std::vector<double>> features;  // n x d, label column stripped
    std::vector<std::string> labels;            // n labels
};

Dataset load_csv(const std::string& path, bool has_header = true);
