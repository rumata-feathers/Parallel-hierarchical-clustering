#include "data_loader.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

static std::vector<std::string> split_line(const std::string& line) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream ss(line);
    while (std::getline(ss, token, ',')) {
        tokens.push_back(token);
    }
    return tokens;
}

static bool is_numeric(const std::string& s) {

    if (s.empty())
        return false;

    for (char c : s) {
        if (!std::isdigit(c) && c != '.' && c != '-' && c != '+')
            return false;
    }
    return true;
}

Dataset load_csv(const std::string& path, bool has_header) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + path);

    Dataset ds;
    std::string line;
    bool skipped_header = false;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        if (!skipped_header) {
            skipped_header = true;
            auto tokens = split_line(line);
            if (has_header || !is_numeric(tokens[0]))
                continue;
        }

        auto tokens = split_line(line);
        if (tokens.size() < 2) continue; // malformed line, skip

        // feature, feature, ..., label (last column)
        std::vector<double> row;
        row.reserve(tokens.size() - 1);
        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            if (!is_numeric(tokens[i]))
                throw std::runtime_error("Cannot read that feature value(non-numeric) '" + tokens[i] +
                                         "' in: " + path);
            row.push_back(std::stod(tokens[i])); // convert to double
        }

        ds.features.emplace_back(row);
        ds.labels.push_back(tokens[tokens.size() - 1]);
    }

    if (ds.features.empty())
        throw std::runtime_error("No data: " + path);

    return ds;
}
