#pragma once
#include <cmath>
#include <stdexcept>
#include <string>

enum class Linkage { SINGLE, COMPLETE, AVERAGE, WARD, CENTROID, MEDIAN };

inline Linkage parse_linkage(const std::string& s) {
    if (s == "single")   return Linkage::SINGLE;
    if (s == "complete") return Linkage::COMPLETE;
    if (s == "average")  return Linkage::AVERAGE;
    if (s == "ward")     return Linkage::WARD;
    if (s == "centroid") return Linkage::CENTROID;
    if (s == "median")   return Linkage::MEDIAN;
    throw std::runtime_error("Unknown linkage: " + s);
}