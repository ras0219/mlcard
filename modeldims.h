#pragma once

#include <map>
#include <string>
#include <vector>

struct ModelDims
{
    std::map<std::string, ModelDims> children;
    std::vector<int> dims;
};
