#pragma once

#include <map>
#include <string>
#include <vector>

struct ModelDims
{
    ModelDims(std::vector<int> dims, std::map<std::string, ModelDims> children = {})
        : children(std::move(children)), dims(std::move(dims))
    {
    }

    std::map<std::string, ModelDims> children;
    std::vector<int> dims;
};
