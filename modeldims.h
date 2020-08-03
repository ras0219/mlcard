#pragma once

#include <map>
#include <string>
#include <vector>

struct ModelDims
{
    ModelDims() = default;
    explicit ModelDims(std::vector<int> dims) : dims(std::move(dims)) { }
    explicit ModelDims(std::string t, std::vector<int> dims) : type(std::move(t)), dims(std::move(dims)) { }
    ModelDims(std::initializer_list<int> dims) : dims(dims) { }
    explicit ModelDims(std::map<std::string, ModelDims> childs) : children(std::move(childs)) { }

    std::string type;
    std::map<std::string, ModelDims> children;
    std::vector<int> dims;
};
