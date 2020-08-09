#pragma once

#include "count_iterator.h"

template<class T>
struct count_range
{
    using iterator = count_iterator<T>;

    constexpr count_range() : b{}, e{} { }
    constexpr count_range(T b, T e) : b(b), e(e) { }
    constexpr count_range(T e) : b(0), e(e) { }

    iterator begin() const { return iterator{b}; }
    iterator end() const { return iterator{e}; }

private:
    T b, e;
};
