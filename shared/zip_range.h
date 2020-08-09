#pragma once

#include "zip_iterator.h"
#include <iterator>
#include <utility>

template<class Range1, class Range2>
struct zip_range
{
    using iterator = zip_iterator<decltype(std::begin(std::declval<const Range1&>())),
                                  decltype(std::begin(std::declval<const Range2&>()))>;

    constexpr zip_range() = default;
    constexpr zip_range(Range1&& a, Range2&& b) : r1(static_cast<Range1&&>(a)), r2(static_cast<Range2&&>(b)) { }

    iterator begin() const { return iterator{std::begin(r1), std::begin(r2)}; }
    iterator end() const { return iterator{std::end(r1), std::end(r2)}; }

private:
    Range1 r1;
    Range2 r2;
};

template<class Range1, class Range2>
zip_range(Range1&, Range2 &&) -> zip_range<Range1&, Range2>;

template<class Range1, class Range2>
zip_range(Range1&&, Range2&) -> zip_range<Range1, Range2&>;

template<class Range1, class Range2>
zip_range(Range1&, Range2&) -> zip_range<Range1&, Range2&>;
