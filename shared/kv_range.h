#pragma once

#include "count_range.h"
#include "zip_range.h"

template<class C>
struct kv_range : zip_range<count_range<size_t>, C>
{
    explicit kv_range(C&& c) : zip_range<count_range<size_t>, C>({0, c.size()}, static_cast<C&&>(c)) { }
};

template<class C>
kv_range(C&) -> kv_range<C&>;
