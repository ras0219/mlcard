#pragma once

#include <iterator>

template<class T>
struct count_iterator
{
    count_iterator() = default;
    explicit count_iterator(T i) : x(i) { }

    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = T;
    using pointer = void;
    using reference = T;

    count_iterator& operator++() { return ++x, *this; }
    count_iterator operator++(int) { return ++x, count_iterator{x - 1}; }
    count_iterator& operator--() { return --x, *this; }
    count_iterator operator--(int) { return --x, count_iterator{x + 1}; }
    count_iterator& operator+=(T n) { return x += n, *this; }
    count_iterator operator+(T n) const { return count_iterator{x + n}; }
    count_iterator& operator-=(T n) { return x -= n, *this; }
    count_iterator operator-(T n) const { return count_iterator{x - n}; }
    T operator-(count_iterator n) const { return x - n.x; }

    T operator*() const { return x; }
    T operator[](T i) const { return x + i; }

    bool operator==(count_iterator o) const { return x == o.x; }
    bool operator!=(count_iterator o) const { return x != o.x; }

private:
    T x;
};

template<class T>
count_iterator<T> operator+(T n, count_iterator<T> o)
{
    return o + n;
};

using int_iterator = count_iterator<int>;
