#pragma once

#include <iterator>
#include <utility>

template<class Iter>
using reference_type = decltype((*std::declval<Iter>()));

template<class I1, class I2>
struct zip_iterator
{
    zip_iterator() = default;
    zip_iterator(I1 i1, I2 i2) : i1(std::move(i1)), i2(std::move(i2)) { }
    using iterator_category = typename I1::iterator_category;
    using difference_type = typename I1::difference_type;
    using pointer = void;
    using reference = std::pair<reference_type<I1>, reference_type<I2>>;

    zip_iterator& operator++() { return ++i1, ++i2, *this; }
    zip_iterator operator++(int)
    {
        zip_iterator tmp = *this;
        return ++i1, ++i2, tmp;
    }
    zip_iterator& operator--() { return --i1, --i2, *this; }
    zip_iterator operator--(int)
    {
        zip_iterator tmp = *this;
        return --i1, --i2, tmp;
    }
    zip_iterator& operator+=(int n) { return i1 += n, i2 += n, *this; }
    zip_iterator operator+(int n) const { return {i1 + n, i2 + n}; }
    zip_iterator& operator-=(int n) { return i1 -= n, i2 -= n, *this; }
    zip_iterator operator-(int n) const { return {i1 - n, i2 - n}; }
    difference_type operator-(zip_iterator n) const
    {
        auto d1 = i1 - n.i1;
        auto d2 = i2 - n.i2;
        return d1 < d2 ? d1 : d2;
    }

    reference operator*() const { return {*i1, *i2}; }
    reference operator[](difference_type i) const { return {i1[i], i2[i]}; }

    bool operator==(zip_iterator o) const { return i1 == o.i1 || i2 == o.i2; }
    bool operator!=(zip_iterator o) const { return !(*this == o); }

private:
    I1 i1;
    I2 i2;
};

template<class I1, class I2>
zip_iterator<I1, I2> operator+(int n, zip_iterator<I1, I2> o)
{
    return o + n;
};
