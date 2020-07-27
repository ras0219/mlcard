#pragma once

#include <memory>
#include <valarray>

#if !defined(NDEBUG) || defined(VEC_ENABLE_CHECKS)
#define VEC_CHECK_BOUNDS(X)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        if ((X).size() != size()) std::terminate();                                                                    \
    } while (0)
#else
#define VEC_CHECK_BOUNDS(X)
#endif

#define VEC_STACK_VEC(X, SZ)                                                                                           \
    auto X##_size = (SZ);                                                                                              \
    double* X##_storage = (double*)_alloca(sizeof(double) * X##_size);                                                 \
    vec_slice X(X##_storage, X##_size)

struct mat_slice;

struct vec_slice
{
    constexpr vec_slice() = default;
    constexpr vec_slice(double* data, size_t len) : m_data(data), m_len(len) { }
    template<size_t Sz>
    constexpr vec_slice(double (&data)[Sz]) : m_data(data), m_len(Sz)
    {
    }
    vec_slice(std::valarray<double>& v) : m_data(&v[0]), m_len(v.size()) { }

    constexpr double* data() { return m_data; }
    constexpr size_t size() const { return m_len; }

    double dot(vec_slice o) const
    {
        VEC_CHECK_BOUNDS(o);
        double sum = 0.0;
        for (size_t i = 0; i < m_len; ++i)
        {
            sum += m_data[i] * o.m_data[i];
        }
        return sum;
    }

    double& operator[](size_t i)
    {
#if !defined(NDEBUG) || defined(VEC_ENABLE_CHECKS)
        if (i >= m_len) std::terminate();
#endif
        return m_data[i];
    }

    double* begin() { return m_data; }
    double* end() { return m_data + m_len; }

    vec_slice slice() { return *this; }
    vec_slice slice(size_t offset) { return {m_data + offset, m_len - offset}; }
    vec_slice slice(size_t offset, size_t len) { return {m_data + offset, len}; }
    std::pair<vec_slice, vec_slice> split(size_t offset)
    {
        return {{m_data, offset}, {m_data + offset, m_len - offset}};
    }
    std::pair<vec_slice, vec_slice> rsplit(size_t offset)
    {
        return {{m_data, m_len - offset}, {m_data + m_len - offset, offset}};
    }

#define VEC_EOP(RHS)                                                                                                   \
    for (size_t i = 0; i < m_len; ++i)                                                                                 \
        m_data[i] RHS;                                                                                                 \
    return *this

    vec_slice& assign(vec_slice o)
    {
        VEC_CHECK_BOUNDS(o);
        VEC_EOP(= o.m_data[i]);
    }
    vec_slice& assign(double d) { VEC_EOP(= d); }

    vec_slice& assign_mult(vec_slice a, vec_slice b)
    {
        VEC_CHECK_BOUNDS(a);
        VEC_CHECK_BOUNDS(b);
        VEC_EOP(= a.m_data[i] * b.m_data[i]);
    }
    vec_slice& assign_mult(vec_slice a, double b)
    {
        VEC_CHECK_BOUNDS(a);
        VEC_EOP(= a.m_data[i] * b);
    }

    vec_slice& assign_add(vec_slice a, vec_slice b)
    {
        VEC_CHECK_BOUNDS(a);
        VEC_CHECK_BOUNDS(b);
        VEC_EOP(= a.m_data[i] + b.m_data[i]);
    }
    vec_slice& assign_add(vec_slice a, double b)
    {
        VEC_CHECK_BOUNDS(a);
        VEC_EOP(= a.m_data[i] + b);
    }

    vec_slice& assign_sub(vec_slice a, vec_slice b)
    {
        VEC_CHECK_BOUNDS(a);
        VEC_CHECK_BOUNDS(b);
        VEC_EOP(= a.m_data[i] - b.m_data[i]);
    }
    vec_slice& assign_sub(double a, vec_slice b)
    {
        VEC_CHECK_BOUNDS(b);
        VEC_EOP(= a - b.m_data[i]);
    }

    vec_slice& fma(vec_slice a, vec_slice b)
    {
        VEC_CHECK_BOUNDS(a);
        VEC_CHECK_BOUNDS(b);
        VEC_EOP(+= a.m_data[i] * b.m_data[i]);
    }
    vec_slice& fma(vec_slice a, double b)
    {
        VEC_CHECK_BOUNDS(a);
        VEC_EOP(+= a.m_data[i] * b);
    }

    vec_slice& add(vec_slice a)
    {
        VEC_CHECK_BOUNDS(a);
        VEC_EOP(+= a.m_data[i]);
    }

    vec_slice& mult(double a) { VEC_EOP(*= a); }

    double max(double init) const
    {
        for (size_t i = 0; i < m_len; ++i)
        {
            if (m_data[i] > init) init = m_data[i];
        }
        return init;
    }

    // ratio is multiplied by x
    vec_slice& decay_average(vec_slice x, double ratio)
    {
        VEC_CHECK_BOUNDS(x);
        VEC_EOP(= m_data[i] * (1 - ratio) + x.m_data[i] * ratio);
    }

    // ratio is multiplied by x
    vec_slice& decay_variance(vec_slice x, double ratio)
    {
        VEC_CHECK_BOUNDS(x);
        VEC_EOP(= m_data[i] * (1 - ratio) + x.m_data[i] * x.m_data[i] * ratio);
    }

#undef VEC_EOP

private:
    double* m_data = nullptr;
    size_t m_len = 0;
};

struct mat_slice
{
    constexpr mat_slice() = default;
    constexpr mat_slice(double* data, size_t rows, size_t cols) : m_data(data), m_rows(rows), m_cols(cols) { }
    constexpr mat_slice(vec_slice data, size_t stride)
        : m_data(data.data()), m_rows(data.size() / stride), m_cols(stride)
    {
#if !defined(NDEBUG) || defined(VEC_ENABLE_CHECKS)
        if (data.size() % stride != 0) std::terminate();
#endif
    }

    double* data() { return m_data; }
    size_t size() const { return m_rows * m_cols; }
    size_t rows() const { return m_rows; }
    size_t cols() const { return m_cols; }

    vec_slice row(size_t i)
    {
#if !defined(NDEBUG) || defined(VEC_ENABLE_CHECKS)
        if (i >= m_rows) std::terminate();
#endif
        return vec_slice(m_data + i * m_cols, m_cols);
    }

    vec_slice last_row()
    {
#if !defined(NDEBUG) || defined(VEC_ENABLE_CHECKS)
        if (m_rows == 0) std::terminate();
#endif
        return vec_slice(m_data + (m_rows - 1) * m_cols, m_cols);
    }

    double* begin() { return m_data; }
    double* end() { return m_data + size(); }

    vec_slice flat() { return {data(), size()}; }

private:
    double* m_data = nullptr;
    size_t m_rows = 0;
    size_t m_cols = 0;
};

struct vec
{
    constexpr vec() = default;
    vec(const vec& o) { *this = o; }

    vec& operator=(const vec& o)
    {
        realloc_uninitialized(o.m_len);
        for (size_t i = 0; i < m_len; ++i)
            m_data[i] = o.m_data[i];
        return *this;
    }
    vec& operator=(vec&&) = default;

    operator vec_slice() { return {m_data.get(), m_len}; }

    vec_slice slice() { return *this; }
    vec_slice slice(size_t offset) { return {m_data.get() + offset, m_len - offset}; }
    vec_slice slice(size_t offset, size_t len) { return {m_data.get() + offset, len}; }

    double* data() { return m_data.get(); }
    size_t size() const { return m_len; }

    double& operator[](size_t i)
    {
#if !defined(NDEBUG) || defined(VEC_ENABLE_CHECKS)
        if (i >= m_len) std::terminate();
#endif
        return m_data[i];
    }

    double* begin() { return m_data.get(); }
    double* end() { return m_data.get() + m_len; }

    const double* begin() const { return m_data.get(); }
    const double* end() const { return m_data.get() + m_len; }

    void realloc(size_t n, double v)
    {
        realloc_uninitialized(n);
        slice().assign(v);
    }

    void realloc_uninitialized(size_t n)
    {
        if (n == 0)
        {
            m_data.reset();
            m_len = 0;
        }
        else if (m_len != n)
        {
            m_data = std::make_unique<double[]>(n);
            m_len = n;
        }
    }

    void alloc_assign(vec_slice v)
    {
        realloc_uninitialized(v.size());
        slice().assign(v);
    }

private:
    std::unique_ptr<double[]> m_data;
    size_t m_len = 0;
};

#undef VEC_CHECK_BOUNDS
