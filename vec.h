#pragma once

#include <memory>
#include <valarray>

#if !defined(NDEBUG) || defined(VEC_ENABLE_CHECKS)
#define VEC_CHECK_BOUNDS(X)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        if ((X).size() != this->size()) std::terminate();                                                              \
    } while (0)
#else
#define VEC_CHECK_BOUNDS(X)
#endif

#if !defined(NDEBUG) || defined(VEC_ENABLE_CHECKS)
#define VEC_CHECK_BOUNDS1(X)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        if ((X).size() + 1 != this->size()) std::terminate();                                                          \
    } while (0)
#else
#define VEC_CHECK_BOUNDS1(X)
#endif

#if !defined(_WIN32)
#define _alloca alloca
#endif
#define VEC_STACK_VEC(X, SZ)                                                                                           \
    auto X##_size = (SZ);                                                                                              \
    float* X##_storage = (float*)_alloca(sizeof(float) * X##_size);                                                    \
    vec_slice X(X##_storage, X##_size)

struct mat_slice;

#define VEC_EOP(RHS)                                                                                                   \
    for (size_t i = 0; i < this->size(); ++i)                                                                          \
        (*this)[i] RHS;                                                                                                \
    return self()

struct vec_slice_base
{
    constexpr vec_slice_base() = default;
    constexpr vec_slice_base(float* data, size_t len) : m_data(data), m_len(len) { }
    template<size_t Sz>
    constexpr vec_slice_base(float (&data)[Sz]) : m_data(data), m_len(Sz)
    {
    }
    vec_slice_base(std::valarray<float>& v) : m_data(&v[0]), m_len(v.size()) { }

    constexpr size_t size() const { return m_len; }

    float& operator[](size_t i) const
    {
#if !defined(NDEBUG) || defined(VEC_ENABLE_CHECKS)
        if (i >= m_len) std::terminate();
#endif
        return m_data[i];
    }

protected:
    float* m_data = nullptr;
    size_t m_len = 0;
};

struct vec_stride_slice_base
{
    constexpr vec_stride_slice_base() = default;
    constexpr vec_stride_slice_base(float* data, ptrdiff_t stride, size_t len)
        : m_data(data), m_stride(stride), m_len(len)
    {
    }

    constexpr size_t size() const { return m_len; }

    float& operator[](size_t i) const
    {
#if !defined(NDEBUG) || defined(VEC_ENABLE_CHECKS)
        if (i >= m_len) std::terminate();
#endif
        return m_data[i * m_stride];
    }

protected:
    float* m_data = nullptr;
    ptrdiff_t m_stride = 0;
    size_t m_len = 0;
};

struct vec_stride_slice_base;

template<class Derived, class Base>
struct vec_ops_mixin : Base
{
    using Base::Base;

    /// <summary>
    /// this[0]*o[0] + this[1]*o[1] + ...
    /// </summary>
    template<class V>
    float dot(V o) const
    {
        VEC_CHECK_BOUNDS(o);
        float sum = 0.0f;
        for (size_t i = 0; i < this->size(); ++i)
        {
            sum += (*this)[i] * o[i];
        }
        return sum;
    }

    /// <summary>
    /// this[0] * o[0] + this[1] * o[1] + ... this[n-1] * o[n-1] + this[n] * 1
    /// </summary>
    template<class V>
    float dot1(V o) const
    {
        VEC_CHECK_BOUNDS1(o);
        float sum = (*this)[o.size()];
        for (size_t i = 0; i < o.size(); ++i)
        {
            sum += (*this)[i] * o[i];
        }
        return sum;
    }

    /// <summary>
    /// this[i] = o[i]
    /// </summary>
    Derived& assign(vec_slice_base o)
    {
        VEC_CHECK_BOUNDS(o);
        VEC_EOP(= o[i]);
    }
    /// <summary>
    /// this[i] = d
    /// </summary>
    Derived& assign(float d) { VEC_EOP(= d); }

    /// <summary>
    /// this[i] = a[i] * b[i]
    /// </summary>
    Derived& assign_mult(vec_slice_base a, vec_slice_base b)
    {
        VEC_CHECK_BOUNDS(a);
        VEC_CHECK_BOUNDS(b);
        VEC_EOP(= a[i] * b[i]);
    }

    /// <summary>
    /// this[i] = a[i] * b
    /// </summary>
    Derived& assign_mult(vec_slice_base a, float b)
    {
        VEC_CHECK_BOUNDS(a);
        VEC_EOP(= a[i] * b);
    }

    /// <summary>
    /// this[i] = a[i] + b[i]
    /// </summary>
    Derived& assign_add(vec_slice_base a, vec_slice_base b)
    {
        VEC_CHECK_BOUNDS(a);
        VEC_CHECK_BOUNDS(b);
        VEC_EOP(= a[i] + b[i]);
    }

    /// <summary>
    /// this[i] = a[i] + b
    /// </summary>
    Derived& assign_add(vec_slice_base a, float b)
    {
        VEC_CHECK_BOUNDS(a);
        VEC_EOP(= a[i] + b);
    }

    /// <summary>
    /// this[i] = a[i] - b[i]
    /// </summary>
    Derived& assign_sub(vec_slice_base a, vec_slice_base b)
    {
        VEC_CHECK_BOUNDS(a);
        VEC_CHECK_BOUNDS(b);
        VEC_EOP(= a[i] - b[i]);
    }
    /// <summary>
    /// this[i] = a - b[i]
    /// </summary>
    Derived& assign_sub(float a, vec_slice_base b)
    {
        VEC_CHECK_BOUNDS(b);
        VEC_EOP(= a - b[i]);
    }

    /// <summary>
    /// this[i] = M[i] dot (v... 1)
    /// </summary>
    template<class Mat, class Vec>
    Derived& assign_mv1_mult(Mat m, Vec v)
    {
        VEC_EOP(= m.row(i).dot1(v));
    }

    /// <summary>
    /// this[i] = M[i] dot v
    /// </summary>
    template<class Mat, class Vec>
    Derived& assign_mv_mult(Mat m, Vec v)
    {
        VEC_EOP(= m.row(i).dot(v));
    }

    /// <summary>
    /// this[i] += a[i] * b[i]
    /// </summary>
    Derived& fma(vec_slice_base a, vec_slice_base b)
    {
        VEC_CHECK_BOUNDS(a);
        VEC_CHECK_BOUNDS(b);
        VEC_EOP(+= a[i] * b[i]);
    }
    /// <summary>
    /// this[i] += a[i] * b
    /// </summary>
    Derived& fma(vec_slice_base a, float b)
    {
        VEC_CHECK_BOUNDS(a);
        VEC_EOP(+= a[i] * b);
    }

    /// <summary>
    /// this[i] += a[i]
    /// </summary>
    Derived& add(vec_slice_base a)
    {
        VEC_CHECK_BOUNDS(a);
        VEC_EOP(+= a[i]);
    }

    /// <summary>
    /// this[i] *= a
    /// </summary>
    Derived& mult(float a) { VEC_EOP(*= a); }

    /// <summary>
    /// max(this[0], this[1], ...)
    /// </summary>
    float max(float init) const
    {
        for (size_t i = 0; i < this->size(); ++i)
        {
            if ((*this)[i] > init) init = (*this)[i];
        }
        return init;
    }

    /// <summary>
    /// min(this[0], this[1], ...)
    /// </summary>
    float min(float init) const
    {
        for (size_t i = 0; i < this->size(); ++i)
        {
            if ((*this)[i] < init) init = (*this)[i];
        }
        return init;
    }

    /// <summary>
    /// this[i] = this[i] * (1 - ratio) + x[i] * ratio
    /// </summary>
    Derived& decay_average(vec_slice_base x, float ratio)
    {
        VEC_CHECK_BOUNDS(x);
        VEC_EOP(= (*this)[i] * (1 - ratio) + x[i] * ratio);
    }

    /// <summary>
    /// this[i] = this[i] * (1 - ratio) + x[i]^2 * ratio
    /// </summary>
    Derived& decay_variance(vec_slice_base x, float ratio)
    {
        VEC_CHECK_BOUNDS(x);
        VEC_EOP(= (*this)[i] * (1 - ratio) + x[i] * x[i] * ratio);
    }

private:
    Derived& self() { return static_cast<Derived&>(*this); }
    const Derived& self() const { return static_cast<const Derived&>(*this); }
};

struct vec_slice : vec_ops_mixin<vec_slice, vec_slice_base>
{
    using vec_ops_mixin::vec_ops_mixin;

    constexpr float* data() const { return m_data; }
    float* begin() { return m_data; }
    float* end() { return m_data + m_len; }

    vec_slice slice() { return *this; }
    vec_slice slice(size_t offset) { return {m_data + offset, m_len - offset}; }
    vec_slice slice(size_t offset, size_t len) { return {m_data + offset, len}; }
    vec_slice rslice(size_t offset, size_t len) { return {m_data + m_len - offset, len}; }
    std::pair<vec_slice, vec_slice> split(size_t offset)
    {
        return {{m_data, offset}, {m_data + offset, m_len - offset}};
    }
    std::pair<vec_slice, vec_slice> rsplit(size_t offset)
    {
        return {{m_data, m_len - offset}, {m_data + m_len - offset, offset}};
    }
};

struct vec_stride_slice : vec_ops_mixin<vec_slice, vec_stride_slice_base>
{
    using vec_ops_mixin::vec_ops_mixin;

    constexpr float* data() const { return m_data; }

    vec_stride_slice slice() { return *this; }
    vec_stride_slice slice(size_t offset) { return {m_data + m_stride * offset, m_stride, m_len - offset}; }
    vec_stride_slice slice(size_t offset, size_t len) { return {m_data + m_stride * offset, m_stride, len}; }
    vec_stride_slice rslice(size_t offset, size_t len) { return {m_data + m_stride * (m_len - offset), m_stride, len}; }
    std::pair<vec_stride_slice, vec_stride_slice> split(size_t offset)
    {
        return {{m_data, m_stride, offset}, {m_data + m_stride * offset, m_stride, m_len - offset}};
    }
    std::pair<vec_stride_slice, vec_stride_slice> rsplit(size_t offset)
    {
        return {{m_data, m_stride, m_len - offset}, {m_data + m_stride * (m_len - offset), m_stride, offset}};
    }
};

#undef VEC_EOP

struct mat_slice_base
{
    constexpr mat_slice_base() = default;
    constexpr mat_slice_base(float* data, size_t rows, size_t cols) : m_data(data), m_rows(rows), m_cols(cols) { }
    constexpr mat_slice_base(vec_slice data, size_t cols)
        : m_data(data.data()), m_rows(data.size() / cols), m_cols(cols)
    {
#if !defined(NDEBUG) || defined(VEC_ENABLE_CHECKS)
        if (data.size() % cols != 0) std::terminate();
#endif
    }

    float* data() { return m_data; }
    size_t size() const { return m_rows * m_cols; }
    size_t rows() const { return m_rows; }
    size_t cols() const { return m_cols; }

    float* begin() { return m_data; }
    float* end() { return m_data + size(); }

    vec_slice flat() { return {data(), size()}; }

protected:
    float* m_data = nullptr;
    size_t m_rows = 0;
    size_t m_cols = 0;
};

struct row_major_mat_slice_base : mat_slice_base
{
    using mat_slice_base::mat_slice_base;

    vec_slice row(size_t i) const
    {
#if !defined(NDEBUG) || defined(VEC_ENABLE_CHECKS)
        if (i >= m_rows) std::terminate();
#endif
        return vec_slice(m_data + i * m_cols, m_cols);
    }

    vec_stride_slice col(size_t i) const
    {
#if !defined(NDEBUG) || defined(VEC_ENABLE_CHECKS)
        if (i >= m_cols) std::terminate();
#endif
        return vec_stride_slice(m_data + i, m_cols, m_rows);
    }

    vec_slice last_row() const { return row(m_rows - 1); }
    vec_stride_slice last_col() const { return col(m_cols - 1); }
};

struct col_major_mat_slice_base : mat_slice_base
{
    using mat_slice_base::mat_slice_base;

    vec_slice col(size_t i) const
    {
#if !defined(NDEBUG) || defined(VEC_ENABLE_CHECKS)
        if (i >= m_cols) std::terminate();
#endif
        return vec_slice(m_data + i * m_rows, m_rows);
    }

    vec_stride_slice row(size_t i) const
    {
#if !defined(NDEBUG) || defined(VEC_ENABLE_CHECKS)
        if (i >= m_rows) std::terminate();
#endif
        return vec_stride_slice(m_data + i, m_rows, m_cols);
    }

    vec_stride_slice last_row() const { return row(m_rows - 1); }
    vec_slice last_col() const { return col(m_cols - 1); }
};

struct transposed_mat_slice;

struct mat_slice : row_major_mat_slice_base
{
    using row_major_mat_slice_base::row_major_mat_slice_base;

    mat_slice slice_rows(size_t offset, size_t len) const
    {
#if !defined(NDEBUG) || defined(VEC_ENABLE_CHECKS)
        if (len + offset > m_rows) std::terminate();
#endif
        return {m_data + offset * m_cols, len, m_cols};
    }
    inline transposed_mat_slice transpose() const;
};

struct transposed_mat_slice : col_major_mat_slice_base
{
    using col_major_mat_slice_base::col_major_mat_slice_base;

    mat_slice slice_cols(size_t offset, size_t len) const
    {
#if !defined(NDEBUG) || defined(VEC_ENABLE_CHECKS)
        if (len + offset > m_cols) std::terminate();
#endif
        return {m_data + offset * m_rows, m_rows, len};
    }
    mat_slice transpose() const { return {m_data, m_cols, m_rows}; }
};

transposed_mat_slice mat_slice::transpose() const { return {m_data, m_cols, m_rows}; }

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

    float* data() { return m_data.get(); }
    size_t size() const { return m_len; }

    float& operator[](size_t i)
    {
#if !defined(NDEBUG) || defined(VEC_ENABLE_CHECKS)
        if (i >= m_len) std::terminate();
#endif
        return m_data[i];
    }

    float* begin() { return m_data.get(); }
    float* end() { return m_data.get() + m_len; }

    const float* begin() const { return m_data.get(); }
    const float* end() const { return m_data.get() + m_len; }

    void realloc(size_t n, float v)
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
            m_data = std::make_unique<float[]>(n);
            m_len = n;
        }
    }

    void alloc_assign(vec_slice v)
    {
        realloc_uninitialized(v.size());
        slice().assign(v);
    }

private:
    std::unique_ptr<float[]> m_data;
    size_t m_len = 0;
};

#undef VEC_CHECK_BOUNDS
