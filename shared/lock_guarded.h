#pragma once

#include <mutex>

template<class T>
struct PtrLockable
{
    constexpr PtrLockable(std::mutex& m, T* t) : m(&m), t(t) { }

private:
    std::mutex* m;
    T* t;
};

template<class T>
struct LockedPtr
{
    constexpr LockedPtr() : t(nullptr) { }
    LockedPtr(PtrLockable<T> p) : m(*p.m), t(p.t) { }
    LockedPtr(std::unique_lock<std::mutex> m, T* t) : m(std::move(m)), t(t) { }
    LockedPtr(LockedPtr&& u) : m(std::move(u.m)), t(u.t) { u.t = nullptr; }
    LockedPtr& operator=(LockedPtr&& u)
    {
        m = std::move(u.m);
        t = u.t;
        u.t = nullptr;
    }

    T& operator*() { return t->t; }
    T* operator->() { return &t->t; }

    void clear() { *this = LockedPtr{}; }

private:
    std::unique_lock<std::mutex> m;
    T* t;
};

template<class T>
struct PtrGuard
{
    PtrGuard(PtrLockable<T> p) : m(&p.m), t(p.t) { m.lock(); }
    PtrGuard(const PtrGuard&) = delete;
    PtrGuard(PtrGuard&&) = delete;
    PtrGuard& operator=(const PtrGuard&) = delete;
    PtrGuard& operator=(PtrGuard&&) = delete;
    ~PtrGuard() { m.unlock(); }

    T& operator*() { return *t; }
    T* operator->() { return t; }

private:
    std::mutex& m;
    T* t;
};

template<class T>
struct LockGuarded
{
    operator PtrLockable<T>() { return {m, &t}; }

private:
    std::mutex m;
    T t;
};
