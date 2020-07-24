#pragma once

#include <atomic>
#include <mutex>

extern std::mutex s_mutex;
extern struct IModel* s_model;
extern bool s_updated;
extern std::atomic<int> s_trials;
extern std::atomic<bool> s_worker_exit;
extern std::atomic<double> s_err[200];
extern std::atomic<double> s_learn_rate;

void worker();
