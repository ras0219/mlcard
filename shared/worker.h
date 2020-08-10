#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "lock_guarded.h"

struct IModel;

struct Worker
{
    std::atomic<int> m_trials = 0;
    std::atomic<float> m_err[200] = {};
    std::atomic<float> m_learn_rate = 0.0002;
    static constexpr size_t compete_size = 200;
    std::atomic<float> m_compete_results[compete_size] = {};

    void replace_model(std::unique_ptr<struct IModel> model);
    std::unique_ptr<struct IModel> clone_model();
    void serialize_model(struct RJWriter& w);
    std::string model_name();

    void start();
    void join();

    void replace_compete_baseline(std::shared_ptr<IModel> m);

private:
    void work();

    std::thread m_th;
    std::atomic<bool> m_worker_exit = false;

    std::vector<std::shared_ptr<IModel>> m_past_models;
    size_t m_compete_idx;
    std::shared_ptr<IModel> m_local_compete_baseline;

    std::mutex m_mutex;
    struct IModel* m_model = nullptr;
    bool m_replace_model = false;
    std::shared_ptr<IModel> m_compete_baseline;
    std::atomic<bool> m_compete_baseline_changed = false;
};
