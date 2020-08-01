#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

struct Worker
{
    std::atomic<int> m_trials = 0;
    std::atomic<float> m_err[200] = {};
    std::atomic<float> m_learn_rate = 0.0002;

    void replace_model(std::unique_ptr<struct IModel> model);
    std::unique_ptr<struct IModel> clone_model();
    void serialize_model(struct RJWriter& w);
    std::string model_name();

    void start();
    void join();

private:
    void work();

    std::thread m_th;
    std::atomic<bool> m_worker_exit = false;

    std::mutex m_mutex;
    struct IModel* m_model = nullptr;
    bool m_replace_model = false;
};
