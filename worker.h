#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

struct Worker
{
    std::atomic<int> m_trials = 0;
    std::atomic<double> m_err[200] = {};
    std::atomic<double> m_learn_rate = 0.0005;

    void replace_model(std::unique_ptr<struct IModel> model);
    std::unique_ptr<struct IModel> clone_model();
    void serialize_model(struct RJWriter& w);

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
