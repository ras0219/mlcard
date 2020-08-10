#include "worker.h"
#include "ai_play.h"
#include "game.h"
#include "kv_range.h"
#include "model.h"

static unsigned int play_game(Game& g, IModel& m, std::vector<Turn>& turns)
{
    g.init();
    unsigned int turn_count = 0;
    bool exploregame = (rand() * 1.0 / RAND_MAX) > 0.5;

    while (g.cur_result() == Game::Result::playing)
    {
        turn_count++;

        if (turn_count > turns.size()) turns.emplace_back();

        auto& turn = turns[turn_count - 1];
        turn.input = g.encode();
        turn.player2_turn = g.player2_turn;
        if (!turn.eval) turn.eval = m.make_eval();
        if (!turn.eval_full) turn.eval_full = m.make_eval();
        m.calc(*turn.eval, turn.input, false);
        m.calc(*turn.eval_full, turn.input, true);

        if (exploregame)
        {
            // choose action to take
            auto r = rand() * 1.0 / RAND_MAX;
            if (r < 0.3)
            {
                turn.chosen_action = static_cast<int>(r * turn.input.avail_actions() / 0.3);
            }
            else
            {
                turn.take_ai_action();
            }
        }
        else
        {
            turn.take_full_ai_action();
        }

        g.advance(turn.chosen_action);
    }
    return turn_count;
}

static void replay_game(IModel& m, std::vector<Turn>& turns)
{
    for (auto&& turn : turns)
    {
        m.calc(*turn.eval, turn.input, false);
        m.calc(*turn.eval_full, turn.input, true);
    }
}

void Worker::replace_model(std::unique_ptr<IModel> model)
{
    std::lock_guard<std::mutex> lk(m_mutex);
    delete m_model;
    m_model = model.release();
    m_replace_model = true;
}

std::unique_ptr<IModel> Worker::clone_model()
{
    std::lock_guard<std::mutex> lk(m_mutex);
    return m_model->clone();
}

std::string Worker::model_name()
{
    std::lock_guard<std::mutex> lk(m_mutex);
    return m_model ? m_model->name() : "none";
}

void Worker::replace_compete_baseline(std::shared_ptr<IModel> m)
{
    std::lock_guard lk(m_mutex);
    m_compete_baseline = std::move(m);
    if (!m_compete_th.joinable())
    {
        m_compete_th = std::thread(&Worker::compete_baseline_work, this);
    }
}

void Worker::serialize_model(struct RJWriter& w)
{
    std::lock_guard<std::mutex> lk(m_mutex);
    m_model->serialize(w);
}

void Worker::start() { m_th = std::thread(&Worker::work, this); }
void Worker::join()
{
    m_worker_exit = true;
    m_th.join();
    {
        std::lock_guard lk(m_mutex);
        m_past_models_cv.notify_one();
    }
    if (m_compete_th.joinable()) m_compete_th.join();
}

#if defined(_WIN32)
extern "C"
{
    __declspec(dllimport) void __stdcall SetThreadPriority(void*, int);
    __declspec(dllimport) void* __stdcall GetCurrentThread();
}
#endif
void Worker::work()
{
#if defined(_WIN32)
    SetThreadPriority(GetCurrentThread(), /*THREAD_MODE_BACKGROUND_BEGIN*/ 0x00010000);
#endif
    int update_tick = 0;
    int learn_tick = 0;
    int i_err = 0;
    size_t i_compete_model = 0;
    std::unique_ptr<IModel> m;
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        m = m_model->clone();
        m_replace_model = false;
        m_past_models.resize(compete_size);
    }
    Game g;
    unsigned int turn_count = 0;
    std::vector<Turn> turns;
    turns.resize(40);

    float total_error = 0.0f;

    m->backprop_init();

    while (!m_worker_exit)
    {
        turn_count = play_game(g, *m, turns);

        // First, fill in the error values
        auto& turn = turns[turn_count - 1];
        auto last_player_won = g.cur_result() == (turn.player2_turn ? Game::Result::p2_win : Game::Result::p1_win);

        total_error = 0.0;

        // full model
        auto predicted = turn.eval_full->pct_for_action(turn.chosen_action);
        auto error = predicted - static_cast<float>(last_player_won);
        turn.error_full.realloc(turn.input.avail_actions(), 0.0f);
        turn.error_full[turn.chosen_action] = error * turn.input.avail_actions();

        m->backprop(*turn.eval_full, turn.input, turn.error_full, true);
        total_error += error * error;

        auto next_turn_expected = static_cast<float>(last_player_won);
        for (int i = (int)turn_count - 2; i >= 0; --i)
        {
            auto& turn = turns[i];
            auto& next_turn = turns[i + 1];
            auto predicted = turn.eval_full->pct_for_action(turn.chosen_action);
            auto expected = next_turn.eval_full->clamped_best_pct(next_turn.chosen_action, next_turn_expected);
            if (next_turn.player2_turn != turn.player2_turn)
            {
                expected = 1.0f - expected;
            }
            auto error = predicted - expected;
            turn.error_full.realloc(turn.input.avail_actions(), 0.0);
            turn.error_full[turn.chosen_action] = error * turn.input.avail_actions();
            m->backprop(*turn.eval_full, turn.input, turn.error_full, true);
            total_error += error * error;
            next_turn_expected = expected;
        }

        for (unsigned i = 0; i < turn_count; i++)
        {
            auto&& turn = turns[i];
            turn.error.realloc_uninitialized(turn.input.avail_actions());
            turn.error.slice().assign_sub(turn.eval->out(), turn.eval_full->out());
            m->backprop(*turn.eval, turn.input, turn.error, false);
            total_error += turn.error.slice().dot(turn.error);
        }

        //// now learn

        m_err[i_err] = total_error;
        i_err = (i_err + 1ULL) % std::size(m_err);

        learn_tick++;
        if (learn_tick >= 10000) learn_tick = 0;
        if (learn_tick % 10 == 9)
        {
            m->learn(m_learn_rate);
            m->backprop_init();
        }
        if (learn_tick % 200 == 199) m->normalize(m_learn_rate * 1e-9f);

        update_tick++;
        if (update_tick >= 300)
        {
            update_tick = 0;
            {
                std::lock_guard<std::mutex> lk(m_mutex);
                if (m_replace_model)
                {
                    m = m_model->clone();
                    m_replace_model = false;
                }
                else
                {
                    m->increment_name();
                    delete m_model;
                    m_model = m->clone().release();
                }
                m_past_models[i_compete_model] = m->clone();
                m_past_models_cv.notify_one();
            }
            i_compete_model++;
            i_compete_model %= compete_size;
        }
        ++m_trials;
    }
}
void Worker::compete_baseline_work()
{
    std::vector<std::shared_ptr<IModel>> past_models_copy(compete_size);
    std::shared_ptr<IModel> compete_baseline;
    while (true)
    {
        std::unique_lock lk(m_mutex);
        if (m_worker_exit) return;
        m_past_models_cv.wait(lk);
        if (m_worker_exit) return;

        intptr_t i = 0;
        if (m_compete_baseline)
        {
            compete_baseline = std::move(m_compete_baseline);
            auto m = m_past_models[0];
            lk.unlock();
            past_models_copy.assign(compete_size, nullptr);
            past_models_copy[0] = std::move(m);
        }
        else
        {
            auto [it1, it2] = std::mismatch(
                m_past_models.begin(), m_past_models.end(), past_models_copy.begin(), past_models_copy.end());
            if (it2 != past_models_copy.end()) *it2 = *it1;
            lk.unlock();
            i = it2 - past_models_copy.begin();
        }

        if (i < past_models_copy.size() && past_models_copy[i])
        {
            auto wins = 0;
            auto losses = 0;
            for (int x = 0; x < 10; ++x)
            {
                auto [w, l] = run_n(*past_models_copy[i], *compete_baseline, 10);
                auto [l2, w2] = run_n(*compete_baseline, *past_models_copy[i], 10);
                wins += w + w2;
                losses += l + l2;
                if (wins + losses == 0)
                    m_compete_results[i] = 0;
                else
                    m_compete_results[i] = wins * 1.0f / (wins + losses);
            }
        }
    }
}