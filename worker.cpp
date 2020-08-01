#include "worker.h"
#include "ai_play.h"
#include "game.h"
#include "model.h"

static void play_game(Game& g, IModel& m, std::vector<Turn>& turns)
{
    g.init();
    turns.clear();
    turns.reserve(40);

    bool exploregame = (rand() * 1.0 / RAND_MAX) > 0.5;

    while (g.cur_result() == Game::Result::playing)
    {
        turns.emplace_back();
        auto& turn = turns.back();
        turn.input = g.encode();
        turn.eval = m.make_eval();
        turn.eval_full = m.make_eval();
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
}

void Worker::work()
{
    int update_tick = 0;
    int learn_tick = 0;
    int i_err = 0;
    std::unique_ptr<IModel> m;
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        m = m_model->clone();
        m_replace_model = false;
    }
    Game g;

    std::vector<Turn> turns;

    double total_error = 0.0;

    while (!m_worker_exit)
    {
        // double total_m_err = 0.0;
        // for (double x : m_err)
        //    total_m_err += x;
        // if (total_error > (2 * total_m_err / std::size(m_err) + 1e-3))
        //    replay_game(m, turns);
        // else
        play_game(g, *m, turns);

        m->backprop_init();

        // First, fill in the error values
        auto last_player_won =
            turns.size() % 2 == 1 ? (g.cur_result() == Game::Result::p1_win) : (g.cur_result() == Game::Result::p2_win);

        total_error = 0.0;

        auto& turn = turns.back();
        // full model
        auto predicted = turn.eval_full->pct_for_action(turn.chosen_action);
        auto error = predicted - static_cast<double>(last_player_won);
        turn.error_full.realloc(turn.input.avail_actions(), 0.0);
        turn.error_full[turn.chosen_action] = error * 10;

        m->backprop(*turn.eval_full, turn.input, turn.error_full, true);
        total_error += error * error;

        double next_turn_expected = static_cast<double>(last_player_won);
        for (int i = (int)turns.size() - 2; i >= 0; --i)
        {
            auto& turn = turns[i];
            auto& next_turn = turns[i + 1];
            auto predicted = turn.eval_full->pct_for_action(turn.chosen_action);
            auto expected = (1.0 - next_turn.eval_full->clamped_best_pct(next_turn.chosen_action, next_turn_expected));
            auto error = predicted - expected;
            turn.error_full.realloc(turn.input.avail_actions(), 0.0);
            turn.error_full[turn.chosen_action] = error * 10;
            m->backprop(*turn.eval_full, turn.input, turn.error_full, true);
            total_error += error * error;
            next_turn_expected = expected;
        }

        for (auto&& turn : turns)
        {
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
        if (learn_tick % 10 == 9) m->learn(m_learn_rate);
        if (learn_tick % 1000 == 999) m->normalize(m_learn_rate);

        update_tick++;
        if (update_tick >= 100)
        {
            update_tick = 0;
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
        }
        ++m_trials;
    }
}
