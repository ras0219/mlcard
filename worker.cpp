#include "worker.h"
#include "ai_play.h"
#include "game.h"
#include "model.h"

std::mutex s_mutex;
std::atomic<bool> s_worker_exit = false;
std::atomic<double> s_err[200] = {0};
std::atomic<double> s_learn_rate = 0.0005;
bool s_updated = false;
bool s_replace_model = false;
std::atomic<int> s_trials = 0;
struct IModel* s_model;

static void play_game(Game& g, IModel& m, std::vector<Turn>& turns)
{
    g.init();
    turns.clear();
    turns.reserve(40);

    while (g.cur_result() == Game::Result::playing)
    {
        turns.emplace_back();
        auto& turn = turns.back();
        turn.input = g.encode();
        turn.eval = m.make_eval();
        turn.eval_full = m.make_eval();
        m.calc(*turn.eval, turn.input, false);
        m.calc(*turn.eval_full, turn.input, true);

        // choose action to take
        auto r = rand() * 1.0 / RAND_MAX;
        if (r < 0.2)
        {
            turn.chosen_action = static_cast<int>(r * turn.input.avail_actions() / 0.2);
        }
        else
        {
            turn.take_ai_action();
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

void worker()
{
    int update_tick = 0;
    int i_err = 0;
    std::unique_ptr<IModel> m;
    {
        std::lock_guard<std::mutex> lk(s_mutex);
        m = s_model->clone();
    }
    Game g;

    std::vector<Turn> turns;

    double total_error = 0.0;

    while (!s_worker_exit)
    {
        // double total_s_err = 0.0;
        // for (double x : s_err)
        //    total_s_err += x;
        // if (total_error > (2 * total_s_err / std::size(s_err) + 1e-3))
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

        m->learn(s_learn_rate);

        //// now learn

        s_err[i_err] = total_error;
        i_err = (i_err + 1ULL) % std::size(s_err);

        update_tick = (update_tick + 1) % 100;
        if (update_tick == 0)
        {
            m->normalize(s_learn_rate);
            std::lock_guard<std::mutex> lk(s_mutex);
            if (s_replace_model)
            {
                m = s_model->clone();
                s_replace_model = false;
            }
            else
            {
                delete s_model;
                s_model = m->clone().release();
                s_updated = true;
            }
        }
        ++s_trials;
    }
}
