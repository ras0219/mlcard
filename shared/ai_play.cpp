#include "ai_play.h"

std::pair<int, int> run_n(IModel& m1, IModel& m2, size_t n)
{
    Game g;
    std::vector<Turn> turns;
    int p1_wins = 0;
    int p2_wins = 0;

    for (int x = 0; x < n; ++x)
    {
        g.init();
        turns.clear();
        turns.reserve(40);

        while (g.cur_result() == Game::Result::playing)
        {
            turns.emplace_back();
            auto& turn = turns.back();
            turn.input = g.encode();
            if (g.player2_turn)
            {
                turn.eval = m2.make_eval();
                m2.calc(*turn.eval, turn.input, false);
            }
            else
            {
                turn.eval = m1.make_eval();
                m1.calc(*turn.eval, turn.input, false);
            }

            // choose action to take
            turn.take_ai_action();

            g.advance(turn.chosen_action);
        }
        if (g.cur_result() == Game::Result::p1_win)
            ++p1_wins;
        else if (g.cur_result() == Game::Result::p2_win)
            ++p2_wins;
    }
    return {p1_wins, p2_wins};
}
