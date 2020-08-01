#pragma once

#include "game.h"
#include "model.h"
#include "vec.h"
#include <memory>
#include <vector>

struct Turn
{
    Encoded input;

    std::unique_ptr<IEval> eval, eval_full;
    int chosen_action;
    vec error, error_full;

    void take_ai_action() { chosen_action = eval->best_action(); }
    void take_full_ai_action() { chosen_action = eval_full->best_action(); }
};
