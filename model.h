#pragma once

#include <memory>
#include <string>

struct Encoded;
struct vec_slice;
struct RJWriter;

struct IEval
{
    virtual ~IEval() { }

    virtual int best_action() = 0;
    virtual double pct_for_action(int i) = 0;
    virtual double clamped_best_pct() = 0;
    virtual vec_slice out() = 0;
};

struct IModel
{
    virtual ~IModel() { }

    virtual std::unique_ptr<IEval> make_eval() = 0;
    virtual void calc(IEval& e, Encoded& input, bool full) = 0;
    virtual void backprop(IEval& e, Encoded& input, vec_slice grad, bool full) = 0;
    virtual void backprop_init() = 0;
    virtual void learn(double learn_rate) = 0;
    virtual void normalize(double learn_rate) = 0;

    virtual std::unique_ptr<IModel> clone() const = 0;
    virtual void serialize(RJWriter& w) const = 0;
};

std::unique_ptr<IModel> make_model();
std::unique_ptr<IModel> load_model(const std::string& s);
