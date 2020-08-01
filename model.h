#pragma once

#include <memory>
#include <string>

struct Encoded;
struct vec_slice;
struct RJWriter;
struct ModelDims;

struct IEval
{
    virtual ~IEval() { }

    virtual int best_action() = 0;
    virtual double pct_for_action(int i) = 0;
    virtual double clamped_best_pct() = 0;
    virtual double clamped_best_pct(int replace_i, double replace_pct) = 0;
    virtual vec_slice out() = 0;
};

struct IModel
{
    IModel(std::string&& name, int i = 0) : m_name(std::move(name)), id(i) { }
    virtual ~IModel() { }

    virtual std::unique_ptr<IEval> make_eval() = 0;
    virtual void calc(IEval& e, Encoded& input, bool full) = 0;
    virtual void backprop(IEval& e, Encoded& input, vec_slice grad, bool full) = 0;
    virtual void backprop_init() = 0;
    virtual void learn(double learn_rate) = 0;
    virtual void normalize(double learn_rate) = 0;

    virtual std::unique_ptr<IModel> clone() const = 0;
    virtual void serialize(RJWriter& w) const = 0;

    void increment_name() { ++id; }
    std::string name() const { return m_name + "#" + std::to_string(id); }

private:
    std::string m_name;
    int id = 0;
};

const ModelDims& default_model_dims();
const ModelDims& medium_model_dims();
const ModelDims& small_model_dims();

std::unique_ptr<IModel> make_model(const ModelDims& dims, const std::string& s);
std::unique_ptr<IModel> load_model(const std::string& s);
