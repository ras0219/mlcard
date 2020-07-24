#include "model.h"
#include "game.h"
#include "vec.h"
#include <algorithm>
#include <atomic>
#include <rapidjson/writer.h>
#include <vector>

struct Nonlinear
{
    struct Eval
    {
        // double[N]
        vec out;
        // double[N]
        vec errs;
    };

    vec_slice calc(Eval& e, vec_slice input) const
    {
        e.out.realloc_uninitialized(input.size());
        for (int i = 0; i < input.size(); ++i)
        {
            e.out[i] = input[i] < 0 ? input[i] / 10 : input[i];
        }
        return e.out;
    }
    vec_slice backprop(Eval& e, vec_slice input, vec_slice grad) const
    {
        e.errs.realloc_uninitialized(grad.size());
        for (int i = 0; i < input.size(); ++i)
        {
            e.errs[i] = input[i] < 0 ? grad[i] / 10 : grad[i];
        }
        return e.errs;
    }
};

struct Layer
{
    // double[4][In+1][Out]
    vec m_data;

    int num_deltas = 0;
    int input_size = 0;
    int output_size = 0;

    mat_slice coefs() { return mat_slice(m_data.data(), input_size, output_size); }
    mat_slice g1s() { return mat_slice(m_data.data() + input_size * output_size, input_size, output_size); }
    mat_slice g2s() { return mat_slice(m_data.data() + 2 * input_size * output_size, input_size, output_size); }
    mat_slice delta() { return mat_slice(m_data.data() + 3 * input_size * output_size, input_size, output_size); }

    struct Eval
    {
        // double[In]
        vec errs;

        // double[Out]
        vec out;
    };

    vec_slice calc(Eval& e, vec_slice input)
    {
        e.out.alloc_assign(coefs().row(input.size()));

        size_t n = std::min(e.out.size(), input.size());
        e.out.slice(0, n).add(input.slice(0, n));

        vec_slice o(e.out);
        for (size_t i = 0; i < input.size(); ++i)
            o.fma(coefs().row(i), input[i]);

        return e.out;
    }

    void backprop_init()
    {
        num_deltas = 0;
        delta().flat().assign(0.0);
    }

    vec_slice backprop(Eval& e, vec_slice input, vec_slice grad)
    {
        e.errs.realloc_uninitialized(input.size());
        for (size_t j = 0; j < input.size(); ++j)
        {
            e.errs.slice()[j] = grad.dot(coefs().row(j));
            delta().row(j).fma(grad, input[j]);
        }

        size_t n = std::min(e.errs.size(), grad.size());
        e.errs.slice(0, n).add(grad.slice(0, n));

        delta().last_row().add(grad);

        ++num_deltas;
        return e.errs;
    }

    void learn(double learn_rate)
    {
        if (num_deltas == 0) return;

        for (size_t i = 0; i < delta().rows(); ++i)
        {
            delta().row(i).mult(1.0 / num_deltas);
            g1s().row(i).decay_average(delta().row(i), 0.1);
            g2s().row(i).decay_variance(delta().row(i), 0.001);

            auto coef = coefs().row(i);
            for (size_t j = 0; j < coef.size(); ++j)
            {
                coef[j] -= learn_rate * g1s().row(i)[j] / sqrt(g2s().row(i)[j] + 1e-8);
            }
        }
    }

    void normalize(double learn_rate)
    {
        auto l1_norm = 1e-11 * learn_rate;

        for (auto& e : coefs())
        {
            // L2 normalization
            e *= (1 - 1e-11 * learn_rate);
            // L1 normalization
            if (e < -l1_norm)
                e += l1_norm;
            else if (e > l1_norm)
                e -= l1_norm;
            else
                e = 0;
        }
    }

    void randomize(int input, int output)
    {
        input_size = input + 1;
        output_size = output;
        m_data.realloc(input_size * output_size * 4, 0.0);
        for (auto& v : coefs())
            v = (rand() * 2.0 / RAND_MAX - 1) / input_size;
    }
};

struct ReLULayer
{
    Layer l;
    Nonlinear n;

    struct Eval
    {
        Layer::Eval l;
        Nonlinear::Eval n;

        vec_slice out() { return n.out; }
        vec_slice errs() { return l.errs; }
    };

    void randomize(int input, int output) { l.randomize(input, output); }

    vec_slice calc(Eval& e, vec_slice input)
    {
        l.calc(e.l, input);
        return n.calc(e.n, e.l.out);
    }
    void backprop_init() { l.backprop_init(); }
    vec_slice backprop(Eval& e, vec_slice input, vec_slice grad)
    {
        n.backprop(e.n, e.l.out, grad);
        return l.backprop(e.l, input, e.n.errs);
    }

    void learn(double learn_rate) { l.learn(learn_rate); }
    void normalize(double learn_rate) { l.normalize(learn_rate); }
};
struct ReLULayers
{
    std::vector<ReLULayer> ls;

    struct Eval
    {
        std::vector<ReLULayer::Eval> l;

        auto out() { return l.back().out(); }
        auto errs() { return l.front().errs(); }
    };

    void randomize(int input, std::initializer_list<int> middle, int output)
    {
        for (auto sz : middle)
        {
            ls.emplace_back();
            ls.back().randomize(input, sz);
            input = sz;
        }
        ls.emplace_back();
        ls.back().randomize(input, output);
    }

    void backprop_init()
    {
        for (auto& l : ls)
            l.backprop_init();
    }

    vec_slice calc(Eval& e, vec_slice in)
    {
        e.l.resize(ls.size());
        for (int i = 0; i < ls.size(); ++i)
        {
            ls[i].calc(e.l[i], in);
            in = e.l[i].out();
        }
        return in;
    }
    void backprop(Eval& e, vec_slice in, vec_slice grad)
    {
        for (intptr_t i = ls.size() - 1; i > 0; --i)
        {
            ls[i].backprop(e.l[i], e.l[i - 1].out(), grad);
            grad = e.l[i].errs();
        }
        ls[0].backprop(e.l[0], in, grad);
    }
    void learn(double learn_rate)
    {
        for (auto& l : ls)
            l.learn(learn_rate);
    }
    void normalize(double learn_rate)
    {
        for (auto& l : ls)
            l.normalize(learn_rate);
    }
};

struct PerCardInputModel
{
    ReLULayers l;
    struct Eval
    {
        vec grad;
        ReLULayers::Eval l;
        vec_slice out() { return l.out(); }
    };

    void randomize(int input_size, int output_size) { l.randomize(input_size, {8}, output_size); }

    void calc(Eval& e, vec_slice input) { l.calc(e.l, input); }
    void backprop_init() { l.backprop_init(); }
    void backprop(Eval& e, vec_slice input) { l.backprop(e.l, input, e.grad); }
    void learn(double learn_rate) { l.learn(learn_rate); }
    void normalize(double learn_rate) { l.normalize(learn_rate); }
};

struct PerYouCardInputModel
{
    ReLULayers l;
    struct Eval
    {
        ReLULayers::Eval l1;
        vec_slice out() { return l1.out(); }
    };

    void randomize(int input_size, int output_size) { l.randomize(input_size, {8}, output_size); }

    void calc(Eval& e, vec_slice input) { l.calc(e.l1, input); }
    void backprop_init() { l.backprop_init(); }
    void backprop(Eval& e, vec_slice input, vec_slice grad) { l.backprop(e.l1, input, grad); }
    void learn(double lr) { l.learn(lr); }
    void normalize(double lr) { l.normalize(lr); }
};

struct PerCardOutputModel
{
    ReLULayers l;
    Layer l2;
    struct Eval
    {
        vec input;
        ReLULayers::Eval l;
        Layer::Eval l2;
        vec_slice out() { return l2.out; }
        vec_slice err() { return l.errs(); }
    };

    void randomize(int input_size)
    {
        l.randomize(input_size, {8, 8}, 8);
        l2.randomize(8, 1);
    }

    void calc(Eval& e)
    {
        l.calc(e.l, e.input);
        l2.calc(e.l2, e.l.out());
    }
    void backprop_init()
    {
        l.backprop_init();
        l2.backprop_init();
    }
    void backprop(Eval& e, vec_slice card_grad)
    {
        l2.backprop(e.l2, e.l.out(), card_grad);
        l.backprop(e.l, e.input, e.l2.errs);
    }
    void learn(double lr)
    {
        l2.learn(lr);
        l.learn(lr);
    }
    void normalize(double lr)
    {
        l2.normalize(lr);
        l.normalize(lr);
    }
};

struct Model final : IModel
{
    ReLULayer b1;
    ReLULayers l;
    Layer p;

    PerCardInputModel card_in_model;
    PerYouCardInputModel you_card_in_model;
    PerCardOutputModel card_out_model;

    struct Eval : IEval
    {
        ReLULayer::Eval b1;
        ReLULayers::Eval l;
        Layer::Eval p;

        vec l_input;
        vec l_grad;

        std::vector<PerCardInputModel::Eval> cards_in;
        std::vector<PerYouCardInputModel::Eval> you_cards_in;
        std::vector<PerCardOutputModel::Eval> cards_out;

        vec all_out;

        double out_p() { return all_out[0]; }
        double out_card(int i) { return all_out[i + 1]; }
        double max_out() { return all_out.slice(1).max(all_out[0]); }

        virtual vec_slice out() { return all_out; }
        virtual double pct_for_action(int i) override { return all_out[i]; }
        virtual int best_action() override
        {
            int x = 0;
            double win_pct = all_out[0];
            for (int i = 1; i < all_out.size(); ++i)
            {
                auto p = all_out[i];
                if (p > win_pct)
                {
                    x = i;
                    win_pct = p;
                }
            }
            return x;
        }

        virtual double clamped_best_pct() override { return std::max(0.0, std::min(1.0, max_out())); }
    };

    virtual std::unique_ptr<IEval> make_eval() { return std::make_unique<Eval>(); }

    static constexpr int card_out_width = 8;
    static constexpr int board_out_width = 10;
    static constexpr int l3_out_width = 18;

    void randomize(int board_size, int card_size)
    {
        b1.randomize(board_size, board_out_width);
        l.randomize(board_out_width + card_out_width, {20, 22, 24, 26}, l3_out_width);
        p.randomize(l3_out_width, 1);
        card_in_model.randomize(card_size, card_out_width);
        card_out_model.randomize(l3_out_width + card_out_width);
        you_card_in_model.randomize(card_size, card_out_width);
    }

    virtual void calc(IEval& e, Encoded& g, bool full) override { calc_inner((Eval&)e, g, full); }
    void calc_inner(Eval& e, Encoded& g, bool full)
    {
        b1.calc(e.b1, g.board());
        e.l_input.realloc_uninitialized(board_out_width + card_out_width);
        e.l_input.slice(0, board_out_width).assign(e.b1.out());
        auto l_input_cards = e.l_input.slice(board_out_width);
        l_input_cards.assign(0);
        e.cards_in.resize(g.me_cards);
        e.cards_out.resize(g.me_cards);
        e.you_cards_in.resize(g.you_cards);

        for (int i = 0; i < g.me_cards; ++i)
        {
            card_in_model.calc(e.cards_in[i], g.me_card(i));
            l_input_cards.add(e.cards_in[i].out());
            e.cards_out[i].input.realloc_uninitialized(l3_out_width + card_out_width);
            e.cards_out[i].input.slice(l3_out_width).assign(e.cards_in[i].out());
        }
        if (full)
        {
            for (int i = 0; i < g.you_cards; ++i)
            {
                you_card_in_model.calc(e.you_cards_in[i], g.you_card(i));
                l_input_cards.add(e.you_cards_in[i].out());
            }
        }
        l.calc(e.l, e.l_input);
        p.calc(e.p, e.l.out());
        e.all_out.realloc_uninitialized(g.avail_actions());
        e.all_out[0] = e.p.out[0];
        for (int i = 0; i < g.me_cards; ++i)
        {
            e.cards_out[i].input.slice(0, l3_out_width).assign(e.l.out());
            card_out_model.calc(e.cards_out[i]);
            e.all_out[i + 1] = e.cards_out[i].out()[0];
        }
    }

    virtual void backprop_init() override
    {
        b1.backprop_init();
        l.backprop_init();
        p.backprop_init();

        card_in_model.backprop_init();
        you_card_in_model.backprop_init();
        card_out_model.backprop_init();
    }

    virtual void backprop(IEval& e, Encoded& g, vec_slice grad, bool full) override
    {
        backprop_inner((Eval&)e, g, grad, full);
    }
    void backprop_inner(Eval& e, Encoded& g, vec_slice grad, bool full)
    {
        vec_slice p_grad = grad.slice(0, 1);
        vec_slice cards_grad = grad.slice(1);

        p.backprop(e.p, e.l.out(), p_grad);

        e.l_grad.alloc_assign(e.p.errs);

        for (int i = 0; i < cards_grad.size(); ++i)
        {
            card_out_model.backprop(e.cards_out[i], cards_grad.slice(i, 1));
            e.l_grad.slice().add(e.cards_out[i].err().slice(0, l3_out_width));
        }
        l.backprop(e.l, e.l_input, e.l_grad);

        auto l_card_errs = e.l.errs().slice(board_out_width);
        if (full)
        {
            for (int i = 0; i < g.you_cards; ++i)
            {
                you_card_in_model.backprop(e.you_cards_in[i], g.you_card(i), l_card_errs);
            }
        }
        for (int i = 0; i < g.me_cards; ++i)
        {
            e.cards_in[i].grad.realloc_uninitialized(card_out_width);
            e.cards_in[i].grad.slice().assign_add(l_card_errs, e.cards_out[i].err().slice(l3_out_width));
            card_in_model.backprop(e.cards_in[i], g.me_card(i));
        }
        b1.backprop(e.b1, g.board(), e.l.errs().slice(0, board_out_width));
    }
    void learn(double lr)
    {
        b1.learn(lr);
        l.learn(lr);
        p.learn(lr);

        card_in_model.learn(lr);
        you_card_in_model.learn(lr);
        card_out_model.learn(lr);
    }

    void normalize(double lr)
    {
        b1.normalize(lr);
        l.normalize(lr);
        p.normalize(lr);
        card_in_model.normalize(lr);
        you_card_in_model.normalize(lr);
        card_out_model.normalize(lr);
    }

    void serialize(rapidjson::Writer<rapidjson::StringBuffer>& w)
    {
        w.StartObject();
        w.Key("type");
        w.String("Model");
        w.EndObject();
    }

    virtual std::unique_ptr<IModel> clone() const { return std::make_unique<Model>(*this); }
};

std::unique_ptr<IModel> make_model()
{
    auto m = std::make_unique<Model>();
    m->randomize(Encoded::board_size, Encoded::card_size);
    return m;
}
