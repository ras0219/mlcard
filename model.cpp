#include "model.h"
#include "game.h"
#include "vec.h"
#include <algorithm>
#include <atomic>
#include <rapidjson/writer.h>
#include <vector>

struct Nonlinear
{
    void calc(vec_slice in, vec_slice out) const
    {
        for (int i = 0; i < in.size(); ++i)
        {
            out[i] = in[i] < 0 ? in[i] / 10 : in[i];
        }
    }
    void backprop(vec_slice errs, vec_slice in, vec_slice grad) const
    {
        for (int i = 0; i < in.size(); ++i)
        {
            errs[i] = in[i] < 0 ? grad[i] / 10 : grad[i];
        }
    }
};

struct Layer
{
    // double[4][In+1][Out]
    vec m_data;

    int m_deltas = 0;
    int m_input = 0;
    int m_output = 0;
    int m_min_io = 0;

    mat_slice coefs() { return mat_slice(m_data.data(), m_input, m_output); }
    mat_slice g1s() { return mat_slice(m_data.data() + m_input * m_output, m_input, m_output); }
    mat_slice g2s() { return mat_slice(m_data.data() + 2 * m_input * m_output, m_input, m_output); }
    mat_slice delta() { return mat_slice(m_data.data() + 3 * m_input * m_output, m_input, m_output); }

    int out_size() const { return m_output; }
    int in_size() const { return m_input - 1; }

    void calc(vec_slice input, vec_slice out)
    {
        auto c = coefs();
        auto r_init = c.row(m_input - 1);

        for (size_t j = 0; j < m_output; ++j)
        {
            double acc = r_init[j];
            for (size_t i = 0; i < m_input - 1; ++i)
            {
                acc += coefs().row(i)[j] * input[i];
            }
            out[j] = acc;
        }

        out.slice(0, m_min_io).add(input.slice(0, m_min_io));
    }

    void backprop_init()
    {
        m_deltas = 0;
        delta().flat().assign(0.0);
    }

    void backprop(vec_slice errs, vec_slice input, vec_slice out, vec_slice grad)
    {
        for (size_t j = 0; j < m_input - 1; ++j)
        {
            errs[j] = grad.dot(coefs().row(j));
            delta().row(j).fma(grad, input[j]);
        }

        errs.slice(0, m_min_io).add(grad.slice(0, m_min_io));

        delta().last_row().add(grad);

        ++m_deltas;
    }

    void learn(double learn_rate)
    {
        if (m_deltas == 0) return;

        for (size_t i = 0; i < delta().rows(); ++i)
        {
            delta().row(i).mult(1.0 / m_deltas);
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
        m_input = input + 1;
        m_output = output;
        m_min_io = std::min(input, output);
        m_data.realloc(m_input * m_output * 4, 0.0);
        for (auto& v : coefs())
            v = (rand() * 2.0 / RAND_MAX - 1) / m_input;
    }
};

struct ReLULayer
{
    Layer l;
    Nonlinear n;

    struct Eval
    {
        int m_out;

        vec m_calc;
        vec m_errs;

        vec_slice out() { return m_calc.slice(m_out); }
        vec_slice errs() { return m_errs.slice(m_out); }

        vec_slice l_out() { return m_calc.slice(0, m_out); }
        vec_slice n_errs() { return m_errs.slice(0, m_out); }
    };

    void randomize(int input, int output) { l.randomize(input, output); }

    void calc(Eval& e, vec_slice input)
    {
        e.m_out = l.out_size();
        e.m_calc.realloc_uninitialized(l.out_size() * 2);
        l.calc(input, e.l_out());
        n.calc(e.l_out(), e.out());
    }
    void backprop_init() { l.backprop_init(); }
    void backprop(Eval& e, vec_slice input, vec_slice grad)
    {
        e.m_errs.realloc_uninitialized(l.out_size() + l.in_size());
        n.backprop(e.n_errs(), e.l_out(), grad);
        l.backprop(e.errs(), input, e.l_out(), e.n_errs());
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
    struct Eval
    {
        vec input;
        ReLULayers::Eval l;
        vec_slice out() { return l.out(); }
        vec_slice err() { return l.errs(); }
    };

    void randomize(int input_size) { l.randomize(input_size, {8, 8, 8}, 1); }

    void calc(Eval& e) { l.calc(e.l, e.input); }
    void backprop_init() { l.backprop_init(); }
    void backprop(Eval& e, vec_slice card_grad) { l.backprop(e.l, e.input, card_grad); }
    void learn(double lr) { l.learn(lr); }
    void normalize(double lr) { l.normalize(lr); }
};

struct Model final : IModel
{
    ReLULayer b1;
    ReLULayers l;
    Layer p;

    PerCardInputModel card_in_model;
    PerYouCardInputModel you_card_in_model;
    PerCardOutputModel card_out_model;

    static constexpr int card_out_width = 8;
    static constexpr int board_out_width = 10;
    static constexpr int l3_out_width = 18;

    struct LInput
    {
        vec data;

        void realloc_uninitialized() { data.realloc_uninitialized(board_out_width + card_out_width); }

        vec_slice all() { return data.slice(); }
        vec_slice board() { return data.slice(0, board_out_width); }
        vec_slice cards() { return data.slice(board_out_width); }
    };

    struct Eval : IEval
    {
        ReLULayer::Eval b1;
        ReLULayers::Eval l;

        LInput l_input;
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
        e.l_input.realloc_uninitialized();
        e.l_input.board().assign(e.b1.out());
        auto l_input_cards = e.l_input.cards();
        l_input_cards.assign(0);
        e.cards_in.resize(g.me_cards);
        e.cards_out.resize(g.me_cards);
        e.you_cards_in.resize(g.you_cards);

        for (int i = 0; i < g.me_cards; ++i)
        {
            card_in_model.calc(e.cards_in[i], g.me_card(i));
            l_input_cards.add(e.cards_in[i].out());
        }
        if (full)
        {
            for (int i = 0; i < g.you_cards; ++i)
            {
                you_card_in_model.calc(e.you_cards_in[i], g.you_card(i));
                l_input_cards.add(e.you_cards_in[i].out());
            }
        }
        l.calc(e.l, e.l_input.all());
        e.all_out.realloc_uninitialized(g.avail_actions());
        p.calc(e.l.out(), e.all_out.slice(0, 1));
        for (int i = 0; i < g.me_cards; ++i)
        {
            e.cards_out[i].input.realloc_uninitialized(l3_out_width + card_out_width);
            e.cards_out[i].input.slice(l3_out_width).assign(e.cards_in[i].out());
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

        e.l_grad.realloc_uninitialized(p.in_size());
        p.backprop(e.l_grad, e.l.out(), e.all_out.slice(0, 1), p_grad);

        for (int i = 0; i < cards_grad.size(); ++i)
        {
            card_out_model.backprop(e.cards_out[i], cards_grad.slice(i, 1));
            e.l_grad.slice().add(e.cards_out[i].err().slice(0, l3_out_width));
        }
        l.backprop(e.l, e.l_input.all(), e.l_grad);

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
