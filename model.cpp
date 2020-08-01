#include "model.h"
#include "game.h"
#include "modeldims.h"
#include "rjwriter.h"
#include "vec.h"
#include <algorithm>
#include <atomic>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <vector>

using rapidjson::Value;

const Value& find_or_throw(const Value& doc, const char* key)
{
    auto it = doc.FindMember(key);
    if (it == doc.MemberEnd()) throw std::runtime_error(fmt::format("could not find .{}", key));
    return it->value;
}

void deserialize(vec& data, const rapidjson::Value& v)
{
    auto w = v.GetArray();
    data.realloc_uninitialized(w.Size());
    for (unsigned i = 0; i < w.Size(); ++i)
    {
        data[i] = w[i].GetFloat();
    }
}
void serialize(const vec& data, RJWriter& w)
{
    w.StartArray();
    for (auto d : data)
        w.Double(d);
    w.EndArray();
}

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
    // float[4][In+1][Out]
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
            float acc = r_init[j];
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

    void learn(float learn_rate)
    {
        if (m_deltas == 0) return;

        for (size_t i = 0; i < delta().rows(); ++i)
        {
            delta().row(i).mult(1.0f / m_deltas);
            g1s().row(i).decay_average(delta().row(i), 0.1f);
            g2s().row(i).decay_variance(delta().row(i), 0.001f);

            auto coef = coefs().row(i);
            for (size_t j = 0; j < coef.size(); ++j)
            {
                coef[j] -= learn_rate * g1s().row(i)[j] / sqrt(g2s().row(i)[j] + 1e-8f);
            }
        }
    }

    void normalize(float learn_rate)
    {
        auto l1_norm = learn_rate;

        for (auto& e : coefs())
        {
            // L2 normalization
            e *= (1 - learn_rate);
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
        m_data.realloc(m_input * m_output * 4, 0.0f);
        for (auto& v : coefs())
            v = (rand() * 2.0f / RAND_MAX - 1) / m_input;
    }

    void deserialize(const Value& v)
    {
        if (find_or_throw(v, "type") != "Layer") throw "Expected type Layer";
        ::deserialize(m_data, find_or_throw(v, "data"));
        m_deltas = find_or_throw(v, "deltas").GetInt();
        m_input = find_or_throw(v, "input").GetInt();
        m_output = find_or_throw(v, "output").GetInt();
        m_min_io = find_or_throw(v, "min_io").GetInt();
    }
    void serialize(RJWriter& w) const
    {
        w.StartObject();
        w.Key("type");
        w.String("Layer");
        w.Key("data");
        ::serialize(m_data, w);
        w.Key("deltas");
        w.Int(m_deltas);
        w.Key("input");
        w.Int(m_input);
        w.Key("output");
        w.Int(m_output);
        w.Key("min_io");
        w.Int(m_min_io);
        w.EndObject();
    }
};

struct ReLULayer
{
    Layer l;
    Nonlinear n;

    void randomize(int input, int output) { l.randomize(input, output); }

    int in_size() const { return l.in_size(); }
    int inner_size() const { return l.out_size(); }
    int out_size() const { return l.out_size(); }

    void calc(vec_slice in, vec_slice inner, vec_slice out)
    {
        l.calc(in, inner);
        n.calc(inner, out);
    }
    void backprop_init() { l.backprop_init(); }
    void backprop(vec_slice errs, vec_slice in, vec_slice inner, vec_slice out, vec_slice grad)
    {
        VEC_STACK_VEC(tmp, l.out_size());

        n.backprop(tmp, inner, grad);
        l.backprop(errs, in, inner, tmp);
    }

    void learn(float learn_rate) { l.learn(learn_rate); }
    void normalize(float learn_rate) { l.normalize(learn_rate); }

    void deserialize(const Value& v) { l.deserialize(v); }
    void serialize(RJWriter& w) const { l.serialize(w); }
};
struct ReLULayers
{
    struct Eval
    {
        vec m_data;
        int m_inner_size = 0;
        int m_out_size = 0;

        void realloc(int errs_size, int inner_size, int out_size)
        {
            m_inner_size = inner_size;
            m_out_size = out_size;
            m_data.realloc_uninitialized(inner_size + out_size + errs_size);
        }

        vec_slice inner() { return m_data.slice(0, m_inner_size); }
        vec_slice out() { return m_data.slice(m_inner_size, m_out_size); }
        vec_slice errs() { return m_data.slice(m_inner_size + m_out_size); }
    };

    std::vector<ReLULayer> ls;
    int m_inner_size = 0;

    int in_size() const { return ls[0].in_size(); }
    int out_size() const { return ls.back().out_size(); }
    int inner_size() const { return m_inner_size; }

    void randomize(int input, const std::vector<int>& middle, int output)
    {
        m_inner_size = 0;
        for (auto sz : middle)
        {
            ls.emplace_back();
            ls.back().randomize(input, sz);
            input = sz;
            m_inner_size += ls.back().inner_size();
            m_inner_size += sz;
        }
        ls.emplace_back();
        ls.back().randomize(input, output);
        m_inner_size += ls.back().inner_size();
    }

    void backprop_init()
    {
        for (auto& l : ls)
            l.backprop_init();
    }

    void calc(Eval& e, vec_slice in)
    {
        e.realloc(in_size(), inner_size(), out_size());
        this->calc(in, e.inner(), e.out());
    }

    void calc(vec_slice in, vec_slice inner, vec_slice out)
    {
        for (int i = 0; i < ls.size() - 1; ++i)
        {
            auto [cur_inner, x] = inner.split(ls[i].inner_size());
            auto [cur_out, new_inner] = x.split(ls[i].out_size());
            ls[i].calc(in, cur_inner, cur_out);
            in = cur_out;
            inner = new_inner;
        }
        ls.back().calc(in, inner, out);
    }

    void backprop(Eval& e, vec_slice in, vec_slice grad) { this->backprop(e.errs(), in, e.inner(), e.out(), grad); }
    void backprop(vec_slice errs, vec_slice in, vec_slice inner, vec_slice out, vec_slice grad)
    {
        if (ls.size() == 0)
        {
            std::terminate();
        }
        else if (ls.size() == 1)
        {
            ls[0].backprop(errs, in, inner, out, grad);
            return;
        }
        else
        {
            int max_in = 0;
            for (int i = 1; i < ls.size(); ++i)
                max_in += ls[i].in_size();

            VEC_STACK_VEC(tmp, max_in);

            for (intptr_t i = ls.size() - 1; i > 0; --i)
            {
                auto [x, cur_inner] = inner.rsplit(ls[i].inner_size());
                auto [new_inner, cur_in] = x.rsplit(ls[i].in_size());
                auto [new_tmp, cur_errs] = tmp.rsplit(ls[i].in_size());
                ls[i].backprop(cur_errs, cur_in, cur_inner, out, grad);
                grad = cur_errs;
                inner = new_inner;
                out = cur_in;
                tmp = new_tmp;
            }
            ls[0].backprop(errs, in, inner, out, grad);
        }
    }
    void learn(float learn_rate)
    {
        for (auto& l : ls)
            l.learn(learn_rate);
    }
    void normalize(float learn_rate)
    {
        for (auto& l : ls)
            l.normalize(learn_rate);
    }

    void deserialize(const Value& v)
    {
        if (find_or_throw(v, "type") != "RELULayers") throw "Expected type RELULayers";
        m_inner_size = find_or_throw(v, "inner_size").GetInt();
        auto data = find_or_throw(v, "data").GetArray();
        ls.resize(data.Size());
        for (unsigned i = 0; i < data.Size(); ++i)
        {
            ls[i].deserialize(data[i]);
        }
    }

    void serialize(RJWriter& w) const
    {
        w.StartObject();
        w.Key("type");
        w.String("RELULayers");
        w.Key("inner_size");
        w.Int(m_inner_size);
        w.Key("data");
        w.StartArray();
        for (auto&& l : ls)
            l.serialize(w);
        w.EndArray();
        w.EndObject();
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

    void randomize(int input_size, const std::vector<int>& middle, int output_size)
    {
        l.randomize(input_size, middle, output_size);
    }

    void calc(Eval& e, vec_slice input) { l.calc(e.l, input); }
    void backprop_init() { l.backprop_init(); }
    void backprop(Eval& e, vec_slice input) { l.backprop(e.l, input, e.grad); }
    void learn(float learn_rate) { l.learn(learn_rate); }
    void normalize(float learn_rate) { l.normalize(learn_rate); }
    void deserialize(const Value& v) { l.deserialize(v); }
    void serialize(RJWriter& w) const { l.serialize(w); }
};

struct PerYouCardInputModel
{
    ReLULayers l;
    struct Eval
    {
        ReLULayers::Eval l1;
        vec_slice out() { return l1.out(); }
    };

    void randomize(int input_size, const std::vector<int>& middle, int output_size)
    {
        l.randomize(input_size, middle, output_size);
    }

    void calc(Eval& e, vec_slice input) { l.calc(e.l1, input); }
    void backprop_init() { l.backprop_init(); }
    void backprop(Eval& e, vec_slice input, vec_slice grad) { l.backprop(e.l1, input, grad); }
    void learn(float lr) { l.learn(lr); }
    void normalize(float lr) { l.normalize(lr); }
    void deserialize(const Value& v) { l.deserialize(v); }
    void serialize(RJWriter& w) const { l.serialize(w); }
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

    void randomize(int input_size, const std::vector<int>& middle) { l.randomize(input_size, middle, 1); }

    void calc(Eval& e) { l.calc(e.l, e.input); }
    void backprop_init() { l.backprop_init(); }
    void backprop(Eval& e, vec_slice card_grad) { l.backprop(e.l, e.input, card_grad); }
    void learn(float lr) { l.learn(lr); }
    void normalize(float lr) { l.normalize(lr); }
    void deserialize(const Value& v) { l.deserialize(v); }
    void serialize(RJWriter& w) const { l.serialize(w); }
};

struct Model final : IModel
{
    Model(std::string&& s, int i) : IModel(std::move(s), i) { }

    ReLULayers b;
    ReLULayers l;
    Layer p;

    PerCardInputModel card_in_model;
    PerYouCardInputModel you_card_in_model;
    PerCardOutputModel card_out_model;

    int card_out_width = 0;

    struct LInput
    {
        vec data;
        int board_out_width = 0;

        void realloc_uninitialized(int w, int board_out_width)
        {
            data.realloc_uninitialized(w);
            this->board_out_width = board_out_width;
        }

        vec_slice all() { return data.slice(); }
        vec_slice board() { return data.slice(0, board_out_width); }
        vec_slice cards() { return data.slice(board_out_width); }
    };

    struct Eval : IEval
    {
        ReLULayers::Eval b;
        ReLULayers::Eval l;

        LInput l_input;
        vec l_grad;

        std::vector<PerCardInputModel::Eval> cards_in;
        std::vector<PerYouCardInputModel::Eval> you_cards_in;
        std::vector<PerCardOutputModel::Eval> cards_out;

        vec all_out;

        float out_p() { return all_out[0]; }
        float out_card(int i) { return all_out[i + 1]; }
        float max_out() { return all_out.slice(1).max(all_out[0]); }

        virtual vec_slice out() { return all_out; }
        virtual float pct_for_action(int i) override { return all_out[i]; }
        virtual int best_action() override
        {
            int x = 0;
            float win_pct = all_out[0];
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

        virtual float clamped_best_pct() override { return std::max(0.0f, std::min(1.0f, max_out())); }
        virtual float clamped_best_pct(int i, float p) override
        {
            for (int x = 0; x < i; ++x)
                p = std::max(p, all_out[x]);
            for (int x = i + 1; x < all_out.size(); ++x)
                p = std::max(p, all_out[x]);
            return std::max(0.0f, std::min(1.0f, p));
        }
    };

    virtual std::unique_ptr<IEval> make_eval() { return std::make_unique<Eval>(); }

    void randomize(int board_size, int card_size, const ModelDims& dims)
    {
        auto b_dims = dims.children.at("b").dims;
        auto board_out_width = b_dims.back();
        b_dims.pop_back();
        b.randomize(board_size, b_dims, board_out_width);

        auto card_in_dims = dims.children.at("card_in").dims;
        card_out_width = card_in_dims.back();
        card_in_dims.pop_back();
        card_in_model.randomize(card_size, card_in_dims, card_out_width);

        auto you_card_in_dims = dims.children.at("you_card_in").dims;
        you_card_in_model.randomize(card_size, you_card_in_dims, card_out_width);

        auto l_dims = dims.children.at("l").dims;
        auto l3_out_width = l_dims.back();
        l_dims.pop_back();
        l.randomize(board_out_width + card_out_width, l_dims, l3_out_width);

        p.randomize(l3_out_width, 1);
        card_out_model.randomize(l3_out_width + card_out_width, dims.children.at("card_out").dims);
    }

    virtual void calc(IEval& e, Encoded& g, bool full) override { calc_inner((Eval&)e, g, full); }
    void calc_inner(Eval& e, Encoded& g, bool full)
    {
        b.calc(e.b, g.board());
        e.l_input.realloc_uninitialized(b.out_size() + card_out_width, b.out_size());
        e.l_input.board().assign(e.b.out());
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
            e.cards_out[i].input.realloc_uninitialized(l.out_size() + card_out_width);
            e.cards_out[i].input.slice(l.out_size()).assign(e.cards_in[i].out());
            e.cards_out[i].input.slice(0, l.out_size()).assign(e.l.out());
            card_out_model.calc(e.cards_out[i]);
            e.all_out[i + 1] = e.cards_out[i].out()[0];
        }
    }

    virtual void backprop_init() override
    {
        b.backprop_init();
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
            e.l_grad.slice().add(e.cards_out[i].err().slice(0, l.out_size()));
        }
        l.backprop(e.l, e.l_input.all(), e.l_grad);

        auto l_card_errs = e.l.errs().slice(b.out_size());
        if (full)
        {
            for (int i = 0; i < g.you_cards; ++i)
            {
                you_card_in_model.backprop(e.you_cards_in[i], g.you_card(i), l_card_errs);
            }
        }
        for (int i = 0; i < g.me_cards; ++i)
        {
            e.cards_in[i].grad.realloc_uninitialized(e.cards_in[i].out().size());
            e.cards_in[i].grad.slice().assign_add(l_card_errs, e.cards_out[i].err().slice(l.out_size()));
            card_in_model.backprop(e.cards_in[i], g.me_card(i));
        }
        b.backprop(e.b, g.board(), e.l.errs().slice(0, b.out_size()));
    }
    void learn(float lr)
    {
        b.learn(lr);
        l.learn(lr);
        p.learn(lr);

        card_in_model.learn(lr);
        you_card_in_model.learn(lr);
        card_out_model.learn(lr);
    }

    void normalize(float lr)
    {
        b.normalize(lr);
        l.normalize(lr);
        p.normalize(lr);
        card_in_model.normalize(lr);
        you_card_in_model.normalize(lr);
        card_out_model.normalize(lr);
    }

    virtual void serialize(RJWriter& w) const override
    {
        w.StartObject();
        w.Key("type");
        w.String("Model");
        w.Key("name");
        w.String(name().c_str());
        w.Key("b");
        b.serialize(w);
        w.Key("l");
        l.serialize(w);
        w.Key("p");
        p.serialize(w);
        w.Key("in");
        card_in_model.serialize(w);
        w.Key("you_in");
        you_card_in_model.serialize(w);
        w.Key("out");
        card_out_model.serialize(w);
        w.Key("card_out_width");
        w.Int(card_out_width);
        w.EndObject();
    }

    void deserialize(const Value& doc)
    {
        if (find_or_throw(doc, "type") != "Model") throw "Expected type Model";
        b.deserialize(find_or_throw(doc, "b"));
        l.deserialize(find_or_throw(doc, "l"));
        p.deserialize(find_or_throw(doc, "p"));
        card_in_model.deserialize(find_or_throw(doc, "in"));
        you_card_in_model.deserialize(find_or_throw(doc, "you_in"));
        card_out_model.deserialize(find_or_throw(doc, "out"));
        card_out_width = find_or_throw(doc, "card_out_width").GetInt();
    }

    virtual std::unique_ptr<IModel> clone() const { return std::make_unique<Model>(*this); }
};

std::unique_ptr<IModel> make_model(const ModelDims& dims, const std::string& s)
{
    auto m = std::make_unique<Model>(std::string(s), 0);
    m->randomize(Encoded::board_size, Encoded::card_size, dims);
    return m;
}

std::unique_ptr<IModel> load_model(const std::string& s)
{
    rapidjson::Document doc;
    doc.Parse(s.c_str(), s.size());
    auto m = std::make_unique<Model>(doc.FindMember("name")->value.GetString(), doc.FindMember("id")->value.GetInt());
    m->deserialize(doc);
    return m;
}

const ModelDims& default_model_dims()
{
    static ModelDims md{{},
                        {
                            {"b", ModelDims({30, 30})},
                            {"l", ModelDims{{50, 40, 30, 30}}},
                            {"card_in", ModelDims{{20, 20}}},
                            {"you_card_in", ModelDims{{20, 20}}},
                            {"card_out", ModelDims{{20, 20}}},
                        }};
    return md;
}

const ModelDims& medium_model_dims()
{
    static ModelDims md{{},
                        {
                            {"b", ModelDims{{30, 30}}},
                            {"l", ModelDims{{30, 30, 30, 30}}},
                            {"card_in", ModelDims{{20, 20}}},
                            {"you_card_in", ModelDims{{20, 20}}},
                            {"card_out", ModelDims{{20, 20}}},
                        }};
    return md;
}

const ModelDims& small_model_dims()
{
    static ModelDims md{{},
                        {
                            {"b", ModelDims{{6}}},
                            {"l", ModelDims{{6}}},
                            {"card_in", ModelDims{{6}}},
                            {"you_card_in", ModelDims{{6}}},
                            {"card_out", ModelDims{{6}}},
                        }};
    return md;
}
