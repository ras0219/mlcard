#define NOMINMAX
//#define VEC_ENABLE_CHECKS

#include "vec.h"
#include <FL/Fl.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Check_Button.H>
#include <FL/Fl_Counter.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Hold_Browser.H>
#include <FL/Fl_Multi_Browser.H>
#include <FL/Fl_Output.H>
#include <FL/Fl_Select_Browser.H>
#include <FL/Fl_Valuator.H>
#include <FL/Fl_Widget.H>
#include <FL/fl_draw.H>
#include <atomic>
#include <fmt/core.h>
#include <fmt/format.h>
#include <memory>
#include <mutex>
#include <stdio.h>
#include <string>
#include <thread>
#include <valarray>
#include <vector>

template<>
struct fmt::formatter<vec_slice>
{
    constexpr auto parse(format_parse_context& ctx)
    {
        if (ctx.begin() != ctx.end() && *ctx.begin() != '}') throw format_error("invalid format");
        return ctx.begin();
    }
    template<typename FormatContext>
    auto format(vec_slice p, FormatContext& ctx) -> decltype(ctx.out())
    {
        if (p.size() == 0) return format_to(ctx.out(), "()");

        auto out = format_to(ctx.out(), "({: 4.2f}", p[0]);

        for (size_t i = 1; i < p.size(); ++i)
            out = format_to(out, ", {: 4.2f}", p[i]);

        return format_to(out, ")");
    }
};

template<>
struct fmt::formatter<std::vector<int>>
{
    constexpr auto parse(format_parse_context& ctx)
    {
        if (ctx.begin() != ctx.end() && *ctx.begin() != '}') throw format_error("invalid format");
        return ctx.begin();
    }
    template<typename FormatContext>
    auto format(std::vector<int> const& p, FormatContext& ctx) -> decltype(ctx.out())
    {
        if (p.size() == 0) return format_to(ctx.out(), "()");

        auto out = format_to(ctx.out(), "({}", p[0]);

        for (size_t i = 1; i < p.size(); ++i)
            out = format_to(out, ", {}", p[i]);

        return format_to(out, ")");
    }
};

std::atomic<bool> s_worker_exit = false;

std::atomic<double> s_err[200] = {0};

struct Card
{
    enum class Type
    {
        Creature,
        Direct,
        Land,
    };

    Type type;

    int value;

    void randomize()
    {
        type = (Type)(rand() % 3);
        if (type == Type::Land)
            value = 10;
        else
            value = 1 + rand() % 7;
    }
    void encode(vec_slice x) const
    {
        x.assign(0.0);
        x[(int)type] = value / 10.0;
    }
};

template<>
struct fmt::formatter<std::vector<Card>>
{
    constexpr auto parse(format_parse_context& ctx)
    {
        if (ctx.begin() != ctx.end() && *ctx.begin() != '}') throw format_error("invalid format");
        return ctx.begin();
    }
    template<typename FormatContext>
    auto format(std::vector<Card> const& p, FormatContext& ctx) -> decltype(ctx.out())
    {
        if (p.size() == 0) return format_to(ctx.out(), "()");

        auto out = format_to(ctx.out(), "({}.{}", (int)p[0].type, p[0].value);

        for (size_t i = 1; i < p.size(); ++i)
            out = format_to(out, ", {}.{}", (int)p[i].type, p[i].value);

        return format_to(out, ")");
    }
};

struct Player
{
    int health = 20;
    int mana = 0;
    int creature = 0;
    int def = 0;
    std::vector<Card> avail;

    static constexpr size_t encoded_size = 4;
    static constexpr size_t encoded_card_size = 3;

    void encode(vec_slice x) const
    {
        x[0] = health / 10.0;
        x[1] = mana / 10.0;
        x[2] = creature / 10.0;
        x[3] = def / 10.0;
    }
    void encode_cards(vec_slice x) const
    {
        for (size_t i = 0; i < avail.size(); ++i)
        {
            auto c = x.slice(i * encoded_card_size, encoded_card_size);
            avail[i].encode(c);
        }
    }
    void init()
    {
        *this = Player();
        avail.clear();
        avail.resize(5);
        for (auto&& c : avail)
            c.randomize();
    }

    int cards() const { return (int)avail.size(); }
};

struct Game
{
    Player p1;
    Player p2;
    bool player2_turn = false;
    int turn = 0;

    static constexpr size_t board_size = 2 + Player::encoded_size * 2;
    static constexpr size_t card_size = Player::encoded_card_size;

    struct Encoded
    {
        vec data;

        vec_slice board() { return data.slice(0, board_size); }
        vec_slice me_cards_in() { return data.slice(board_size, me_cards * card_size); }
        vec_slice you_cards_in() { return data.slice(board_size + me_cards * card_size, you_cards * card_size); }
        vec_slice me_card(int i) { return data.slice(board_size + i * card_size, card_size); }
        vec_slice you_card(int i) { return data.slice(board_size + (i + me_cards) * card_size, card_size); }

        int me_cards = 0;
        int you_cards = 0;

        int avail_actions() const { return me_cards + 1; }
    };

    Encoded encode() const
    {
        Encoded e;
        e.data.realloc_uninitialized(board_size + card_size * (p1.cards() + p2.cards()));
        e.data[0] = turn / 30.0;
        e.data[1] = player2_turn;

        auto [me, x2] = e.data.slice(2).split(Player::encoded_size);
        auto [you, x3] = x2.split(Player::encoded_size);

        if (player2_turn)
        {
            p2.encode(me);
            p1.encode(you);
            e.me_cards = p2.cards();
            e.you_cards = p1.cards();
            p2.encode_cards(e.me_cards_in());
            p1.encode_cards(e.you_cards_in());
        }
        else
        {
            p1.encode(me);
            p2.encode(you);
            e.me_cards = p1.cards();
            e.you_cards = p2.cards();
            p1.encode_cards(e.me_cards_in());
            p2.encode_cards(e.you_cards_in());
        }
        Encoded e2 = static_cast<Encoded&&>(e);
        return std::move(e);
    }

    void init()
    {
        p1.init();
        p2.init();
        player2_turn = false;
        turn = 0;
        p2.mana = 1;
        p1.mana = 1;
    }

    Player& cur_player() { return player2_turn ? p2 : p1; }

    void advance(int action)
    {
        auto& me = player2_turn ? p2 : p1;
        auto& you = player2_turn ? p1 : p2;

        if (action < 0 || action > me.cards() + 1)
        {
            action = 0;
        }

        if (action > 0)
        {
            auto& card = me.avail[action - 1];
            if (card.type == Card::Type::Land)
            {
                me.mana++;
            }
            else if (me.mana >= card.value)
            {
                if (card.type == Card::Type::Creature)
                {
                    me.creature = std::max(me.creature, card.value);
                }
                else if (card.type == Card::Type::Direct)
                {
                    you.health -= card.value;
                }
                else
                    std::terminate();
            }
            else
            {
                me.mana++;
            }

            // Discard
            me.avail.erase(me.avail.begin() + action - 1);
        }
        me.avail.emplace_back();
        me.avail.back().randomize();

        you.health -= std::max(0, me.creature - you.def);
        player2_turn = !player2_turn;
        ++turn;
    }

    std::string format() const
    {
        return fmt::format("Turn {}: P1{}: [hp: {}, atk: {}, def: {}, mana: {}, {}] P2{}: [hp: {}, atk: {}, def: "
                           "{}, mana: {}, {}]",
                           turn + 1,
                           player2_turn ? ' ' : '*',
                           p1.health,
                           p1.creature,
                           p1.def,
                           p1.mana,
                           p1.avail,
                           player2_turn ? '*' : ' ',
                           p2.health,
                           p2.creature,
                           p2.def,
                           p2.mana,
                           p2.avail);
    }

    std::vector<std::string> format_actions()
    {
        std::vector<std::string> actions{"Pass"};
        auto& p = cur_player();
        for (int i = 0; i < p.cards(); ++i)
        {
            if (p.avail[i].type == Card::Type::Land)
                actions.push_back("Play Land");
            else
            {
                const char* prefix = p.avail[i].value > p.mana ? "Play" : "Play";
                const char* suffix = p.avail[i].value > p.mana ? " as Land" : "";
                if (p.avail[i].type == Card::Type::Creature)
                {
                    actions.push_back(fmt::format("{} Creature {}{}", prefix, p.avail[i].value, suffix));
                }
                else if (p.avail[i].type == Card::Type::Direct)
                {
                    actions.push_back(fmt::format("{} Damage {}{}", prefix, p.avail[i].value, suffix));
                }
                else
                    std::terminate();
            }
        }
        return actions;
    }

    enum class Result
    {
        p1_win,
        p2_win,
        playing,
        timeout,
    };

    Result cur_result() const
    {
        if (p1.health <= 0) return Result::p2_win;
        if (p2.health <= 0) return Result::p1_win;
        if (turn > 30) return Result::timeout;
        return Result::playing;
    }
};

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

std::atomic<double> s_learn_rate = 0.0005;

struct Layer
{
    // double[In+1][Out]
    std::vector<vec> coefs;

    // double[In+1][Out]
    std::vector<vec> g1s;

    // double[In+1][Out]
    std::vector<vec> g2s;

    int num_deltas = 0;
    std::vector<vec> delta;

    struct Eval
    {
        // double[In]
        vec errs;

        // double[Out]
        vec out;
    };

    vec_slice calc(Eval& e, vec_slice input)
    {
        e.out.alloc_assign(coefs[input.size()]);

        size_t n = std::min(e.out.size(), input.size());
        e.out.slice(0, n).add(input.slice(0, n));

        vec_slice o(e.out);
        for (size_t i = 0; i < input.size(); ++i)
            o.fma(coefs[i], input[i]);

        return e.out;
    }

    void backprop_init()
    {
        num_deltas = 0;
        delta.resize(coefs.size());
        for (int i = 0; i < delta.size(); ++i)
        {
            delta[i].realloc(coefs[i].size(), 0.0);
        }
    }

    vec_slice backprop(Eval& e, vec_slice input, vec_slice grad)
    {
        e.errs.realloc_uninitialized(input.size());
        for (size_t j = 0; j < input.size(); ++j)
        {
            e.errs.slice()[j] = grad.dot(coefs[j]);
            delta[j].slice().fma(grad, input[j]);
        }

        size_t n = std::min(e.errs.size(), grad.size());
        e.errs.slice(0, n).add(grad.slice(0, n));

        delta.back().slice().add(grad);

        ++num_deltas;
        return e.errs;
    }

    void learn()
    {
        if (num_deltas == 0) return;

        double learn_rate = s_learn_rate;

        for (size_t i = 0; i < delta.size(); ++i)
        {
            delta[i].slice().mult(1.0 / num_deltas);
            g1s[i].slice().decay_average(delta[i], 0.1);
            g2s[i].slice().decay_variance(delta[i], 0.001);

            for (size_t j = 0; j < coefs[i].size(); ++j)
            {
                coefs[i][j] -= learn_rate * g1s[i][j] / sqrt(g2s[i][j] + 1e-8);
            }
        }
    }

    void normalize()
    {
        double learn_rate = s_learn_rate;
        auto l1_norm = 1e-11 * learn_rate;

        for (auto& r : coefs)
        {
            for (auto& e : r)
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
    }

    void randomize(int input, int output)
    {
        coefs.resize(input + 1);
        for (auto&& coef : coefs)
            coef.realloc(output, 0.0);
        g1s.resize(input + 1);
        for (auto&& coef : g1s)
            coef.realloc(output, 0.0);
        g2s.resize(input + 1);
        for (auto&& coef : g2s)
            coef.realloc(output, 0.0);
        for (auto& row : coefs)
            for (auto& v : row)
                v = (rand() * 2.0 / RAND_MAX - 1) / row.size();
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

    void learn() { l.learn(); }
    void normalize() { l.normalize(); }
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
    void learn()
    {
        for (auto& l : ls)
            l.learn();
    }
    void normalize()
    {
        for (auto& l : ls)
            l.normalize();
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
    void learn() { l.learn(); }
    void normalize() { l.normalize(); }
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
    void learn() { l.learn(); }
    void normalize() { l.normalize(); }
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
    void learn()
    {
        l2.learn();
        l.learn();
    }
    void normalize()
    {
        l2.normalize();
        l.normalize();
    }
};

struct Model
{
    ReLULayer b1;
    ReLULayers l;
    Layer p;

    PerCardInputModel card_in_model;
    PerYouCardInputModel you_card_in_model;
    PerCardOutputModel card_out_model;

    struct Eval
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
        double out_action(int i) { return all_out[i]; }
        double max_out() { return all_out.slice(1).max(all_out[0]); }
    };

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

    void calc(Eval& e, Game::Encoded& g, bool full)
    {
        b1.calc(e.b1, g.board());
        e.l_input.realloc_uninitialized(board_out_width + card_out_width);
        e.l_input.slice(0, board_out_width).assign(e.b1.out());
        auto l_input_cards = e.l_input.slice(board_out_width);
        l_input_cards.assign(0);
        e.cards_in.resize(g.me_cards);
        e.cards_out.resize(g.me_cards);
        e.you_cards_in.resize(g.you_cards);

        static constexpr size_t card_size = Game::card_size;
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

    void backprop_init()
    {
        b1.backprop_init();
        l.backprop_init();
        p.backprop_init();

        card_in_model.backprop_init();
        you_card_in_model.backprop_init();
        card_out_model.backprop_init();
    }

    void backprop(Eval& e, Game::Encoded& g, vec_slice grad, bool full)
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
    void learn()
    {
        b1.learn();
        l.learn();
        p.learn();

        card_in_model.learn();
        you_card_in_model.learn();
        card_out_model.learn();
    }

    void normalize()
    {
        b1.normalize();
        l.normalize();
        p.normalize();
        card_in_model.normalize();
        you_card_in_model.normalize();
        card_out_model.normalize();
    }
};

std::mutex s_mutex;
Model s_model;
bool s_updated = false;
std::atomic<int> s_trials = 0;

struct Turn
{
    Game::Encoded input;

    Model::Eval eval, eval_full;
    int chosen_action;
    vec error, error_full;
};

void ai_action(Turn& turn)
{
    auto& all_out = turn.eval.all_out;

    turn.chosen_action = 0;
    double win_pct = turn.eval.all_out[0];
    for (int i = 1; i < all_out.size(); ++i)
    {
        auto p = all_out[i];
        if (p > win_pct)
        {
            turn.chosen_action = i;
            win_pct = p;
        }
    }
}

void play_game(Game& g, Model& m, std::vector<Turn>& turns)
{
    g.init();
    turns.clear();
    turns.reserve(20);

    while (g.cur_result() == Game::Result::playing)
    {
        turns.emplace_back();
        auto& turn = turns.back();
        turn.input = g.encode();
        m.calc(turn.eval, turn.input, false);
        m.calc(turn.eval_full, turn.input, true);

        // choose action to take
        auto r = rand() * 1.0 / RAND_MAX;
        if (r < 0.3)
        {
            turn.chosen_action = static_cast<int>(r * turn.input.avail_actions() / 0.3);
        }
        else
        {
            ai_action(turn);
        }

        g.advance(turn.chosen_action);
    }
}

void replay_game(Model& m, std::vector<Turn>& turns)
{
    for (auto&& turn : turns)
    {
        m.calc(turn.eval, turn.input, false);
        m.calc(turn.eval_full, turn.input, true);
    }
}

void worker()
{
    int update_tick = 0;
    int i_err = 0;
    Model m;
    {
        std::lock_guard<std::mutex> lk(s_mutex);
        m = s_model;
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
        play_game(g, m, turns);

        m.backprop_init();

        // First, fill in the error values
        auto last_player_won =
            turns.size() % 2 == 1 ? (g.cur_result() == Game::Result::p1_win) : (g.cur_result() == Game::Result::p2_win);

        total_error = 0.0;

        auto& turn = turns.back();
        // full model
        auto predicted = turn.eval_full.out_action(turn.chosen_action);
        auto error = predicted - static_cast<double>(last_player_won);
        turn.error_full.realloc(turn.input.avail_actions(), 0.0);
        turn.error_full[turn.chosen_action] = error * 10;

        m.backprop(turn.eval_full, turn.input, turn.error_full, true);
        total_error += error * error;

        for (int i = (int)turns.size() - 2; i >= 0; --i)
        {
            auto& turn = turns[i];
            auto& next_turn = turns[i + 1];
            auto predicted = turn.eval_full.out_action(turn.chosen_action);
            auto error = predicted - (1.0 - std::max(0.0, std::min(1.0, next_turn.eval_full.max_out())));
            turn.error_full.realloc(turn.input.avail_actions(), 0.0);
            turn.error_full[turn.chosen_action] = error * 10;
            m.backprop(turn.eval_full, turn.input, turn.error_full, true);
            total_error += error * error;
        }

        for (auto&& turn : turns)
        {
            turn.error.realloc_uninitialized(turn.input.avail_actions());
            turn.error.slice().assign_sub(turn.eval.all_out, turn.eval_full.all_out);
            m.backprop(turn.eval, turn.input, turn.error, false);
            total_error += turn.error.slice().dot(turn.error);
        }

        m.learn();

        //// now learn

        s_err[i_err] = total_error;
        i_err = (i_err + 1ULL) % std::size(s_err);

        update_tick = (update_tick + 1) % 100;
        if (update_tick == 0)
        {
            m.normalize();
            std::lock_guard<std::mutex> lk(s_mutex);
            s_model = m;
            s_updated = true;
        }
        ++s_trials;
    }
}

struct Graph : Fl_Widget
{
    Graph(int x, int y, int w, int h, const char* label = 0) : Fl_Widget(x, y, w, h, label) { }
    virtual void draw() override
    {
        this->draw_box(FL_FLAT_BOX, FL_BACKGROUND_COLOR);
        this->draw_box(FL_FRAME, FL_BACKGROUND_COLOR);
        this->draw_label();

        if (valss.empty()) return;

        auto n = 0.0;
        auto m = 0.0001;
        for (auto&& vals : valss)
        {
            for (auto&& val : vals)
            {
                n = std::min(n, val);
                m = std::max(m, val);
            }
        }

        for (auto&& vals : valss)
        {
            if (vals.size() < 2) return;
            auto inc_w = this->w() * 1.0 / (vals.size() - 1);
            fl_begin_line();
            for (int i = 0; i < vals.size(); ++i)
            {
                fl_vertex(this->x() + i * inc_w, (int)(this->y() + (vals[i] - n) * this->h() / (m - n)));
            }
            fl_end_line();
        }
    }

    std::vector<std::vector<double>> valss;
};

struct MLStats_Group : Fl_Group
{
    MLStats_Group(int x, int y, int w, int h, const char* label = 0)
        : Fl_Group(x, y, w, h, label)
        , m_err(x + 40, y, w - 40, 20, "Error")
        , m_trials(x + 40, y + 22, w - 40, 20, "Trials")
        , m_learn_rate(x, y + 44, w, 20, "Learning Rate")
        , m_error_graph(x, y + 82, w, h - 82, "Error")
        , m_resize_box(x + w / 2, y + 82, 1, h - 82)
    {
        m_learn_rate.step(0.00001, 0.0001);
        m_learn_rate.bounds(0.0, 0.01);
        m_learn_rate.value(s_learn_rate);

        this->resizable(&m_resize_box);
        this->end();
    }

    void error_value(double e)
    {
        auto label = fmt::format("{:+f}", e);
        m_err.value(label.c_str());
    }

    void trials_value(int i)
    {
        auto label = fmt::format("{}", i);
        m_trials.value(label.c_str());
    }

    Fl_Output m_err;
    Fl_Output m_trials;
    Fl_Counter m_learn_rate;
    Graph m_error_graph;
    Fl_Box m_resize_box;
};

struct Turn_Viewer : Fl_Group
{
    Turn_Viewer(int x, int y, int w, int h, const char* label = 0)
        : Fl_Group(x, y, w, h, label)
        , m_actions(x, y, w / 2 - 5, h - 18, "Actions")
        , m_cur_turn(x + w / 2 + 5, y, w / 2 - 5, h - 18, "Current State")
    {
        m_actions.callback([](Fl_Widget* w, void* p) { ((Turn_Viewer*)w->parent())->on_player_action(); });
        this->end();
    }

    void on_player_action();

    Fl_Select_Browser m_actions;
    Fl_Browser m_cur_turn;
};

struct Game_Group : Fl_Group
{
    Game_Group(int x, int y, int w, int h, const char* label = 0)
        : Fl_Group(x, y, w, h, label)
        , m_turn_viewer(x, y + 22, w, (h - 22) / 2)
        , m_gamelog(x, y + 22 + (h - 22) / 2, w, (h - 22) / 2 - 17, "Gamelog")
        , m_new_game(x + w - 80, y, 80, 20, "New Game")
        , m_ai_plays_p1(x, y, 50, 20, "AI P1")
        , m_ai_plays_p2(x + 52, y, 50, 20, "AI P2")
        , m_resize_box(x + w / 2 - 20, y + 22 + (h - 22) / 2 + 1, 40, (h - 22) / 2 - 17)
    {
        m_ai_plays_p1.value(1);
        m_ai_plays_p1.callback([](Fl_Widget* w, void*) { ((Game_Group*)w->parent())->update_game(); });
        m_ai_plays_p2.value(1);
        m_ai_plays_p2.callback([](Fl_Widget* w, void*) { ((Game_Group*)w->parent())->update_game(); });
        m_new_game.callback([](Fl_Widget* w, void*) {
            auto& self = *(Game_Group*)w->parent();
            {
                std::lock_guard<std::mutex> lk(s_mutex);
                self.cur_model = s_model;
            }
            self.g.init();
            self.turns.clear();
            self.m_gamelog.clear();
            self.start_next_turn();
            self.update_game();
        });
        this->resizable(&m_resize_box);
        this->end();
    }

    void start_next_turn()
    {
        turns.emplace_back();
        auto& turn = turns.back();
        turn.input = g.encode();
        cur_model.calc(turn.eval, turn.input, false);
        cur_model.calc(turn.eval_full, turn.input, true);
        m_gamelog.add(("@." + g.format()).c_str());
        m_turn_viewer.m_actions.clear();
        auto actions = g.format_actions();
        for (size_t i = 0; i < actions.size(); ++i)
        {
            m_turn_viewer.m_actions.add(actions[i].c_str(), (void*)i);
        }
        m_turn_viewer.m_cur_turn.clear();
        switch (g.cur_result())
        {
            case Game::Result::p1_win: m_turn_viewer.m_cur_turn.add("@.Player 1 won"); break;
            case Game::Result::p2_win: m_turn_viewer.m_cur_turn.add("@.Player 2 won"); break;
            case Game::Result::timeout: m_turn_viewer.m_cur_turn.add("@.Timeout"); break;
            default: break;
        }
        if (g.player2_turn)
            m_turn_viewer.m_cur_turn.add(fmt::format("@.Turn {}: Player 2's turn", g.turn + 1).c_str());
        else
            m_turn_viewer.m_cur_turn.add(fmt::format("@.Turn {}: Player 1's turn", g.turn + 1).c_str());
        m_turn_viewer.m_cur_turn.add(fmt::format("@.P1 Health: {}", g.p1.health).c_str());
        m_turn_viewer.m_cur_turn.add(fmt::format("@.P1 Mana: {}", g.p1.mana).c_str());
        m_turn_viewer.m_cur_turn.add(fmt::format("@.P1 Creature: {}", g.p1.creature).c_str());
        m_turn_viewer.m_cur_turn.add("");
        m_turn_viewer.m_cur_turn.add(fmt::format("@.P2 Health: {}", g.p2.health).c_str());
        m_turn_viewer.m_cur_turn.add(fmt::format("@.P2 Mana: {}", g.p2.mana).c_str());
        m_turn_viewer.m_cur_turn.add(fmt::format("@.P2 Creature: {}", g.p2.creature).c_str());
    }

    void on_player_action()
    {
        if (turns.empty() || g.cur_result() != Game::Result::playing) return;
        if (m_turn_viewer.m_actions.value() == 0) return;

        auto& turn = turns.back();
        turn.chosen_action = m_turn_viewer.m_actions.value() - 1;
        m_turn_viewer.m_actions.deselect();
        advance_turn();
        update_game();
    }

    void advance_turn()
    {
        auto& turn = turns.back();
        m_gamelog.add(fmt::format("@.Turn {}:   AI: {} FAI: {}  Action: {}",
                                  turns.size(),
                                  turn.eval.all_out.slice(),
                                  turn.eval_full.all_out.slice(),
                                  g.format_actions()[turn.chosen_action])
                          .c_str());
        g.advance(turn.chosen_action);
        start_next_turn();
    }

    void update_game()
    {
        if (turns.empty()) return;
        while (g.cur_result() == Game::Result::playing)
        {
            if (turns.size() % 2 == 1 && !m_ai_plays_p1.value()) break;
            if (turns.size() % 2 == 0 && !m_ai_plays_p2.value()) break;

            auto& turn = turns.back();

            // choose action to take
            ai_action(turn);
            advance_turn();
        }
    }

    Turn_Viewer m_turn_viewer;
    Fl_Browser m_gamelog;
    Fl_Button m_new_game;
    Fl_Check_Button m_ai_plays_p1;
    Fl_Check_Button m_ai_plays_p2;
    Fl_Box m_resize_box;

    Game g;
    std::vector<Turn> turns;
    Model cur_model;
    Model cur_model_full;
};

void Turn_Viewer::on_player_action() { ((Game_Group*)parent())->on_player_action(); }

MLStats_Group* s_mlgroup;
Game_Group* s_gamegroup;

int main(int argc, char* argv[])
{
    srand((unsigned int)time(NULL));
    s_model.randomize(Game::board_size, Game::card_size);

    auto win = std::make_unique<Fl_Double_Window>(490, 400, "MLCard");
    win->begin();
    s_mlgroup = new MLStats_Group(10, 10, win->w() - 20, win->h() - 20);
    s_mlgroup->m_learn_rate.value(s_learn_rate);
    s_mlgroup->m_learn_rate.callback([](Fl_Widget* w, void* data) {
        auto self = (Fl_Counter*)w;
        s_learn_rate = self->value();
    });
    s_mlgroup->m_error_graph.valss.resize(1);
    Fl::add_idle([](void* v) {
        s_mlgroup->error_value(s_err[0]);
        s_mlgroup->trials_value(s_trials.load());

        s_mlgroup->m_error_graph.valss[0].clear();
        std::copy(std::begin(s_err), std::end(s_err), std::back_inserter(s_mlgroup->m_error_graph.valss[0]));
        s_mlgroup->m_error_graph.damage(FL_DAMAGE_ALL);
        s_mlgroup->m_error_graph.redraw();
    });
    win->end();
    win->resizable(new Fl_Box(10, 10, win->w() - 20, win->h() - 20));
    win->show(argc, argv);

    auto win2 = std::make_unique<Fl_Double_Window>(490, 400, "MLCard Game");
    win2->begin();
    s_gamegroup = new Game_Group(10, 10, win2->w() - 20, win2->h() - 20);
    win2->end();
    win2->resizable(new Fl_Box(10, 10, win2->w() - 20, win2->h() - 20));
    win2->show();

    std::thread th(worker);
    auto rc = Fl::run();
    s_worker_exit = true;
    th.join();
    return rc;
}
