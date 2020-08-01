#pragma once

#include "vec.h"
#include <fmt/format.h>
#include <string>
#include <vector>

enum class ArtifactType
{
    DirectImmune,
    CreatureImmune,
    DoubleMana,
    HealCauseDamage,
    LandCauseDamage,
    Count,
};

struct Card
{
    enum class Type
    {
        Creature,
        Direct,
        Heal,
        Land,
        Draw3,
        Artifact,
        Count,
    };

    Type type;

    union
    {
        int value;
        ArtifactType artifact;
    };

    void randomize();
    void encode(vec_slice x) const;

    static constexpr size_t encoded_size = (size_t)Type::Count + (size_t)ArtifactType::Count;
};

struct Player
{
    int health = 20;
    int mana = 1;
    int creature = 0;
    ArtifactType artifact = ArtifactType::Count;
    std::vector<Card> avail;

    static constexpr size_t encoded_size = 4 + (size_t)ArtifactType::Count;

    void encode(vec_slice x) const;
    void encode_cards(vec_slice x) const;
    void init(bool p1);

    int cards() const { return (int)avail.size(); }
};

struct Encoded
{
    static constexpr size_t board_size = 2 + Player::encoded_size * 2;
    static constexpr size_t card_size = Card::encoded_size;

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

struct Game
{
    Player p1;
    Player p2;
    bool player2_turn = false;
    int turn = 0;
    Encoded encode() const;

    void init();

    Player& cur_player() { return player2_turn ? p2 : p1; }

    void advance(int action);

    std::string format() const;
    std::vector<std::string> format_public_lines() const;
    static const char* help_html(std::string_view pg);
    std::vector<std::string> format_actions();

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
