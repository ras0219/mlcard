#include "game.h"
#include <cstdlib>
#include <fmt/format.h>

void Card::randomize()
{
    type = (Type)(rand() % 4);
    if (type == Type::Land)
        value = 10;
    else
        value = 1 + rand() % 7;
}
void Card::encode(vec_slice x) const
{
    x.assign(0.0);
    x[(int)type] = value / 10.0;
}

void Player::encode(vec_slice x) const
{
    x[0] = health / 10.0;
    x[1] = mana / 10.0;
    x[2] = creature / 10.0;
    x[3] = def / 10.0;
}
void Player::encode_cards(vec_slice x) const
{
    for (size_t i = 0; i < avail.size(); ++i)
    {
        auto c = x.slice(i * encoded_card_size, encoded_card_size);
        avail[i].encode(c);
    }
}
void Player::init(bool p1)
{
    *this = Player();
    avail.clear();
    avail.resize(p1 ? 3 : 5);
    for (auto&& c : avail)
        c.randomize();
}

Encoded Game::encode() const
{
    Encoded e;
    e.data.realloc_uninitialized(Encoded::board_size + Encoded::card_size * (p1.cards() + p2.cards()));
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

void Game::init()
{
    p1.init(true);
    p2.init(false);
    player2_turn = false;
    turn = 0;
}

void Game::advance(int action)
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
            else if (card.type == Card::Type::Heal)
            {
                me.health += card.value;
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

std::string Game::format() const
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

std::vector<std::string> Game::format_actions()
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
            else if (p.avail[i].type == Card::Type::Heal)
            {
                actions.push_back(fmt::format("{} Heal {}{}", prefix, p.avail[i].value, suffix));
            }
            else
                std::terminate();
        }
    }
    return actions;
}
