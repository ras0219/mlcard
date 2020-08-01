#include "game.h"
#include <cstdlib>
#include <fmt/format.h>

struct card_encode_slice : vec_slice
{
    card_encode_slice(vec_slice v) : vec_slice(v) { }

    vec_slice artifact_slice() { return slice((int)Card::Type::Count); }
};

void Card::randomize()
{
    type = (Type)(rand() % (int)Type::Count);
    if (type == Type::Land)
        value = 10;
    else if (type == Type::Artifact)
        artifact = (ArtifactType)(rand() % (int)ArtifactType::Count);
    else
        value = 1 + rand() % 7;
}
void Card::encode(vec_slice x) const
{
    card_encode_slice c(x);
    x.assign(0.0);
    if (type != Type::Artifact)
    {
        x[(int)type] = value / 10.0;
    }
    else
    {
        x[(int)Type::Artifact] = 1;
        c.artifact_slice()[(int)artifact] = 1;
    }
}

void Player::encode(vec_slice x) const
{
    x.assign(0.0);
    x[0] = health / 10.0;
    x[1] = mana / 10.0;
    x[2] = creature / 10.0;
    x[3] = avail.size() / 10.0;
    if (artifact != ArtifactType::Count)
    {
        x.slice(4)[(int)artifact] = 1;
    }
}
void Player::encode_cards(vec_slice x) const
{
    for (size_t i = 0; i < avail.size(); ++i)
    {
        auto c = x.slice(i * Card::encoded_size, Card::encoded_size);
        avail[i].encode(c);
    }
}
void Player::init(bool p1)
{
    *this = Player();
    avail.clear();
    avail.resize(p1 ? 5 : 7);
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

const char* Game::help_html(std::string_view pg)
{
    return "<h1>How to Play</h1>"
           "<h2>Setup and Objective</h2>"
           "<p>Player 1 initially has 5 cards in hand. Player 2 initially has 7 cards in hand. Both players start "
           "with 20 health and 1 mana.</p>"
           "<p>Player 1 wins once Player 2's health reaches 0 or less. Player 2 wins once Player 1's health reaches 0 "
           "or less. If 32 turns have elapsed and neither player has won, it is a timeout and the game should be "
           "restarted.</p>"
           "<h2>Per Turn</h2>"
           "<p>Each turn, the current player plays up to one card. If the card is a land, the player's mana "
           "increases by one. If the card is not a land and costs less than or equal to the player's current mana, "
           "the card's effect takes place. If the card is not a land and costs more than the player's current "
           "mana, the player's mana increases by one (called 'Play as Land').</p>"
           "<p>Finally, the opposing player loses health equal to the current player's creature value, the current "
           "player draws a card, and the opposing player takes their turn.<p>"
           "<h2>Card Effects</h2>"
           "<ul>"
           "<li>Damage X: Costs X. Reduce the opponent's health by X.</li>"
           "<li>Heal X: Costs X. Increase the current player's health by X.</li>"
           "<li>Creature X: Costs X. Set the current player's creature value to X if X is larger than the current "
           "player's creature value.</li>"
           "<li>Draw X: Costs X. Draws 3 cards.</li>"
           "<li>Artifact X: Costs 0. Replaces the player's current artifact."
           "<ul>"
           "<li>Double Mana: Player can play cards at half cost.</li>"
           "<li>Half Creature Damage: Player takes half damage rounded down from creatures.</li>"
           "<li>Direct Damage Immunity: Player takes no damage from non-creature sources.</li>"
           "<li>Heals cause Damage: Heal X additionally acts as Damage X.</li>"
           "<li>Lands cause Damage: Lands deal damage when played equal to the new mana amount.</li>"
           "</ul>"
           "</li>"
           "</ul>";
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
            if (you.artifact != ArtifactType::DirectImmune && me.artifact == ArtifactType::LandCauseDamage)
            {
                you.health -= me.mana;
            }
        }
        else if (card.type == Card::Type::Artifact)
        {
            me.artifact = card.artifact;
        }
        else if (me.mana >= card.value || (me.artifact == ArtifactType::DoubleMana && me.mana * 2 >= card.value))
        {
            if (card.type == Card::Type::Creature)
            {
                me.creature = std::max(me.creature, card.value);
            }
            else if (card.type == Card::Type::Direct)
            {
                if (you.artifact != ArtifactType::DirectImmune)
                {
                    you.health -= card.value;
                }
            }
            else if (card.type == Card::Type::Draw3)
            {
                me.avail.emplace_back();
                me.avail.back().randomize();
                me.avail.emplace_back();
                me.avail.back().randomize();
            }
            else if (card.type == Card::Type::Heal)
            {
                me.health += card.value;
                if (you.artifact != ArtifactType::DirectImmune && me.artifact == ArtifactType::HealCauseDamage)
                {
                    you.health -= card.value;
                }
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

    if (you.artifact == ArtifactType::CreatureImmune)
    {
        you.health -= me.creature / 2;
    }
    else
    {
        you.health -= me.creature;
    }
    player2_turn = !player2_turn;
    ++turn;
}

std::string Game::format() const
{
    return fmt::format("Turn {}: P1{}: [hp: {}, atk: {}, art: {}, mana: {}, {}] P2{}: [hp: {}, atk: {}, art: "
                       "{}, mana: {}, {}]",
                       turn + 1,
                       player2_turn ? ' ' : '*',
                       p1.health,
                       p1.creature,
                       p1.artifact,
                       p1.mana,
                       p1.avail,
                       player2_turn ? '*' : ' ',
                       p2.health,
                       p2.creature,
                       p2.artifact,
                       p2.mana,
                       p2.avail);
}
const char* artifact_name(ArtifactType t)
{
    switch (t)
    {
        case ArtifactType::CreatureImmune: return "Half Creature Damage";
        case ArtifactType::DirectImmune: return "Direct Damage Immunity";
        case ArtifactType::HealCauseDamage: return "Heals cause Damage";
        case ArtifactType::LandCauseDamage: return "Lands cause Damage";
        case ArtifactType::DoubleMana: return "Double Mana";
        default: std::terminate();
    }
}

std::vector<std::string> Game::format_public_lines() const
{
    std::vector<std::string> ret;

    switch (cur_result())
    {
        case Game::Result::p1_win: ret.push_back("Player 1 won"); break;
        case Game::Result::p2_win: ret.push_back("Player 2 won"); break;
        case Game::Result::timeout: ret.push_back("Timeout"); break;
        default: break;
    }
    if (player2_turn)
        ret.push_back(fmt::format("Turn {}: Player 2's turn", turn + 1));
    else
        ret.push_back(fmt::format("Turn {}: Player 1's turn", turn + 1));

    auto format_player = [&ret](char c, const Player& p) {
        ret.push_back(fmt::format("P{} Health: {}    Hand: {}    Mana: {}", c, p.health, p.avail.size(), p.mana));
        if (p.artifact != ArtifactType::Count)
        {
            ret.push_back(fmt::format("P{} Artifact: {}", c, artifact_name(p.artifact)));
        }
        if (p.creature != 0)
        {
            ret.push_back(fmt::format("P{} Creature: {}", c, p.creature));
        }
    };

    format_player('1', p1);
    ret.push_back("");
    format_player('2', p2);

    return ret;
}

std::vector<std::string> Game::format_actions()
{
    std::vector<std::string> actions{"Pass"};
    auto& p = cur_player();
    for (int i = 0; i < p.cards(); ++i)
    {
        if (p.avail[i].type == Card::Type::Land)
            actions.push_back("Play Land");
        else if (p.avail[i].type == Card::Type::Artifact)
        {
            actions.push_back(fmt::format("Play Artifact: {}", artifact_name(p.avail[i].artifact)));
        }
        else
        {
            const char* prefix = "Play";
            auto mana = p.artifact != ArtifactType::DoubleMana ? p.mana : p.mana * 2;
            const char* suffix = p.avail[i].value > mana ? " as Land" : "";
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
            else if (p.avail[i].type == Card::Type::Draw3)
            {
                actions.push_back(fmt::format("{} Draw {}{}", prefix, p.avail[i].value, suffix));
            }
            else
                std::terminate();
        }
    }
    return actions;
}
