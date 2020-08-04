#include "game.h"
#include "kv_range.h"
#include "rjwriter.h"
#include <cstdlib>
#include <fmt/format.h>
#include <rapidjson/document.h>

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
        x[(int)type] = value / 10.0f;
    }
    else
    {
        x[(int)Type::Artifact] = 1;
        c.artifact_slice()[(int)artifact] = 1;
    }
}
static const char* card_encoded_desc(int i)
{
    if (i < (int)Card::Type::Count) return card_name((Card::Type)i);
    return artifact_name((ArtifactType)(i - (int)Card::Type::Count));
}

void Player::encode(vec_slice x) const
{
    x.assign(0.0f);
    x[0] = health / 20.0f;
    x[1] = land / 10.0f;
    x[2] = creature / 10.0f;
    x[3] = avail.size() / 14.0f;
    if (artifact != ArtifactType::Count)
    {
        x.slice(4)[(int)artifact] = 1;
    }
}
void Player::encode_cards(vec_slice x) const
{
    for (auto&& [k, v] : kv_range(avail))
    {
        auto c = x.slice(k * Card::encoded_size, Card::encoded_size);
        v.encode(c);
    }
}
void Player::init(bool p1)
{
    *this = Player();
    avail.clear();
    avail.resize(p1 ? 6 : 7);
    for (auto&& c : avail)
        c.randomize();
}

std::vector<std::string> Game::input_descs()
{
    std::vector<std::string> ret;
    ret.push_back("turn");
    ret.push_back("player2_turn");
    ret.push_back("me_health");
    ret.push_back("me_mana");
    ret.push_back("me_creature");
    ret.push_back("me_handsize");
    for (int i = 0; i < (int)ArtifactType::Count; ++i)
        ret.push_back(fmt::format("me_have_{}", artifact_name((ArtifactType)i)));
    ret.push_back("you_health");
    ret.push_back("you_mana");
    ret.push_back("you_creature");
    ret.push_back("you_handsize");
    for (int i = 0; i < (int)ArtifactType::Count; ++i)
        ret.push_back(fmt::format("you_have_{}", artifact_name((ArtifactType)i)));
    auto& me = player2_turn ? p2 : p1;
    auto& you = player2_turn ? p1 : p2;
    for (int i = 0; i < me.avail.size(); ++i)
        for (int j = 0; j < Card::encoded_size; ++j)
            ret.push_back(fmt::format("me_card{}_{}", i, card_encoded_desc(j)));

    for (int i = 0; i < you.avail.size(); ++i)
        for (int j = 0; j < Card::encoded_size; ++j)
            ret.push_back(fmt::format("you_card{}_{}", i, card_encoded_desc(j)));

    return ret;
}

Encoded Game::encode() const
{
    Encoded e;
    e.data.realloc_uninitialized(Encoded::board_size + Encoded::card_size * (p1.cards() + p2.cards()));
    e.data[0] = turn / 30.0f;
    e.data[1] = player2_turn;
    e.data[2] = mana / 10.0f;
    e.data[3] = played_land;

    auto [me, x2] = e.data.slice(4).split(Player::encoded_size);
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
    mana = cur_player().land;
    played_land = false;
}

const char* Game::help_html(std::string_view pg)
{
    return "<h1>How to Play</h1>"
           "<h2>Setup and Objective</h2>"
           "<p>Player 1 initially has 6 cards in hand. Player 2 initially has 7 cards in hand. Both players start "
           "with 20 health and 1 land.</p>"
           "<p>Player 1 wins once Player 2's health reaches 0 or less. Player 2 wins once Player 1's health reaches 0 "
           "or less. If 32 turns have elapsed and neither player has won, it is a timeout and the game should be "
           "restarted.</p>"
           "<h2>Per Turn</h2>"
           "<p>Each turn, the current player starts with mana equal to their lands. The current player may then play "
           "any number of cards. If the card is a land, the player's lands "
           "increase by one. Only one land may be played each turn. If the card is not a land and costs less than or "
           "equal to the player's current mana, the card's effect takes place, and the player's mana is decreased by "
           "the cost. If the card costs more than the player's current mana, that card is treated like a Land (called "
           "'Play as Land') and subject to the same one-per-turn limit.</p>"
           "<p>Once the current player has finished playign cards, the opposing player loses health equal to the "
           "current player's creature value, the current player draws a card, and the opposing player takes their "
           "turn.<p>"
           "<h2>Card Effects</h2>"
           "<ul>"
           "<li>Damage X: Costs X. Reduce the opponent's health by X.</li>"
           "<li>Heal X: Costs X. Increase the current player's health by X.</li>"
           "<li>Creature X: Costs X. Set the current player's creature value to X if X is larger than the current "
           "player's creature value.</li>"
           "<li>Draw X: Costs X. Draws 3 cards.</li>"
           "<li>Artifact X: Costs 0. Replaces the player's current artifact."
           "<ul>"
           "<li>Double Mana: Player starts the turn with mana equal to double their lands.</li>"
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

    bool passed = action == 0;
    if (action > 0)
    {
        auto& card = me.avail[action - 1];
        if (card.type == Card::Type::Land)
        {
            if (played_land)
            {
                passed = true;
            }
            else
            {
                played_land = true;
                me.land++;
                if (you.artifact != ArtifactType::DirectImmune && me.artifact == ArtifactType::LandCauseDamage)
                {
                    you.health -= me.land;
                }
            }
        }
        else if (card.type == Card::Type::Artifact)
        {
            me.artifact = card.artifact;
        }
        else
        {
            if (mana >= card.value)
            {
                mana -= card.value;
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
                if (played_land)
                {
                    passed = true;
                }
                else
                {
                    played_land = true;
                    me.land++;
                    if (you.artifact != ArtifactType::DirectImmune && me.artifact == ArtifactType::LandCauseDamage)
                    {
                        you.health -= me.land;
                    }
                }
            }
        }

        // Discard
        if (!passed)
        {
            me.avail.erase(me.avail.begin() + action - 1);
        }
    }

    if (passed)
    {
        me.avail.emplace_back();
        me.avail.back().randomize();

        ++turn;

        if (you.artifact == ArtifactType::CreatureImmune)
        {
            you.health -= me.creature / 2;
        }
        else
        {
            you.health -= me.creature;
        }
        player2_turn = !player2_turn;
        mana = cur_player().land;
        if (cur_player().artifact == ArtifactType::DoubleMana) mana *= 2;
        played_land = false;
    }
}

std::string Game::format() const
{
    return fmt::format("Turn {}: {}: P1{}: [hp: {}, atk: {}, art: {}, land: {}, {}] P2{}: [hp: {}, atk: {}, art: "
                       "{}, land: {}, {}]",
                       turn + 1,
                       mana,
                       player2_turn ? ' ' : '*',
                       p1.health,
                       p1.creature,
                       p1.artifact,
                       p1.land,
                       p1.avail,
                       player2_turn ? '*' : ' ',
                       p2.health,
                       p2.creature,
                       p2.artifact,
                       p2.land,
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
const char* card_name(Card::Type t)
{
    switch (t)
    {
        case Card::Type::Creature: return "Creature";
        case Card::Type::Direct: return "Direct";
        case Card::Type::Heal: return "Heal";
        case Card::Type::Land: return "Land";
        case Card::Type::Draw3: return "Draw3";
        case Card::Type::Artifact: return "Artifact";
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
        ret.push_back(fmt::format("Turn {}: Player 2's turn: {} mana", turn + 1, mana));
    else
        ret.push_back(fmt::format("Turn {}: Player 1's turn: {} mana", turn + 1, mana));

    auto format_player = [&ret](char c, const Player& p) {
        ret.push_back(fmt::format("P{} Health: {}    Hand: {}    Land: {}", c, p.health, p.avail.size(), p.land));
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

void Game::serialize(RJWriter& w)
{
    w.StartObject();
    w.Key("current_player");
    w.String(player2_turn ? "player2" : "player1");
    w.Key("actions");
    w.StartArray();
    for (auto&& a : format_actions())
        w.String(a.c_str());
    w.EndArray();
    auto srlz_player = [&w](Player& p) {
        w.StartObject();
        if (p.artifact != ArtifactType::Count)
        {
            w.Key("artifact");
            w.String(artifact_name(p.artifact));
            w.Key("artifact_id");
            w.Int((int)p.artifact);
        }
        if (p.creature > 0)
        {
            w.Key("creature");
            w.Int(p.creature);
        }
        w.Key("health");
        w.Int(p.health);
        w.Key("land");
        w.Int(p.land);
        w.Key("cards");
        w.StartArray();
        for (auto&& c : p.avail)
        {
            w.StartObject();
            w.Key("id");
            w.Int((int)c.type);
            w.Key("type");
            w.String(card_name(c.type));
            if (c.type == Card::Type::Artifact)
            {
                w.Key("artifact");
                w.String(artifact_name(c.artifact));
                w.Key("artifact_id");
                w.Int((int)c.artifact);
            }
            else
            {
                w.Key("value");
                w.Int(c.value);
            }
            w.EndObject();
        }
        w.EndArray();
        w.EndObject();
    };
    w.Key("player1");
    srlz_player(p1);
    w.Key("player2");
    srlz_player(p2);
    w.Key("turn");
    w.Int(turn);
    w.Key("mana");
    w.Int(mana);
    w.Key("played_land");
    w.Bool(played_land);
    w.EndObject();
}

using rapidjson::Value;

static const Value& find_or_throw(const Value& doc, const char* key)
{
    auto it = doc.FindMember(key);
    if (it == doc.MemberEnd()) throw std::runtime_error(fmt::format("could not find .{}", key));
    return it->value;
}

void Game::deserialize(const std::string& s)
{
    rapidjson::Document doc;
    doc.Parse(s.c_str(), s.size());
    player2_turn = find_or_throw(doc, "current_player").GetString() == std::string_view("player2");
    played_land = find_or_throw(doc, "played_land").GetBool();
    mana = find_or_throw(doc, "mana").GetInt();
    turn = find_or_throw(doc, "turn").GetInt();
    auto desrlz_player = [](const Value& v, Player& p) {
        p.land = find_or_throw(v, "land").GetInt();
        p.health = find_or_throw(v, "health").GetInt();
        auto it_creature = v.FindMember("creature");
        if (it_creature == v.MemberEnd())
            p.creature = 0;
        else
            p.creature = it_creature->value.GetInt();

        auto it_artifact = v.FindMember("artifact_id");
        if (it_artifact == v.MemberEnd())
            p.artifact = ArtifactType::Count;
        else
            p.artifact = (ArtifactType)it_artifact->value.GetInt();

        auto desrlz_card = [](const Value& v, Card& c) {
            c.type = (Card::Type)find_or_throw(v, "type").GetInt();
            auto it_artifact = v.FindMember("artifact_id");
            if (it_artifact == v.MemberEnd())
                c.value = find_or_throw(v, "value").GetInt();
            else
                c.artifact = (ArtifactType)it_artifact->value.GetInt();
        };
        p.avail.clear();
        for (auto&& c : find_or_throw(v, "cards").GetArray())
        {
            p.avail.emplace_back();
            desrlz_card(c, p.avail.back());
        }
    };
    desrlz_player(find_or_throw(doc, "player1"), p1);
    desrlz_player(find_or_throw(doc, "player2"), p2);
}

std::vector<std::string> Game::format_actions()
{
    std::vector<std::string> actions{"Pass"};
    auto& p = cur_player();
    for (auto&& card : p.avail)
    {
        if (card.type == Card::Type::Land)
        {
            if (played_land)
                actions.push_back("Pass - Play Land");
            else
                actions.push_back("Play Land");
        }
        else if (card.type == Card::Type::Artifact)
        {
            actions.push_back(fmt::format("Play Artifact: {}", artifact_name(card.artifact)));
        }
        else
        {
            const char* prefix = "Play";
            if (card.value > mana && played_land) prefix = "Pass -";
            const char* suffix = card.value > mana ? " as Land" : "";
            if (card.type == Card::Type::Creature)
            {
                actions.push_back(fmt::format("{} Creature {}{}", prefix, card.value, suffix));
            }
            else if (card.type == Card::Type::Direct)
            {
                actions.push_back(fmt::format("{} Damage {}{}", prefix, card.value, suffix));
            }
            else if (card.type == Card::Type::Heal)
            {
                actions.push_back(fmt::format("{} Heal {}{}", prefix, card.value, suffix));
            }
            else if (card.type == Card::Type::Draw3)
            {
                actions.push_back(fmt::format("{} Draw {}{}", prefix, card.value, suffix));
            }
            else
                std::terminate();
        }
    }
    return actions;
}
