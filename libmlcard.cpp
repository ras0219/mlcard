#include "ai_play.h"
#include "game.h"
#include "model.h"
#include "rjwriter.h"
#include <rapidjson/writer.h>

#define API extern "C" __declspec(dllexport)

struct APIGame
{
    Game g;
    rapidjson::StringBuffer s;
};

struct APIModel
{
    std::shared_ptr<IModel> m;
};

API APIGame* alloc_game()
{
    auto r = std::make_unique<APIGame>();
    r->g.init();
    return r.release();
}
API APIModel* alloc_model(const char* json)
{
    auto r = std::make_unique<APIModel>();
    r->m = load_model(json);
    return r.release();
}
API void free_game(APIGame* g) { std::unique_ptr<APIGame> u(g); }
API void free_model(APIModel* m) { std::unique_ptr<APIModel> u(m); }

API const char* serialize_game(APIGame* g)
{
    rapidjson::Writer w(g->s);
    w.StartObject();
    w.Key("current_player");
    w.String(g->g.player2_turn ? "player2" : "player1");
    w.Key("actions");
    w.StartArray();
    for (auto&& a : g->g.format_actions())
        w.String(a.c_str());
    w.EndArray();
    auto srlz_player = [&w, g](Player& p) {
        w.StartObject();
        if (p.artifact != ArtifactType::Count)
        {
            w.Key("artifact");
            w.String(artifact_name(p.artifact));
        }
        if (p.creature > 0)
        {
            w.Key("creature");
            w.Int(p.creature);
        }
        w.Key("health");
        w.Int(p.health);
        w.Key("mana");
        w.Int(p.mana);
        w.Key("cards");
        w.StartArray();
        for (auto&& c : p.avail)
        {
            w.StartObject();
            w.Key("type");
            w.String(card_name(c.type));
            if (c.type == Card::Type::Artifact)
            {
                w.Key("artifact");
                w.String(artifact_name(c.artifact));
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
    srlz_player(g->g.p1);
    w.Key("player2");
    srlz_player(g->g.p2);
    w.Key("turn");
    w.Int(g->g.turn);
    w.EndObject();
    return g->s.GetString();
}

API void take_action(APIGame* g, int action) { g->g.advance(action); }

API void ai_take_action(APIGame* g, APIModel* m)
{
    auto e = m->m->make_eval();
    auto enc = g->g.encode();
    m->m->calc(*e, enc, false);
    g->g.advance(e->best_action());
}
