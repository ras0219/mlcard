#include "ai_play.h"
#include "game.h"
#include "model.h"
#include "rjwriter.h"
#include <rapidjson/writer.h>
#include <time.h>

#if defined(_MSC_VER)
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

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

API void init() { srand((unsigned int)time(NULL)); }

API const char* serialize_game(APIGame* g)
{
    g->s.Clear();
    RJWriter w(g->s);
    g->g.serialize(w);
    return g->s.GetString();
}
API bool deserialize_game(APIGame* g, const char* text)
{
    try
    {
        g->g.deserialize(text);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

API const char* game_help_html(const char* page) { return Game::help_html(page); }

API void take_action(APIGame* g, int action) { g->g.advance(action); }

API int ai_take_action(APIGame* g, APIModel* m)
{
    auto e = m->m->make_eval();
    auto enc = g->g.encode();
    m->m->calc(*e, enc, false);
    auto a = e->best_action();
    g->g.advance(a);
    return a;
}
