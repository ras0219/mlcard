#define NOMINMAX
//#define VEC_ENABLE_CHECKS

#include "ai_play.h"
#include "game.h"
#include "graph.h"
#include "kv_range.h"
#include "lock_guarded.h"
#include "margins.h"
#include "model.h"
#include "modeldims.h"
#include "rjwriter.h"
#include "thunks.h"
#include "vec.h"
#include "worker.h"
#include <FL/Fl.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Check_Button.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Counter.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_File_Chooser.H>
#include <FL/Fl_Help_View.H>
#include <FL/Fl_Hold_Browser.H>
#include <FL/Fl_Menu_Bar.H>
#include <FL/Fl_Multi_Browser.H>
#include <FL/Fl_Output.H>
#include <FL/Fl_Pack.H>
#include <FL/Fl_Return_Button.H>
#include <FL/Fl_Scroll.H>
#include <FL/Fl_Select_Browser.H>
#include <FL/Fl_Valuator.H>
#include <FL/Fl_Widget.H>
#include <FL/fl_draw.H>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <execution>
#include <fmt/format.h>
#include <fstream>
#include <memory>
#include <mutex>
#include <rapidjson/writer.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <thread>
#include <time.h>
#include <valarray>
#include <vector>
#if defined(_WIN32)
#include <shobjidl_core.h>
#include <winrt/base.h>
#pragma comment(lib, "windowsapp")
#endif

using namespace rapidjson;

struct ModelsList
{
    std::vector<std::shared_ptr<IModel>> models;

    void push_back(std::shared_ptr<IModel> m);
    void replace(int i, std::shared_ptr<IModel> m);
    void erase(int i);
    void clear();
} s_models_list;

std::vector<std::unique_ptr<Worker>> s_workers;
struct Game_Group* s_gamegroup;
struct Tournament_Group* s_tgroup;
struct Model_Explorer* s_mexp;
struct Manager_Window* s_mwin;

static long long get_nanos()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (long long)ts.tv_sec * 1000000000L + ts.tv_nsec;
}

struct MLStats_Group : Fl_Group
{
    MLStats_Group(int x, int y, int w, int h, const char* label = 0)
        : Fl_Group(x, y, w, h, label)
        , m_err(x + 40, y, w - 350, 20, "Error")
        , m_trials(x + 40, y + 22, w - 40, 20, "Trials")
        , m_trialsPerSec(x + 350 + 20, y, 75, 20, "Trials/Second")
        , m_FPS(x + 200, y, 70, 20, "FPS")
        , m_learn_rate(x, y + 44, w, 20, "Learning Rate")
        , m_error_graph(x, y + 82, w, h - 82, "Error")
        , m_resize_box(x + w / 2, y + 82, 1, h - 82)
    {
        m_learn_rate.step(0.00001, 0.0001);
        m_learn_rate.bounds(0.0, 0.01);
        m_learn_rate.value(s_workers[0]->m_learn_rate);
        m_learn_rate.callback([](Fl_Widget* w, void* data) {
            auto self = (Fl_Counter*)w;
            s_workers[0]->m_learn_rate = (float)self->value();
        });
        m_error_graph.valss.resize(1);
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
    void trialsPerSec_value(double f)
    {
        auto label = fmt::format("{:+.1f}", f);
        m_trialsPerSec.value(label.c_str());
    }
    void FPS_value(double f)
    {
        auto label = fmt::format("{:+.1f}", f);
        m_FPS.value(label.c_str());
    }

    void on_idle()
    {
        static constexpr double FPS = 60.0;
        static constexpr long long NS_PER_MS = (long long)1e6;
        static constexpr long long MS_PER_S = (long long)1e3;
        static constexpr long long NS_PER_S = (long long)1e9;
        static constexpr long long NS_PER_FRAME = (long long)(NS_PER_S / FPS);

        static long prevDelta = 0;
        static long long prevTimer = get_nanos();
        static double smoothed_tps = 0.0;
        static double smoothed_fps = 60.0;

        long long timer = get_nanos();
        long true_delta = (long)(timer - prevTimer);
        long delta = (long)(true_delta + prevDelta);

        if (delta >= NS_PER_FRAME)
        {
            prevDelta = delta % NS_PER_FRAME;
            prevTimer = timer;

            static int previousTrials_value = 0;
            int trials = s_workers[0]->m_trials.load();

            error_value(s_workers[0]->m_err[0]);
            trials_value(trials);

            smoothed_tps = 0.99 * smoothed_tps + 0.01 * (trials - previousTrials_value) * 1e9 / delta;
            trialsPerSec_value(smoothed_tps);

            smoothed_fps = 0.9 * smoothed_fps + 0.1 * NS_PER_S / true_delta;
            FPS_value(smoothed_fps);

            m_error_graph.valss[0].clear();
            std::copy(std::begin(s_workers[0]->m_err),
                      std::end(s_workers[0]->m_err),
                      std::back_inserter(m_error_graph.valss[0]));
            m_error_graph.damage(FL_DAMAGE_ALL);
            m_error_graph.redraw();

            previousTrials_value = trials;
        }
    }

    Fl_Output m_err;
    Fl_Output m_trials;
    Fl_Output m_trialsPerSec;
    Fl_Output m_FPS;
    Fl_Counter m_learn_rate;
    Graph m_error_graph;
    Fl_Box m_resize_box;
};

struct MLStats_Window : Fl_Double_Window
{
    MLStats_Window(int w, int h, const char* label) : Fl_Double_Window(w, h, label), group(10, 10, w - 20, h - 20)
    {
        this->resizable(group);
        this->end();
    }

    MLStats_Group group;
};

struct Turn_Viewer : Fl_Group
{
    Turn_Viewer(int x, int y, int w, int h, const char* label = 0)
        : Fl_Group(x, y, w, h, label)
        , m_actions(x, y, w / 2, h, "Actions")
        , m_cur_turn(x + w / 2, y, w / 2, h, "Current State")
    {
        m_actions.child().callback([](Fl_Widget* w, void* p) { ((Turn_Viewer*)w->parent())->on_player_action(); });
        this->end();
    }

    void on_player_action();

    Margins<Fl_Select_Browser, 0, 0, 5, 18> m_actions;
    Margins<Fl_Browser, 5, 0, 0, 18> m_cur_turn;
};

const std::vector<std::shared_ptr<IModel>>& s_tgroup_models();

struct Model_Explorer : Fl_Double_Window
{
    Model_Explorer(int w, int h, const char* label = 0)
        : Fl_Double_Window(w, h, label)
        , m_sweep(10, 10, w - 20, 20, "Sweep Variable")
        , m_graphscroll(10, 40, w - 20, h - 60)
        , m_pack(0, 0, w, h)
    {
        m_sweep.m.callback(thunkv<Model_Explorer, &Model_Explorer::cb_OnChangeSweep>, this);
        m_graphscroll.box(FL_DOWN_BOX);
        m_graphscroll.type(Fl_Scroll::VERTICAL_ALWAYS);
        m_pack.end();
        m_graphscroll.end();
        this->resizable(&m_graphscroll);
        this->end();
    }

    void resize(int x, int y, int w, int h)
    {
        Fl_Double_Window::resize(x, y, w, h);
        if (m_graphscroll.h() >= m_pack.h()) m_graphscroll.scroll_to(0, 0);
    }

    void set_model(std::shared_ptr<IModel> model, Game& g)
    {
        m_model = std::move(model);
        m_encoded = g.encode();
        m_sweep.m.clear();
        for (auto&& sw : g.input_descs())
            m_sweep.m.add(sw.c_str());
        m_sweep.m.value(0);
        m_sweep.m.damage(FL_DAMAGE_ALL);
        m_sweep.m.redraw();
        auto actions = g.format_actions();

        m_pack.begin();
        for (int i = 0; i < actions.size(); ++i)
        {
            if (i == m_graphs.size())
            {
                m_graphs.emplace_back(std::make_unique<Graph>(0, 0, 100, 100));
                m_graphs[i]->align(FL_ALIGN_TOP | FL_ALIGN_INSIDE);
            }
            m_graphs[i]->max_y = 1.0;
            m_graphs[i]->min_y = 0.0;
            m_graphs[i]->copy_label(actions[i].c_str());
        }
        m_graphs.resize(actions.size());
        m_pack.end();

        cb_OnChangeSweep();
    }

    void cb_OnChangeSweep()
    {
        int v = m_sweep.m.value();
        if (v == -1) return;
        Encoded e = m_encoded;
        auto eval = m_model->make_eval();
        for (auto&& g : m_graphs)
        {
            g->valss.resize(2);
            g->valss[0].resize(0);
            g->valss[1].resize(0);
        }
        for (int x = 0; x < 11; ++x)
        {
            e.data[v] = x * 0.1f;
            m_model->calc(*eval, e, false);

            double min_val = eval->out().min(1.0f);
            for (int i = 0; i < eval->out().size() && i < m_graphs.size(); ++i)
            {
                m_graphs[i]->valss[0].push_back(1.0 - eval->out()[i]);
                m_graphs[i]->valss[1].push_back(1.0 - (eval->out()[i] - min_val));
            }
        }
        for (auto&& g : m_graphs)
        {
            g->damage(FL_DAMAGE_ALL);
            g->redraw();
        }
    }

    Margins<Fl_Choice, 100, 0> m_sweep;
    Fl_Scroll m_graphscroll;
    Fl_Pack m_pack;
    std::vector<std::unique_ptr<Graph>> m_graphs;

    std::shared_ptr<IModel> m_model;
    Encoded m_encoded;
};

struct Game_Group : Fl_Group
{
    Game_Group(int x, int y, int w, int h, const char* label = 0)
        : Fl_Group(x, y, w, h, label)
        , m_turn_viewer(x, y + 22, w, 180)
        , m_gamelog(x, y + 22 + 180, w, (h - 22) - 180 - 17, "Gamelog")
        , m_new_game(x + w - 80, y, 80, 20, "New Game")
        , m_ai_plays_p1(x, y, 50, 20, "AI P1")
        , m_ai_plays_p2(x + 52, y, 50, 20, "AI P2")
        , m_ai_choice(x + 106, y, 100, 20)
        , m_resize_box(x + w / 2 - 20, y + 22 + 180 + 1, 40, (h - 22) / 2 - 180 - 17)
    {
        m_ai_plays_p1.value(1);
        m_ai_plays_p1.callback([](Fl_Widget* w, void*) { ((Game_Group*)w->parent())->update_game(); });
        m_ai_plays_p2.value(1);
        m_ai_plays_p2.callback([](Fl_Widget* w, void*) { ((Game_Group*)w->parent())->update_game(); });
        m_new_game.callback([](Fl_Widget* w, void*) {
            auto& self = *(Game_Group*)w->parent();
            if (0 < self.m_ai_choice.value() || self.m_ai_choice.value() >= s_models_list.models.size())
                self.cur_model = s_workers[0]->clone_model();
            else
                self.cur_model = s_models_list.models[self.m_ai_choice.value()]->clone();

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
        turn.player2_turn = g.player2_turn;
        turn.eval = cur_model->make_eval();
        turn.eval_full = cur_model->make_eval();
        cur_model->calc(*turn.eval, turn.input, false);
        cur_model->calc(*turn.eval_full, turn.input, true);
        m_gamelog.add(("@." + g.format()).c_str());
        m_turn_viewer.m_actions.child().clear();
        auto actions = g.format_actions();
        for (auto&& [k, v] : kv_range(actions))
        {
            m_turn_viewer.m_actions.child().add(v.c_str(), (void*)k);
        }
        m_turn_viewer.m_cur_turn.child().clear();
        for (auto&& l : g.format_public_lines())
        {
            static const char disable_format[] = "@.";
            l.insert(l.begin(), disable_format, disable_format + 2);
            m_turn_viewer.m_cur_turn.child().add(l.c_str());
        }

        s_mexp->set_model(cur_model, g);
    }

    void on_player_action()
    {
        if (turns.empty() || g.cur_result() != Game::Result::playing) return;
        if (m_turn_viewer.m_actions.child().value() == 0) return;

        auto& turn = turns.back();
        turn.chosen_action = m_turn_viewer.m_actions.child().value() - 1;
        m_turn_viewer.m_actions.child().deselect();
        advance_turn();
        update_game();
    }

    void advance_turn()
    {
        auto& turn = turns.back();
        m_gamelog.add(
            fmt::format("@.Turn {}: AI: {} FAI: {}", g.turn + 1, turn.eval->out(), turn.eval_full->out()).c_str());
        m_gamelog.add(fmt::format("@bTurn {}: Action: {}", g.turn + 1, g.format_actions()[turn.chosen_action]).c_str());
        g.advance(turn.chosen_action);
        start_next_turn();
    }

    void update_game()
    {
        if (turns.empty()) return;
        while (g.cur_result() == Game::Result::playing)
        {
            if (!turns.back().player2_turn && !m_ai_plays_p1.value()) break;
            if (turns.back().player2_turn && !m_ai_plays_p2.value()) break;

            auto& turn = turns.back();

            // choose action to take
            turn.take_ai_action();
            advance_turn();
        }
    }

    Turn_Viewer m_turn_viewer;
    Fl_Browser m_gamelog;
    Fl_Button m_new_game;
    Fl_Check_Button m_ai_plays_p1;
    Fl_Check_Button m_ai_plays_p2;
    Fl_Choice m_ai_choice;
    Fl_Box m_resize_box;

    Game g;
    std::vector<Turn> turns;
    std::shared_ptr<IModel> cur_model;
};

void Turn_Viewer::on_player_action() { ((Game_Group*)parent())->on_player_action(); }

std::unique_ptr<IModel> open_model()
{
#if defined(_WIN32)
    HRESULT hr;

    COMDLG_FILTERSPEC rgSpec[] = {
        {L"JSON Files", L"*.json"},
        {L"All Files", L"*.*"},
    };

    winrt::com_ptr<IFileOpenDialog> dialog = winrt::create_instance<IFileOpenDialog>(winrt::guid_of<FileOpenDialog>());

    hr = dialog->SetFileTypes(ARRAYSIZE(rgSpec), rgSpec);
    if (!SUCCEEDED(hr)) return nullptr;

    hr = dialog->Show(NULL);
    if (!SUCCEEDED(hr)) return nullptr;

    winrt::com_ptr<IShellItem> pItem;
    hr = dialog->GetResult(pItem.put());
    if (!SUCCEEDED(hr)) return nullptr;
    PWSTR pszPath;
    hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &pszPath);
    if (!SUCCEEDED(hr)) return nullptr;
    std::wstring p(pszPath);
    CoTaskMemFree(pszPath);

    std::stringstream ss;
    std::ifstream is(p);
    is.get(*ss.rdbuf());

    try
    {
        return load_model(ss.str());
    }
    catch (const std::exception& e)
    {
        fmt::print(L"Failed while loading model from {}: ", p);
        fmt::print("{}\n", e.what());
    }
    catch (const char* e)
    {
        fmt::print(L"Failed while loading model from {}: ", p);
        fmt::print("{}\n", e);
    }
#endif
    return nullptr;
}

void save_model(IModel& model)
{
#if defined(_WIN32)
    HRESULT hr;

    COMDLG_FILTERSPEC rgSpec[] = {
        {L"JSON Files", L"*.json"},
        {L"All Files", L"*.*"},
    };

    winrt::com_ptr<IFileSaveDialog> dialog = winrt::create_instance<IFileSaveDialog>(winrt::guid_of<FileSaveDialog>());

    hr = dialog->SetFileTypes(ARRAYSIZE(rgSpec), rgSpec);
    if (!SUCCEEDED(hr)) return;

    hr = dialog->Show(NULL);
    if (!SUCCEEDED(hr)) return;

    winrt::com_ptr<IShellItem> pItem;
    hr = dialog->GetResult(pItem.put());
    if (!SUCCEEDED(hr)) return;
    UINT i;
    hr = dialog->GetFileTypeIndex(&i);
    if (!SUCCEEDED(hr)) return;

    PWSTR pszPath;
    hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &pszPath);
    if (!SUCCEEDED(hr)) return;

    std::wstring path = pszPath;
    if (i == 1 && (path.size() < 6 || path.substr(path.size() - 5) != L".json"))
    {
        path += L".json";
    }
    CoTaskMemFree(pszPath);

    try
    {
        StringBuffer s;
        RJWriter wr(s);
        model.serialize(wr);

        std::ofstream os(path);
        os.write(s.GetString(), s.GetSize());
        fmt::print(L"Wrote {}\n", path);
    }
    catch (std::exception& e)
    {
        fmt::print(L"Failed to write {}: ", path);
        fmt::print("{}", e.what());
    }
#endif
}

template<class Container, class F>
void sync_browser(Fl_Browser& b, const Container& c, F transformer)
{
    bool changed = false;
    if (b.size() != c.size())
    {
        b.clear();
        for (auto&& x : c)
            b.add(transformer(x).c_str());
        changed = true;
    }
    else
    {
        for (size_t i = 0; i < c.size(); ++i)
        {
            std::string s = transformer(c[i]);
            if (b.text((int)i) != s)
            {
                b.text((int)i, s.c_str());
                changed = true;
            }
        }
    }
    if (changed)
    {
        b.damage(FL_DAMAGE_ALL);
        b.redraw();
    }
}

// template<class... Args>
// struct Event
//{
//    template<class T, void (T::*F)(Args...)>
//    void on(T* t)
//    {
//        m_observers.emplace_back(t, [](void* v, Args... args) { (((T*)v)->*F)(static_cast<Args&&>(args)...); });
//    }
//
//    void fire(Args&&... args)
//    {
//        for (auto p : m_observers)
//            p.second(p.first, args...);
//    }
//
// private:
//    std::vector<std::pair<void*, void (*)(void*, Args...)>> m_observers;
//};

struct Rename_Modal_Window : Fl_Window
{
    void cb_Ok()
    {
        this->hide();
        if (m_ok_cb) m_ok_cb(this, m_ok_u);
    }
    void cb_Cancel() { this->hide(); }

    const char* value() { return m_new_name.value(); }
    void value(const char* v) { m_new_name.value(v); }
    void on_submit(Fl_Callback* cb, void* u)
    {
        m_ok_cb = cb;
        m_ok_u = u;
    }

    Rename_Modal_Window(const char* label = 0)
        : Fl_Window(314, 88, label)
        , m_ok(111, 47, 99, 30, "Ok")
        , m_cancel(220, 47, 86, 30, "Cancel")
        , m_new_name(98, 7, 208, 30, "New Name")
    {
        m_ok.callback(thunkv<Rename_Modal_Window, &Rename_Modal_Window::cb_Ok>, this);
        m_cancel.callback(thunkv<Rename_Modal_Window, &Rename_Modal_Window::cb_Cancel>, this);

        this->set_modal();
        this->end();
    }

    Fl_Return_Button m_ok;
    Fl_Button m_cancel;
    Fl_Input m_new_name;
    Fl_Callback* m_ok_cb = nullptr;
    void* m_ok_u = nullptr;
};

struct SWindows
{
    Fl_Window* help;
    MLStats_Window* worker0;
    Fl_Window* tournament;
    Fl_Window* play;
};
static SWindows s_windows;

struct Manager_Window : Fl_Double_Window
{
    struct Workers_Browser : Fl_Group
    {
        Workers_Browser(int x, int y, int w, int h, const char* label = 0)
            : Fl_Group(x, y, w, h, label)
            , m_freeze(x, y, w / 2, 26, "Freeze")
            , m_thaw(x + w / 2, y, w / 2, 26, "Thaw")
            , m_browser(x, y + 26, w, h - 26 - 15, "Workers")
        {
            this->resizable(m_browser);
            this->end();
        }

        Fl_Button m_freeze, m_thaw;
        Fl_Hold_Browser m_browser;
    };
    struct Tournament_Browser : Fl_Group
    {
        Tournament_Browser(int x, int y, int w, int h, const char* label = 0)
            : Fl_Group(x, y, w, h, label)
            , m_tfreeze(x, y, w / 2, 26, "Freeze")
            , m_remove(x + w / 2, y, w / 2, 26, "---")
            , m_browser(x, y + 26, w, h - 26 - 15, "Tournament")
        {
            this->resizable(m_browser);
            this->end();
        }

        Fl_Button m_tfreeze, m_remove;
        Fl_Multi_Browser m_browser;
    };

    void on_idle()
    {
        for (auto&& [k, v] : kv_range(s_workers))
        {
            if (k >= m_workers.child().m_browser.size())
                m_workers.child().m_browser.add(v->model_name().c_str());
            else
                m_workers.child().m_browser.text((int)k + 1, v->model_name().c_str());
        }
    }

    void cb_Freeze()
    {
        int line = m_workers.child().m_browser.value();
        if (line > s_workers.size()) return;
        auto& b = m_models.child();

        if (line == 0)
        {
            for (auto&& w : s_workers)
            {
                s_models_list.push_back(w->clone_model());
            }
        }
        else
        {
            s_models_list.push_back(s_workers[line - 1]->clone_model());
        }
    }
    void cb_Thaw()
    {
        int w_line = m_workers.child().m_browser.value();
        if (w_line == 0 || m_workers.child().m_browser.size() != s_workers.size()) return;

        auto& b = m_models.child();
        int b_line = b.value();
        if (b_line == 0) return;

        if (b.size() != s_models_list.models.size()) return;
        auto&& m = s_models_list.models[b_line - 1];

        s_workers[w_line - 1]->replace_model(m->clone());
        m_workers.child().m_browser.text(w_line, m->name().c_str());
    }
    void cb_New() { }
    void cb_Open()
    {
        auto model = open_model();
        if (!model) return;
        s_models_list.push_back(std::move(model));
    }
    void cb_Save()
    {
        auto& b = m_models.child();
        int b_line = b.value();
        if (b_line == 0) return;

        if (b.size() != s_models_list.models.size()) return;
        save_model(*s_models_list.models[b_line - 1]);
    }
    void cb_Rename()
    {
        if (m_modal_rename.visible())
        {
            m_modal_rename.activate();
        }
        else
        {
            auto& b = m_models.child();
            int b_line = b.value();
            if (b_line == 0) return;

            if (b.size() != s_models_list.models.size()) return;
            m_renaming = s_models_list.models[b_line - 1];
            m_modal_rename.value(m_renaming->root_name().c_str());
            m_modal_rename.show();
        }
        m_modal_rename.m_new_name.take_focus();
    }
    void cb_Rename_Ok()
    {
        auto new_model = m_renaming->clone();
        new_model->set_root_name(m_modal_rename.value());
        auto& s_models = s_models_list.models;

        auto it = std::find(s_models.begin(), s_models.end(), m_renaming);
        if (it == s_models.end())
        {
            s_models_list.push_back(std::move(new_model));
        }
        else
        {
            s_models_list.replace((int)(it - s_models.begin()), std::move(new_model));
        }
    }
    std::shared_ptr<IModel> m_renaming;
    void cb_Delete()
    {
        auto& b = m_models.child();
        if (b.size() != s_models_list.models.size()) return;
        for (int i = 0; i < b.size();)
        {
            if (b.selected(i + 1))
            {
                s_models_list.erase(i);
            }
            else
            {
                ++i;
            }
        }
    }
    void cb_TFreeze()
    {
        for (auto&& m : s_tgroup_models())
            s_models_list.push_back(m);
    }
    void cb_TRemove() { }

    Manager_Window(int w, int h, const char* name = "MLCard Manager - MLCard")
        : Fl_Double_Window(w, h, name)
        , m_menu_bar(0, 0, w, 30)
        , m_models(0, 0, w / 2, h, "Models")
        , m_workers(w / 2, 0, w / 2, h / 2)
        , m_tourny(w / 2, h / 2, w / 2, h / 2)
        , m_modal_rename("Rename Model")
    {
        m_menu_bar.menu(s_menu_items);
        m_modal_rename.on_submit(thunkv<Manager_Window, &Manager_Window::cb_Rename_Ok>, this);
        m_workers.child().m_freeze.callback((Fl_Callback*)::thunkv<Manager_Window, &Manager_Window::cb_Freeze>, this);
        m_workers.child().m_thaw.callback((Fl_Callback*)::thunkv<Manager_Window, &Manager_Window::cb_Thaw>, this);
        m_tourny.child().m_tfreeze.callback((Fl_Callback*)::thunkv<Manager_Window, &Manager_Window::cb_TFreeze>, this);
        m_tourny.child().m_remove.callback((Fl_Callback*)::thunkv<Manager_Window, &Manager_Window::cb_TRemove>, this);
        this->callback([](Fl_Widget* p, void*) { std::exit(0); });
        this->resizable(this);
        this->end();
    }

    void cb_show_play() { s_windows.play->show(); }
    void cb_show_help() { s_windows.help->show(); }
    void cb_show_tournament() { s_windows.tournament->show(); }
    void cb_show_worker0() { s_windows.worker0->show(); }
    void cb_show_explorer() { s_mexp->show(); }

    Fl_Menu_Bar m_menu_bar;
    Margins<Fl_Multi_Browser, 10, 35, 5, 25> m_models;
    Margins<Workers_Browser, 5, 35, 10, 5> m_workers;
    Margins<Tournament_Browser, 5, 5, 10, 10> m_tourny;
    Rename_Modal_Window m_modal_rename;

    using MW = Manager_Window;

    static inline const Fl_Menu_Item s_menu_items[] = {
        {"&File", 0, 0, 0, 64, (uchar)FL_NORMAL_LABEL, 0, 14, 0},
        {"&New Model", 0x4006e, thunk1<MW, &MW::cb_New>, 0, 0, (uchar)FL_NORMAL_LABEL, 0, 14, 0},
        {"&Open", 0x4006f, thunk1<MW, &MW::cb_Open>, 0, 0, (uchar)FL_NORMAL_LABEL, 0, 14, 0},
        {"&Save", 0x40073, thunk1<MW, &MW::cb_Save>, 0, 0, (uchar)FL_NORMAL_LABEL, 0, 14, 0},
        {"&Rename", 0xffbf, thunk1<MW, &MW::cb_Rename>, 0, 0, (uchar)FL_NORMAL_LABEL, 0, 14, 0},
        {"&Delete", 0xffff, thunk1<MW, &MW::cb_Delete>, 0, 0, (uchar)FL_NORMAL_LABEL, 0, 14, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0},
        {"&Windows", 0, 0, 0, 64, (uchar)FL_NORMAL_LABEL, 0, 14, 0},
        {"&Play", 0, thunk1<MW, &MW::cb_show_play>, 0, 0, (uchar)FL_NORMAL_LABEL, 0, 14, 0},
        {"&Explorer", 0, thunk1<MW, &MW::cb_show_explorer>, 0, 0, (uchar)FL_NORMAL_LABEL, 0, 14, 0},
        {"&Worker0", 0, thunk1<MW, &MW::cb_show_worker0>, 0, 0, (uchar)FL_NORMAL_LABEL, 0, 14, 0},
        {"&Tournament", 0, thunk1<MW, &MW::cb_show_tournament>, 0, 0, (uchar)FL_NORMAL_LABEL, 0, 14, 0},
        {"&How to Play", 0, thunk1<MW, &MW::cb_show_help>, 0, 0, (uchar)FL_NORMAL_LABEL, 0, 14, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0}};
};

struct TournamentUIModelsList
{
    const std::vector<std::shared_ptr<IModel>>& models() { return m_models; }
    const std::vector<std::string>& names() { return m_names; }

    void assign(const std::vector<std::shared_ptr<IModel>>& new_values);

private:
    std::vector<std::shared_ptr<IModel>> m_models;
    std::vector<std::string> m_names;
};

struct Tournament_Group : Fl_Group
{
    struct WinStats
    {
        int p1;
        int p2;
        int tie;
    };

    Tournament_Group(int x, int y, int w, int h, const char* label = 0)
        : Fl_Group(x, y, w, h, label)
        , m_button(x, y, w, 20, "Pause/Unpause Tournament")
        , m_browser(x, y + 30, w, h - 10 - 30 - 17, "AI Comparison")
    {
        m_button.callback([](Fl_Widget* w, void*) { ((Tournament_Group*)w->parent())->run_tournament(); });
        this->resizable(&m_browser);
        this->end();
    }
    ~Tournament_Group() { m_worker.exit(); }

    void run_tournament() { m_worker.toggle_pause(); }

    template<class T>
    static void erase_ns(std::vector<T>& v, const std::vector<bool>& to_erase)
    {
        int i = 0;
        size_t e = v.size();
        for (;; ++i)
        {
            if (i == e) return;
            if (to_erase[i]) break;
        }

        int j = i + 1;
        for (; j < e; ++j)
        {
            if (!to_erase[j]) v[i++] = std::move(v[j]);
        }
        v.erase(v.begin() + i, v.end());
    }

    static std::vector<std::pair<double, int>> winrates(const std::vector<std::vector<WinStats>>& stats)
    {
        std::vector<std::pair<double, int>> ret;
        auto num_models = stats.size();
        for (int i = 0; i < num_models; ++i)
        {
            int winpct_count = 0;
            double winpct = 0;
            for (int j = 0; j < num_models; ++j)
            {
                if (j == i) continue;
                const auto& d = stats[i][j];
                if (d.p1 + d.p2 == 0) continue;
                winpct += 100.0 * d.p1 / (d.p1 + d.p2);
                winpct_count++;

                const auto& d2 = stats[j][i];
                if (d2.p1 + d2.p2 == 0) continue;
                winpct += 100.0 * d2.p2 / (d2.p1 + d2.p2);
                winpct_count++;
            }
            ret.emplace_back(winpct / std::max(winpct_count, 1), i);
        }
        return ret;
    }

    struct Worker
    {
        void exit()
        {
            if (th.joinable())
            {
                exit_worker = true;
                unpause();
                th.join();
            }
        }
        void pause() { paused = true; }
        void unpause()
        {
            paused = false;
            if (!th.joinable())
            {
                th = std::thread(&Worker::work, this);
            }
            else
            {
                std::lock_guard lk(m);
                m_cv.notify_all();
            }
        }
        void toggle_pause()
        {
            if (paused || !th.joinable())
                unpause();
            else
                pause();
        }

        void work()
        {
            std::mutex local_mutex;
            std::vector<std::vector<WinStats>> local_data;
            std::vector<std::shared_ptr<IModel>> local_models;
            while (!exit_worker)
            {
                {
                    std::unique_lock lk(m);
                    m_cv.wait(lk, [this]() { return !paused; });
                    if (restart)
                    {
                        local_data = data;
                        local_models = models;
                        restart = false;
                    }
                    else
                    {
                        data = local_data;
                        models = local_models;
                        updated = true;
                    }
                }

                int sz = (int)local_models.size();

                std::for_each(int_iterator{0}, int_iterator{sz * sz}, [&](int n) {
                    static constexpr auto max_samples = 250;
                    int i = n / sz;
                    int j = n % sz;
                    auto& ld = local_data[i][j];
                    std::unique_lock guard(local_mutex);
                    while (ld.p1 + ld.p2 + ld.tie < max_samples)
                    {
                        guard.unlock();
                        if (restart) return;
                        if (paused)
                        {
                            std::unique_lock lk(m);
                            m_cv.wait(lk, [this]() { return !paused; });
                        }
                        auto [x, y] = run_100(*local_models[i], *local_models[j]);
                        guard.lock();
                        ld.p1 += x;
                        ld.p2 += y;
                        ld.tie += 100 - x - y;
                        if (!updated)
                        {
                            std::unique_lock lk(m);
                            data = local_data;
                            models = local_models;
                            updated = true;
                        }
                    }
                });

                if (restart) continue;

                // All competitions have sufficient samples. Update models.
                static const size_t target_tournament = 12; // must be larger than s_workers.size()
                auto new_size = std::min(target_tournament, s_workers.size() + local_models.size());

                if (target_tournament < s_workers.size() + local_models.size())
                {
                    auto wrs = winrates(local_data);
                    std::sort(wrs.begin(), wrs.end());
                    auto num_to_erase = s_workers.size() + local_models.size() - target_tournament;
                    std::vector<bool> to_erase(local_models.size(), false);
                    for (int i = 0; i < num_to_erase; ++i)
                        to_erase[wrs[i].second] = true;
                    erase_ns(local_data, to_erase);
                    erase_ns(local_models, to_erase);
                    for (auto&& x : local_data)
                        erase_ns(x, to_erase);
                }

                for (auto&& x : local_data)
                    x.resize(new_size);
                for (size_t i = local_data.size(); i < new_size; ++i)
                    local_data.emplace_back(new_size);
                for (auto&& w : s_workers)
                    local_models.push_back(w->clone_model());
            }
        }

        std::pair<int, int> run_100(IModel& m1, IModel& m2)
        {
            Game g;
            std::vector<Turn> turns;
            int p1_wins = 0;
            int p2_wins = 0;

            for (int x = 0; x < 100; ++x)
            {
                g.init();
                turns.clear();
                turns.reserve(40);

                while (g.cur_result() == Game::Result::playing)
                {
                    turns.emplace_back();
                    auto& turn = turns.back();
                    turn.input = g.encode();
                    if (g.player2_turn)
                    {
                        turn.eval = m2.make_eval();
                        m2.calc(*turn.eval, turn.input, false);
                    }
                    else
                    {
                        turn.eval = m1.make_eval();
                        m1.calc(*turn.eval, turn.input, false);
                    }

                    // choose action to take
                    turn.take_ai_action();

                    g.advance(turn.chosen_action);
                }
                if (g.cur_result() == Game::Result::p1_win)
                    ++p1_wins;
                else if (g.cur_result() == Game::Result::p2_win)
                    ++p2_wins;
            }
            return {p1_wins, p2_wins};
        }

        std::atomic<bool> updated = false;
        std::atomic<bool> restart = false;
        std::atomic<bool> exit_worker = false;
        std::mutex m;
        std::condition_variable m_cv;
        std::atomic<bool> paused = false;
        std::vector<std::vector<WinStats>> data;
        std::vector<std::shared_ptr<IModel>> models;
        std::thread th;
    } m_worker;

    TournamentUIModelsList m_ui_models;

    void on_idle()
    {
        if (!m_worker.updated) return;
        std::lock_guard<std::mutex> lk(m_worker.m);
        m_worker.updated = false;
        m_ui_models.assign(m_worker.models);
        m_browser.clear();
        static int widths[] = {200, 100, 0};
        m_browser.column_widths(widths);
        m_browser.column_char('\t');
        m_browser.add("Tournament Results:");
        for (auto&& wr : winrates(m_worker.data))
        {
            m_browser.add(fmt::format("{} overall:\t{:.2f}%", m_ui_models.names()[wr.second], wr.first).c_str());
        }
        m_browser.add("");

        for (int i = 0; i < m_ui_models.names().size(); ++i)
        {
            for (int j = 0; j < m_ui_models.names().size(); ++j)
            {
                const auto& d = m_worker.data[i][j];
                if (d.p1 + d.p2 > 0)
                    m_browser.add(fmt::format("{} vs {}:\t{} vs {}:\t{:.2f}%",
                                              m_ui_models.names()[i],
                                              m_ui_models.names()[j],
                                              d.p1,
                                              d.p2,
                                              100.0 * d.p1 / (d.p1 + d.p2))
                                      .c_str());
            }
        }
    }

    Fl_Button m_button;
    Fl_Browser m_browser;
};

const std::vector<std::shared_ptr<IModel>>& s_tgroup_models() { return s_tgroup->m_ui_models.models(); }

void TournamentUIModelsList::assign(const std::vector<std::shared_ptr<IModel>>& new_values)
{
    if (m_models == new_values) return;

    m_models.resize(new_values.size());
    m_names.resize(new_values.size());
    for (int i = 0; i < m_models.size(); ++i)
    {
        if (m_models[i] != new_values[i])
        {
            m_models[i] = new_values[i];
            m_names[i] = m_models[i]->name();
        }
    }
}

std::string escape_menu_item(const std::string& str)
{
    std::string r;
    for (char ch : str)
    {
        if (ch == '\\' || ch == '&' || ch == '/' || ch == '_') r.push_back('\\');
        r.push_back(ch);
    }
    return r;
}

void ModelsList::push_back(std::shared_ptr<IModel> m)
{
    s_gamegroup->m_ai_choice.add(escape_menu_item(m->name()).c_str());
    s_mwin->m_models.child().add(m->name().c_str());
    models.push_back(std::move(m));
}
void ModelsList::replace(int i, std::shared_ptr<IModel> m)
{
    s_gamegroup->m_ai_choice.replace(i, escape_menu_item(m->name()).c_str());
    s_mwin->m_models.child().text(i + 1, m->name().c_str());
    models[i] = std::move(m);
}
void ModelsList::erase(int i)
{
    s_gamegroup->m_ai_choice.remove(i);
    s_mwin->m_models.child().remove(i + 1);
    models.erase(models.begin() + i);
}
void ModelsList::clear()
{
    s_gamegroup->m_ai_choice.clear();
    s_mwin->m_models.child().clear();
    models.clear();
}

static void on_timeout(void* v)
{
    s_windows.worker0->group.on_idle();
    s_tgroup->on_idle();
    s_mwin->on_idle();
    Fl::repeat_timeout(1.0 / 120, on_timeout);
};

int main(int argc, char* argv[])
{
    srand((unsigned int)time(NULL));
    s_workers.push_back(std::make_unique<Worker>());
    s_workers.push_back(std::make_unique<Worker>());
    s_workers.push_back(std::make_unique<Worker>());
    s_workers.push_back(std::make_unique<Worker>());
    s_workers[0]->replace_model(make_model(default_model_dims(), "bgA"));
    auto bgB = default_model_dims();
    bgB.children["card_out"].dims = {40, 30, 20};
    bgB.children["l"].dims = {40, 10};
    bgB.children["l"].type = "ReLUCascade";
    s_workers[1]->replace_model(make_model(bgB, "clC432/4/1"));
    bgB = default_model_dims();
    bgB.children["l"].dims = {40, 20};
    bgB.children["l"].type = "ReLUCascade";
    s_workers[2]->replace_model(make_model(bgB, "lC40/20"));
    bgB = default_model_dims();
    bgB.children["l"].dims = {48, 30};
    bgB.children["l"].type = "ReLUCascade";
    s_workers[3]->replace_model(make_model(bgB, "lC48/30"));

    auto win = std::make_unique<MLStats_Window>(490, 400, "Worker 0 - MLCard");
    s_windows.worker0 = win.get();

    auto winx = std::make_unique<Fl_Double_Window>(600, 700, "MLCard Tournament - MLCard");
    winx->begin();
    s_tgroup = new Tournament_Group(10, 10, winx->w() - 20, winx->h() - 20);
    winx->end();
    winx->resizable(new Fl_Box(10, 10, winx->w() - 20, winx->h() - 20));
    s_windows.tournament = winx.get();
    winx->show();

    auto win2 = std::make_unique<Fl_Double_Window>(600, 700, "MLCard Game - MLCard");
    win2->begin();
    s_gamegroup = new Game_Group(10, 10, win2->w() - 20, win2->h() - 20);
    win2->end();
    win2->resizable(new Fl_Box(10, 10, win2->w() - 20, win2->h() - 20));
    s_windows.play = win2.get();

    auto win3 = std::make_unique<Fl_Double_Window>(600, 700, "How to Play - MLCard");
    win3->begin();
    auto hv = new Fl_Help_View(10, 10, win3->w() - 20, win3->h() - 20);
    hv->value(Game::help_html("index"));
    hv->textsize(16);
    hv->link([](Fl_Widget* w, const char* uri) -> const char* {
        auto self = (Fl_Help_View*)w;
        self->value(Game::help_html(uri));
        return NULL;
    });
    win3->end();
    win3->resizable(new Fl_Box(10, 10, win3->w() - 20, win3->h() - 20));
    s_windows.help = win3.get();

    s_mexp = new Model_Explorer(600, 700, "Explorer");

    s_mwin = new Manager_Window(600, 700);
    s_mwin->show(argc, argv);

    Fl::add_timeout(1.0 / 60, on_timeout);

    for (auto&& w : s_workers)
        w->start();
    auto rc = Fl::run();
    for (auto&& w : s_workers)
        w->join();
    return rc;
}
