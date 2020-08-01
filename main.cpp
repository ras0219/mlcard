#define NOMINMAX
//#define VEC_ENABLE_CHECKS

#include "ai_play.h"
#include "game.h"
#include "model.h"
#include "rjwriter.h"
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
#include <FL/Fl_Return_Button.H>
#include <FL/Fl_Select_Browser.H>
#include <FL/Fl_Valuator.H>
#include <FL/Fl_Widget.H>
#include <FL/fl_draw.H>
#include <atomic>
#include <fmt/format.h>
#include <fstream>
#include <memory>
#include <mutex>
#include <rapidjson/writer.h>
#include <shobjidl_core.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <thread>
#include <time.h>
#include <valarray>
#include <vector>
#include <winrt/base.h>
#pragma comment(lib, "windowsapp")

using namespace rapidjson;

std::vector<std::unique_ptr<Worker>> s_workers;
struct MLStats_Group* s_mlgroup;
struct Game_Group* s_gamegroup;
struct Tournament_Group* s_tgroup;

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
            s_workers[0]->m_learn_rate = self->value();
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
    void trialsPerSec_value(float f)
    {
        auto label = fmt::format("{:+.1f}", f);
        m_trialsPerSec.value(label.c_str());
    }
    void FPS_value(float f)
    {
        auto label = fmt::format("{:+.1f}", f);
        m_FPS.value(label.c_str());
    }

    void on_idle()
    {
        static constexpr double FPS = 60.0;
        static constexpr long long NS_PER_MS = 1e6;
        static constexpr long long MS_PER_S = 1e3;
        static constexpr long long NS_PER_S = 1e9;
        static constexpr long long NS_PER_FRAME = NS_PER_S / FPS;

        static long prevDelta = 0;
        static long long prevTimer = get_nanos();
        static float smoothed_tps = 0.0;
        static double smoothed_fps = 60.0;

        long long timer = get_nanos();
        long delta = timer - prevTimer + prevDelta;
        long true_delta = timer - prevTimer;

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

const std::vector<std::shared_ptr<IModel>>& s_tgroup_models();

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
            if (-1 == self.m_ai_choice.value() || self.m_ai_choice.value() >= s_tgroup_models().size())
                self.cur_model = s_workers[0]->clone_model();
            else
                self.cur_model = s_tgroup_models()[self.m_ai_choice.value()]->clone();

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
        turn.eval = cur_model->make_eval();
        turn.eval_full = cur_model->make_eval();
        cur_model->calc(*turn.eval, turn.input, false);
        cur_model->calc(*turn.eval_full, turn.input, true);
        m_gamelog.add(("@." + g.format()).c_str());
        m_turn_viewer.m_actions.clear();
        auto actions = g.format_actions();
        for (size_t i = 0; i < actions.size(); ++i)
        {
            m_turn_viewer.m_actions.add(actions[i].c_str(), (void*)i);
        }
        m_turn_viewer.m_cur_turn.clear();
        m_turn_viewer.m_cur_turn.format_char(0);
        for (auto&& l : g.format_public_lines())
        {
            m_turn_viewer.m_cur_turn.add(l.c_str());
        }
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
                                  turn.eval->out(),
                                  turn.eval_full->out(),
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
    std::unique_ptr<IModel> cur_model;
};

void Turn_Viewer::on_player_action() { ((Game_Group*)parent())->on_player_action(); }

void open_cb(Fl_Widget* w, void* v)
{
    HRESULT hr;

    COMDLG_FILTERSPEC rgSpec[] = {
        {L"JSON Files", L"*.json"},
        {L"All Files", L"*.*"},
    };

    winrt::com_ptr<IFileOpenDialog> dialog = winrt::create_instance<IFileOpenDialog>(winrt::guid_of<FileOpenDialog>());

    hr = dialog->SetFileTypes(ARRAYSIZE(rgSpec), rgSpec);
    if (!SUCCEEDED(hr)) return;

    hr = dialog->Show(NULL);
    if (!SUCCEEDED(hr)) return;

    winrt::com_ptr<IShellItem> pItem;
    hr = dialog->GetResult(pItem.put());
    if (!SUCCEEDED(hr)) return;
    PWSTR pszPath;
    hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &pszPath);
    if (!SUCCEEDED(hr)) return;
    std::wstring p(pszPath);
    CoTaskMemFree(pszPath);

    std::stringstream ss;
    std::ifstream is(p);
    is.get(*ss.rdbuf());

    try
    {
        auto model = load_model(ss.str());
        s_workers[0]->replace_model(std::move(model));
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
}

void save_cb(Fl_Widget* w, void* v)
{
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

    auto path = fmt::format(L"{}{}", pszPath, i == 1 ? L".json" : L"");
    CoTaskMemFree(pszPath);

    try
    {
        StringBuffer s;
        RJWriter wr(s);
        s_workers[0]->serialize_model(wr);

        std::ofstream os(path);
        os.write(s.GetString(), s.GetSize());
        fmt::print(L"Wrote {}\n", path);
    }
    catch (std::exception& e)
    {
        fmt::print(L"Failed to write {}: ", path);
        fmt::print("{}", e.what());
    }
}

template<class T, int pad_left, int pad_top, int pad_right = pad_left, int pad_bottom = pad_top>
struct Margins : Fl_Group
{
    Margins(int x, int y, int w, int h, const char* label = 0)
        : Fl_Group(x, y, w, h), m(x + pad_left, y + pad_top, w - pad_left - pad_right, h - pad_top - pad_bottom, label)
    {
        this->resizable(m);
        this->end();
    }

    T& child() { return m; }

    T m;
};

struct Models_List
{
    std::mutex m;
    std::vector<std::shared_ptr<IModel>> models;
} s_models_list;

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

template<class T, void (T::*F)()>
static void thunk0(Fl_Widget* w, void*)
{
    (((T*)w)->*F)();
}

template<class T, void (T::*F)()>
static void thunk1(Fl_Widget* w, void*)
{
    (((T*)w->parent())->*F)();
}

template<class T, void (T::*F)()>
static void thunkv(Fl_Widget*, void* w)
{
    (((T*)w)->*F)();
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

template<class T>
struct LockGuarded;

template<class T>
struct UniqueLocked
{
    constexpr UniqueLocked() : t(nullptr) { }
    constexpr UniqueLocked(UniqueLocked&& u) : t(u.t) { u.t = nullptr; }
    constexpr UniqueLocked& operator=(UniqueLocked&& u)
    {
        clear();
        t = u.t;
        u.t = nullptr;
    }
    ~UniqueLocked()
    {
        if (t) t->m.unlock();
    }

    T& operator*() { return t->t; }
    T* operator->() { return &t->t; }

    void clear()
    {
        if (t)
        {
            t->m.unlock();
            t = nullptr;
        }
    }

    friend struct LockGuarded<T>;

private:
    UniqueLocked(LockGuarded<T>* t) : t(t) { }
    LockGuarded<T>* t;
};

template<class T>
struct LockGuard
{
    LockGuard(LockGuarded<T>& t) : t(&t) { t.m.lock(); }
    LockGuard(const LockGuard&) = delete;
    LockGuard(LockGuard&&) = delete;
    LockGuard& operator=(const LockGuard&) = delete;
    LockGuard& operator=(LockGuard&&) = delete;
    ~LockGuard() { t->m.unlock(); }

    T& operator*() { return t->t; }
    T* operator->() { return &t->t; }

    friend struct LockGuarded<T>;

private:
    LockGuarded<T>* t;
};

template<class T>
struct LockGuarded
{
    UniqueLocked<T> lock()
    {
        m.lock();
        return this;
    }

    friend struct UniqueLocked<T>;
    friend struct LockGuard<T>;

    using Guard = LockGuard<T>;

private:
    std::mutex m;
    T t;
};

LockGuarded<std::vector<std::shared_ptr<IModel>>> s_tourny_list;

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
            update();
            this->resizable(m_browser);
            this->end();
        }

        void update()
        {
            sync_browser(this->m_browser, s_workers, [](const std::unique_ptr<Worker>& p) { return p->model_name(); });
        }

        Fl_Button m_freeze, m_thaw;
        Fl_Hold_Browser m_browser;
    };
    struct Tournament_Browser : Fl_Group
    {
        Tournament_Browser(int x, int y, int w, int h, const char* label = 0)
            : Fl_Group(x, y, w, h, label)
            , m_add(x, y, w / 2, 26, "Add")
            , m_remove(x + w / 2, y, w / 2, 26, "Remove")
            , m_browser(x, y + 26, w, h - 26 - 15, "Tournament")
        {
            this->resizable(m_browser);
            this->end();
        }

        Fl_Button m_add, m_remove;
        Fl_Multi_Browser m_browser;
    };

    void cb_Freeze()
    {
        int line = m_workers.child().m_browser.value();
        if (line > s_workers.size()) return;
        auto& b = m_models.child();

        std::lock_guard<std::mutex> lk(s_models_list.m);
        if (line == 0)
        {
            for (auto&& w : s_workers)
            {
                s_models_list.models.push_back(w->clone_model());
                b.add(s_models_list.models.back()->name().c_str());
            }
        }
        else
        {
            s_models_list.models.push_back(s_workers[line - 1]->clone_model());
            b.add(s_models_list.models.back()->name().c_str());
        }
        b.damage(FL_DAMAGE_ALL);
        b.redraw();
    }
    void cb_Thaw()
    {
        int w_line = m_workers.child().m_browser.value();
        if (w_line == 0 || m_workers.child().m_browser.size() != s_workers.size()) return;

        auto& b = m_models.child();
        int b_line = b.value();
        if (b_line == 0) return;

        std::shared_ptr<IModel> m;
        {
            std::lock_guard<std::mutex> lk(s_models_list.m);
            if (b.size() != s_models_list.models.size()) return;
            m = s_models_list.models[b_line - 1];
        }
        s_workers[w_line - 1]->replace_model(m->clone());
        m_workers.child().m_browser.text(w_line, m->name().c_str());
        m_workers.child().m_browser.damage(FL_DAMAGE_ALL);
        m_workers.child().m_browser.redraw();
    }
    void cb_New() { }
    void cb_Open() { }
    void cb_Save() { }
    void cb_Rename()
    {
        if (m_modal_rename.visible())
        {
            m_modal_rename.activate();
            return;
        }
        else
        {
            auto& b = m_models.child();
            int b_line = b.value();
            if (b_line == 0) return;

            {
                std::lock_guard<std::mutex> lk(s_models_list.m);
                if (b.size() != s_models_list.models.size()) return;
                m_renaming = s_models_list.models[b_line - 1];
            }
            m_modal_rename.value(m_renaming->root_name().c_str());
            m_modal_rename.show();
        }
    }
    void cb_Rename_Ok()
    {
        std::lock_guard<std::mutex> lk(s_models_list.m);
        auto new_model = m_renaming->clone();
        new_model->set_root_name(m_modal_rename.value());
        auto& s_models = s_models_list.models;
        auto it = std::find(s_models.begin(), s_models.end(), m_renaming);
        if (it == s_models.end())
        {
            m_models.m.add(new_model->name().c_str());
            m_models.m.damage(FL_DAMAGE_ALL);
            m_models.m.redraw();
            s_models.push_back(std::move(new_model));
        }
        else
        {
            m_models.m.text((int)(it - s_models.begin() + 1), new_model->name().c_str());
            m_models.m.damage(FL_DAMAGE_ALL);
            m_models.m.redraw();
            *it = std::move(new_model);
        }
    }
    std::shared_ptr<IModel> m_renaming;
    void cb_Delete()
    {
        auto& b = m_models.child();
        std::lock_guard lk(s_models_list.m);
        if (b.size() != s_models_list.models.size()) return;
        for (int i = 0; i < b.size();)
        {
            if (b.selected(i + 1))
            {
                s_models_list.models.erase(s_models_list.models.begin() + i);
                b.remove(i + 1);
            }
            else
            {
                ++i;
            }
        }
        b.damage(FL_DAMAGE_ALL);
        b.redraw();
    }
    void cb_TAdd()
    {
        auto& b = m_models.child();
        std::lock_guard lk(s_models_list.m);
        if (s_models_list.models.size() != b.size()) return;
        LockGuard tourny_list(s_tourny_list);
        for (int i = 0; i < b.size(); ++i)
        {
            if (b.selected(i + 1))
            {
                tourny_list->push_back(s_models_list.models[i]);
            }
        }
    }
    void cb_TRemove() { }

    Manager_Window(int w, int h, const char* name = "Manager")
        : Fl_Double_Window(w, h, name)
        , m_menu_bar(0, 0, w, 30)
        , m_models(0, 0, w / 2, h, "Models")
        , m_workers(w / 2, 0, w / 2, h / 2)
        , m_tourny(w / 2, h / 2, w / 2, h / 2)
        , m_modal_rename("Rename Model")
    {
        m_menu_bar.menu(s_menu_items);
        {
            std::lock_guard<std::mutex> lk(s_models_list.m);
            update();
        }
        m_modal_rename.on_submit(thunkv<Manager_Window, &Manager_Window::cb_Rename_Ok>, this);
        m_workers.child().m_freeze.callback((Fl_Callback*)::thunkv<Manager_Window, &Manager_Window::cb_Freeze>, this);
        m_workers.child().m_thaw.callback((Fl_Callback*)::thunkv<Manager_Window, &Manager_Window::cb_Thaw>, this);
        m_tourny.child().m_add.callback((Fl_Callback*)::thunkv<Manager_Window, &Manager_Window::cb_TAdd>, this);
        m_tourny.child().m_remove.callback((Fl_Callback*)::thunkv<Manager_Window, &Manager_Window::cb_TRemove>, this);
        this->resizable(this);
        this->end();
    }

    void update()
    {
        sync_browser(
            m_models.child(), s_models_list.models, [](const std::shared_ptr<IModel>& model) { return model->name(); });
    }

    Fl_Menu_Bar m_menu_bar;
    Margins<Fl_Multi_Browser, 10, 35, 5, 25> m_models;
    Margins<Workers_Browser, 5, 35, 10, 5> m_workers;
    Margins<Tournament_Browser, 5, 5, 10, 10> m_tourny;
    Rename_Modal_Window m_modal_rename;

    static inline const Fl_Menu_Item s_menu_items[] = {
        {"&File", 0, 0, 0, 64, (uchar)FL_NORMAL_LABEL, 0, 14, 0},
        {"&New Model", 0x4006e, thunk1<Manager_Window, &cb_New>, 0, 0, (uchar)FL_NORMAL_LABEL, 0, 14, 0},
        {"&Open", 0x4006f, thunk1<Manager_Window, &cb_Open>, 0, 0, (uchar)FL_NORMAL_LABEL, 0, 14, 0},
        {"&Save", 0x40073, thunk1<Manager_Window, &cb_Save>, 0, 0, (uchar)FL_NORMAL_LABEL, 0, 14, 0},
        {"&Rename", 0xffbf, thunk1<Manager_Window, &cb_Rename>, 0, 0, (uchar)FL_NORMAL_LABEL, 0, 14, 0},
        {"&Delete", 0xffff, thunk1<Manager_Window, &cb_Delete>, 0, 0, (uchar)FL_NORMAL_LABEL, 0, 14, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0}};
};

struct Tournament_Group : Fl_Group
{
    Tournament_Group(int x, int y, int w, int h, const char* label = 0)
        : Fl_Group(x, y, w, h, label)
        , m_button(x, y, w, 20, "Run tournament")
        , m_browser(x, y + 30, w, h - 10 - 30 - 17, "AI comparison")
    {
        m_button.callback([](Fl_Widget* w, void*) { ((Tournament_Group*)w->parent())->run_tournament(); });
        this->resizable(&m_browser);
        this->end();
    }
    ~Tournament_Group()
    {
        exit_worker = true;
        if (th.joinable()) th.join();
    }

    void run_tournament()
    {
        std::lock_guard<std::mutex> lk(m);
        updated = false;
        restart = true;
        static const size_t target_tournament = 10; // must be larger than s_workers.size()
        auto new_size = std::min(target_tournament, s_workers.size() + models.size());

        int active = s_gamegroup->m_ai_choice.value();
        auto active_p = active < 0 || active >= models.size() ? nullptr : models[active].get();
        if (target_tournament < s_workers.size() + models.size())
        {
            auto wrs = winrates(data);
            std::sort(wrs.begin(), wrs.end());
            auto num_to_erase = s_workers.size() + models.size() - target_tournament;
            std::vector<bool> to_erase(models.size(), false);
            for (int i = 0; i < num_to_erase; ++i)
                to_erase[wrs[i].second] = true;
            erase_ns(data, to_erase);
            erase_ns(models, to_erase);
            for (auto&& x : data)
                erase_ns(x, to_erase);
        }

        for (auto&& x : data)
            x.resize(new_size);
        for (size_t i = data.size(); i < new_size; ++i)
            data.emplace_back(new_size);
        for (auto&& w : s_workers)
            models.push_back(w->clone_model());

        num_models = models.size();
        model_names.resize(models.size());
        for (int i = 0; i < models.size(); ++i)
        {
            model_names[i] = models[i]->name();
        }

        s_gamegroup->m_ai_choice.clear();
        for (auto&& n : model_names)
            s_gamegroup->m_ai_choice.add(n.c_str());
        for (int i = 0; i < models.size(); ++i)
        {
            if (models[i].get() == active_p)
            {
                s_gamegroup->m_ai_choice.value(i);
                break;
            }
        }
        s_gamegroup->m_ai_choice.damage(FL_DAMAGE_ALL);
        s_gamegroup->m_ai_choice.redraw();

        if (!th.joinable())
        {
            th = std::thread(&Tournament_Group::work, this);
        }
    }

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

    static std::vector<std::pair<double, int>> winrates(const std::vector<std::vector<std::pair<int, int>>>& stats)
    {
        std::vector<std::pair<double, int>> ret;
        auto num_models = stats.size();
        for (int i = 0; i < num_models; ++i)
        {
            double winpct = 0;
            for (int j = 0; j < num_models; ++j)
            {
                if (j == i) continue;
                const auto& d = stats[i][j];
                if (d.first + d.second == 0) continue;
                winpct += 100.0 * d.first / (d.first + d.second);

                const auto& d2 = stats[j][i];
                if (d2.first + d2.second == 0) continue;
                winpct += 100.0 * d2.second / (d2.first + d2.second);
            }
            ret.emplace_back(winpct / 2 / (num_models - 1), i);
        }
        return ret;
    }

    void work()
    {
        std::vector<std::vector<std::pair<int, int>>> local_data;
        std::vector<std::shared_ptr<IModel>> local_models;
        int i = 0;
        int j = 0;
        while (!exit_worker)
        {
            if (restart || !updated)
            {
                std::lock_guard<std::mutex> lk(m);
                if (restart)
                {
                    local_data = data;
                    local_models = models;
                    restart = false;
                }
                else if (!updated)
                {
                    data = local_data;
                    updated = true;
                }
            }
            i++;
            if (i >= local_data.size())
            {
                i = 0;
                j++;
            }
            if (j >= local_data.size())
            {
                j = 0;
            }
            auto [x, y] = run_100(*local_models[i], *local_models[j]);
            local_data[i][j].first += x;
            local_data[i][j].second += y;
        }
    }

    std::atomic<bool> updated = false;
    std::atomic<bool> restart = false;
    std::atomic<bool> exit_worker = false;
    std::vector<std::string> model_names;
    std::mutex m;
    std::vector<std::vector<std::pair<int, int>>> data;
    size_t num_models = 0;
    std::vector<std::shared_ptr<IModel>> models;
    std::thread th;

    void on_idle()
    {
        if (!updated) return;
        std::lock_guard<std::mutex> lk(m);
        updated = false;
        m_browser.clear();
        static int widths[] = {150, 100, 0};
        m_browser.column_widths(widths);
        m_browser.column_char('\t');
        m_browser.add("Tournament Results:");
        for (auto&& wr : winrates(data))
        {
            m_browser.add(fmt::format("{} overall:\t{}%", model_names[wr.second], wr.first).c_str());
        }
        m_browser.add("");

        for (int i = 0; i < num_models; ++i)
        {
            for (int j = 0; j < num_models; ++j)
            {
                const auto& d = data[i][j];
                if (d.first + d.second > 0)
                    m_browser.add(fmt::format("{} vs {}:\t{} vs {}:\t{}%",
                                              model_names[i],
                                              model_names[j],
                                              d.first,
                                              d.second,
                                              100.0 * d.first / (d.first + d.second))
                                      .c_str());
            }
        }

        m_browser.damage(FL_DAMAGE_ALL);
        m_browser.redraw();
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

    Fl_Button m_button;
    Fl_Browser m_browser;
};

const std::vector<std::shared_ptr<IModel>>& s_tgroup_models() { return s_tgroup->models; }

int main(int argc, char* argv[])
{
    srand((unsigned int)time(NULL));
    s_workers.push_back(std::make_unique<Worker>());
    s_workers.push_back(std::make_unique<Worker>());
    s_workers.push_back(std::make_unique<Worker>());
    s_workers.push_back(std::make_unique<Worker>());
    s_workers.push_back(std::make_unique<Worker>());
    s_workers[0]->replace_model(make_model(default_model_dims(), "bgA"));
    s_workers[1]->replace_model(make_model(default_model_dims(), "bgB"));
    s_workers[2]->replace_model(make_model(medium_model_dims(), "llA"));
    s_workers[3]->replace_model(make_model(medium_model_dims(), "llB"));
    s_workers[4]->replace_model(make_model(small_model_dims(), "sml"));

    auto win = std::make_unique<Fl_Double_Window>(490, 400, "MLCard");
    win->begin();
    Fl_Menu_Bar menu_bar(0, 0, win->w(), 30);
    menu_bar.add("&File/&Open", "^o", &open_cb);
    menu_bar.add("&File/&Save", "^s", &save_cb);
    s_mlgroup = new MLStats_Group(10, 40, win->w() - 20, win->h() - 50);
    win->end();
    win->resizable(new Fl_Box(10, 10, win->w() - 20, win->h() - 20));
    win->show(argc, argv);

    auto winx = std::make_unique<Fl_Double_Window>(600, 700, "Crossplay");
    winx->begin();
    s_tgroup = new Tournament_Group(10, 10, winx->w() - 20, winx->h() - 20);
    winx->end();
    winx->resizable(new Fl_Box(10, 10, winx->w() - 20, winx->h() - 20));
    winx->show();

    auto win2 = std::make_unique<Fl_Double_Window>(600, 700, "MLCard Game");
    win2->begin();
    s_gamegroup = new Game_Group(10, 10, win2->w() - 20, win2->h() - 20);
    win2->end();
    win2->resizable(new Fl_Box(10, 10, win2->w() - 20, win2->h() - 20));
    win2->show();

    auto win3 = std::make_unique<Fl_Double_Window>(600, 700, "How to Play");
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
    win3->show();

    (new Manager_Window(600, 700))->show();

    Fl::add_idle([](void* v) {
        s_mlgroup->on_idle();
        s_tgroup->on_idle();
    });

    for (auto&& w : s_workers)
        w->start();
    auto rc = Fl::run();
    for (auto&& w : s_workers)
        w->join();
    return rc;
}
