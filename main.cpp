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
#include <FL/Fl_Counter.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_File_Chooser.H>
#include <FL/Fl_Help_View.H>
#include <FL/Fl_Hold_Browser.H>
#include <FL/Fl_Menu_Bar.H>
#include <FL/Fl_Multi_Browser.H>
#include <FL/Fl_Output.H>
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
#include <valarray>
#include <vector>
#include <winrt/base.h>
#pragma comment(lib, "windowsapp")

using namespace rapidjson;

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
        , m_turn_viewer(x, y + 22, w, 180)
        , m_gamelog(x, y + 22 + 180, w, (h - 22) - 180 - 17, "Gamelog")
        , m_new_game(x + w - 80, y, 80, 20, "New Game")
        , m_ai_plays_p1(x, y, 50, 20, "AI P1")
        , m_ai_plays_p2(x + 52, y, 50, 20, "AI P2")
        , m_resize_box(x + w / 2 - 20, y + 22 + 180 + 1, 40, (h - 22) / 2 - 180 - 17)
    {
        m_ai_plays_p1.value(1);
        m_ai_plays_p1.callback([](Fl_Widget* w, void*) { ((Game_Group*)w->parent())->update_game(); });
        m_ai_plays_p2.value(1);
        m_ai_plays_p2.callback([](Fl_Widget* w, void*) { ((Game_Group*)w->parent())->update_game(); });
        m_new_game.callback([](Fl_Widget* w, void*) {
            auto& self = *(Game_Group*)w->parent();
            {
                std::lock_guard<std::mutex> lk(s_mutex);
                self.cur_model = s_model->clone();
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
    Fl_Box m_resize_box;

    Game g;
    std::vector<Turn> turns;
    std::unique_ptr<IModel> cur_model;
};

void Turn_Viewer::on_player_action() { ((Game_Group*)parent())->on_player_action(); }

MLStats_Group* s_mlgroup;
Game_Group* s_gamegroup;

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
        {
            std::lock_guard<std::mutex> lk(s_mutex);
            delete s_model;
            s_model = model.release();
            s_replace_model = true;
            s_updated = false;
        }
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

    StringBuffer s;
    RJWriter wr(s);
    {
        std::lock_guard<std::mutex> lk(s_mutex);
        s_model->serialize(wr);
    }

    try
    {
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

int main(int argc, char* argv[])
{
    srand((unsigned int)time(NULL));
    s_model = make_model().release();

    auto win = std::make_unique<Fl_Double_Window>(490, 400, "MLCard");
    win->begin();
    Fl_Menu_Bar menu_bar(0, 0, win->w(), 30);
    menu_bar.add("&File/&Open", "^o", &open_cb);
    menu_bar.add("&File/&Save", "^s", &save_cb);
    s_mlgroup = new MLStats_Group(10, 40, win->w() - 20, win->h() - 50);
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

    std::thread th(worker);
    auto rc = Fl::run();
    s_worker_exit = true;
    th.join();
    return rc;
}
