#include "graph.h"
#include "kv_range.h"
#include <FL/fl_draw.H>
#include <fmt/format.h>

void Graph::draw()
{
    this->draw_box(FL_FLAT_BOX, FL_BACKGROUND_COLOR);
    this->draw_box(FL_FRAME, FL_BACKGROUND_COLOR);
    this->draw_label();

    if (valss.empty()) return;

    auto n = min_y;
    auto m = max_y;
    if (!fixed_y_axis)
    {
        for (auto&& vals : valss)
        {
            for (auto&& val : vals)
            {
                n = std::min(n, val);
                m = std::max(m, val);
            }
        }
    }

    fl_draw(fmt::format("{:.2f}", m).c_str(), this->x(), this->y() + fl_height() - fl_descent());
    fl_draw(fmt::format("{:.2f}", n).c_str(), this->x(), this->y() + this->h() - fl_descent());

    auto color = fl_color();
    for (auto&& [x, vals] : kv_range(valss))
    {
        if (vals.size() < 2) return;
        auto inc_w = this->w() * 1.0 / (vals.size() - 1);
        if (x == 0)
            fl_color(0);
        else if (x == 1)
            fl_color(14);
        else
            fl_color(11);
        fl_begin_line();
        for (int i = 0; i < vals.size(); ++i)
        {
            fl_vertex(this->x() + i * inc_w, (int)(this->y() - (vals[i] - m) * this->h() / (m - n)));
        }
        fl_end_line();
    }
    fl_color(color);
}
