#pragma once

#include <FL/Fl_Widget.H>

#include <vector>

struct Graph : Fl_Widget
{
    Graph(int x, int y, int w, int h, const char* label = 0) : Fl_Widget(x, y, w, h, label) { }
    virtual void draw() override;

    bool fixed_y_axis = false;
    double min_y = 0.0;
    double max_y = 0.0001;

    std::vector<std::vector<double>> valss;
};
