#pragma once

#include <FL/Fl_Group.H>

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
