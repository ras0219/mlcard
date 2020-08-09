#pragma once

class Fl_Widget;

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
