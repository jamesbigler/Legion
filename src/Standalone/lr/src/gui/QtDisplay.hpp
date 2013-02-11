
// Copyright (C) 2011 R. Keith Morley
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// (MIT/X11 License)

#ifndef LR_GUI_QT_DISPLAY_HPP_
#define LR_GUI_QT_DISPLAY_HPP_

#include <Legion/Objects/Display/IDisplay.hpp>
#include <string>

namespace lr
{

class QtDisplay : public legion::IDisplay
{
public:
    QtDisplay( legion::Context* context );
    ~QtDisplay();

    void beginScene    ( const std::string& scene_name );
    void setUpdateCount( unsigned m_update_count );
    void beginFrame    ();
    void updateFrame   ( const legion::Index2& res, const float* pixels );
    void completeFrame ( const legion::Index2& res, const float* pixels );
private:
    static const int  s_field_width = 28;

    std::string       m_scene_name;
    unsigned          m_update_count;
    unsigned          m_cur_update;
};

} // end namespace lr

#endif // LR_GUI_QT_DISPLAY_HPP_
