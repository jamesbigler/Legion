
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

#ifndef LR_GUI_LEGION_DISPLAY_HPP_
#define LR_GUI_LEGION_DISPLAY_HPP_

#include <Legion/Objects/Display/IDisplay.hpp>
#include <Legion/Common/Util/Timer.hpp>
#include <gui/RenderThread.hpp>
#include <string>

namespace lr
{

class LegionDisplay : public legion::IDisplay
{
public:
    LegionDisplay( legion::Context*  context,
                   lr::RenderThread* render_thread,
                   const char*       filename );
    
    FrameType getUpdateFrameType()       { return BYTE;  }
    FrameType getCompleteFrameType()     { return FLOAT; }
    
    void setUpdateCount( unsigned m_update_count );

    void beginScene   ( const std::string& scene_name );
    void beginFrame   ( const legion::Index2& resolution );
    bool updateFrame  ( const float* fpix, const Byte* cpix );
    void completeFrame( const float* fpix, const Byte* cpix );
private:
    static const int s_result_width = 29;
    static const int s_field_width  = 21;
    static void printTime( const std::string& phase, double duration );

    RenderThread*     m_render_thread;

    std::string       m_scene_name;
    unsigned          m_update_count;
    unsigned          m_cur_update;
    legion::Index2    m_resolution;
    std::string       m_filename;

    legion::Timer     m_display_timer;
    legion::Timer     m_preprocess_timer;
    legion::Timer     m_render_timer;
    legion::Timer     m_postprocess_timer;
};

}

#endif // LR_GUI_LEGION_DISPLAY_HPP_
